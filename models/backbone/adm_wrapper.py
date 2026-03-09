"""
models/backbone/adm_wrapper.py
------------------------------
MSSP-specific ADM backbone wrapper.

Key difference from DRIFT_new's adm_wrapper:
  - Provides probe() method for single-step multi-scale probing
  - Does NOT do DDIM inversion (no invert() method needed)
  - Uses full 1000-step alphas_cumprod for forward noising
  - Calls UNet only once per timestep (5 times total)

Algorithm (MSSP):
    for t_i in T_set = {200, 400, 600, 800, 999}:
        x_{t_i} = sqrt(αbar_{t_i}) * x₀ + sqrt(1-αbar_{t_i}) * ε_i
        ε̂_{t_i} = UNet_ADM(x_{t_i}, t_i)   # single forward pass
        r_i = x_{t_i} - ε̂_{t_i}             # score residual

The forward noising uses the original 1000-step linear beta schedule,
computed analytically (no diffusion object required for this).
"""

from __future__ import annotations

import sys
import os
import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrapping — add guided-diffusion to sys.path
# ---------------------------------------------------------------------------
_PREPROCESSING_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),  # models/backbone/
    "..", "..", "..",           # up to AIGCDetectBenchmark-main/
    "preprocessing_model",
)
_PREPROCESSING_MODEL_DIR = os.path.abspath(_PREPROCESSING_MODEL_DIR)

if _PREPROCESSING_MODEL_DIR not in sys.path:
    sys.path.insert(0, _PREPROCESSING_MODEL_DIR)


# ---------------------------------------------------------------------------
# Beta schedule — linear schedule matching ADM checkpoint
# ---------------------------------------------------------------------------

def _make_linear_alphas_cumprod(num_steps: int = 1000) -> np.ndarray:
    """Compute the full 1000-step alphas_cumprod using ADM's linear beta schedule.

    ADM uses: beta_t = linspace(0.0001, 0.02, 1000)
    This matches the guided-diffusion get_named_beta_schedule("linear", 1000).

    Returns:
        np.ndarray of shape (num_steps,), values in (0, 1).
    """
    betas = np.linspace(0.0001, 0.02, num_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas_cumprod.astype(np.float32)


# ---------------------------------------------------------------------------
# MSSPBackbone
# ---------------------------------------------------------------------------

class MSSPBackbone:
    """MSSP-specific wrapper around the ADM UNet.

    Provides probe(): add noise at fixed levels, run single UNet forward,
    return residuals. No DDIM inversion required.

    Args:
        model_path: Absolute path to ADM checkpoint (.pt file).
            Pass "" or "mock" for mock mode (no checkpoint needed).
        device: PyTorch device string.
        image_size: Spatial resolution (default 256).
        probe_timesteps: List of noise timesteps in [0, 999].
            Corresponds to T_set in the MSSP paper.
            T_set = {200, 400, 600, 800, 1000} → [200, 400, 600, 800, 999].
        noise_seed: Base random seed for deterministic noise generation.
            Each timestep t uses seed = noise_seed + t.

    Example::

        backbone = MSSPBackbone(
            model_path="/checkpoints/256x256_diffusion_uncond.pt",
            device="cuda",
        )
        # Returns dict: {200: {x_t, eps_hat, residual}, 400: {...}, ...}
        results = backbone.probe(x0)
    """

    DEFAULT_TIMESTEPS = [200, 400, 600, 800, 999]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        image_size: int = 256,
        probe_timesteps: Optional[List[int]] = None,
        noise_seed: int = 42,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.image_size = image_size
        self.probe_timesteps = probe_timesteps or self.DEFAULT_TIMESTEPS
        self.noise_seed = noise_seed

        self._mock_mode: bool = model_path in ("", "mock")
        self._model = None

        # Full 1000-step alphas_cumprod (computed analytically, no model needed)
        _acp = _make_linear_alphas_cumprod(1000)
        self._alphas_cumprod = torch.tensor(_acp, dtype=torch.float32)

        if not self._mock_mode:
            self._load_model()
        else:
            logger.info(
                "MSSPBackbone initialised in MOCK mode — "
                "probe() will return Gaussian noise."
            )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the ADM UNet from self.model_path.

        Uses guided_diffusion.script_util. The diffusion object is not
        needed for MSSP (only the UNet is used), but we create it for
        completeness and to validate model architecture.
        """
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"ADM checkpoint not found at '{self.model_path}'. "
                "Download from: "
                "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
                "256x256_diffusion_uncond.pt"
            )

        try:
            from guided_diffusion import script_util
        except ImportError as exc:
            raise ImportError(
                "Cannot import guided_diffusion. "
                f"Expected package at: {_PREPROCESSING_MODEL_DIR}\n"
                f"Original error: {exc}"
            ) from exc

        logger.info("Loading ADM model from '%s' ...", self.model_path)

        # MSSP only needs the UNet, not the diffusion scheduler.
        # Load with minimal diffusion config (1000 steps, not DDIM-spaced).
        model_kwargs = script_util.model_and_diffusion_defaults()
        model_kwargs.update(
            {
                "image_size": self.image_size,
                "timestep_respacing": "1000",   # full schedule
                "num_channels": 256,
                "num_head_channels": 64,
                "num_heads": -1,
                "attention_resolutions": "32,16,8",
                "resblock_updown": True,
                "use_fp16": (self.device.type == "cuda"),
                "learn_sigma": True,
            }
        )

        self._model, _ = script_util.create_model_and_diffusion(**model_kwargs)

        state_dict = torch.load(self.model_path, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)

        if model_kwargs.get("use_fp16") and self.device.type == "cuda":
            self._model.convert_to_fp16()

        self._model.eval()
        logger.info(
            "ADM model loaded. image_size=%d, timesteps=%s",
            self.image_size,
            self.probe_timesteps,
        )

    # ------------------------------------------------------------------
    # Core probe method
    # ------------------------------------------------------------------

    @torch.no_grad()
    def probe(
        self,
        x0: torch.Tensor,
        timesteps: Optional[List[int]] = None,
        noise_seed: Optional[int] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Single-step multi-scale probing.

        For each timestep t in T_set:
          1. Add noise: x_t = sqrt(αbar_t) * x0 + sqrt(1-αbar_t) * ε
          2. Single UNet forward: ε̂ = UNet(x_t, t)
          3. Residual: r = x_t - ε̂

        The noise ε is deterministic (seed = base_seed + t) to ensure
        reproducibility across different images.

        Args:
            x0: Input images [B, 3, H, W] in [-1, 1] range.
            timesteps: Override the probe timesteps (0-999). Defaults to
                self.probe_timesteps = [200, 400, 600, 800, 999].
            noise_seed: Override the noise seed. Defaults to self.noise_seed.

        Returns:
            Dict mapping timestep t → {
                'x_t': noised image [B, 3, H, W],
                'eps_hat': predicted noise [B, 3, H, W],
                'residual': score residual x_t - eps_hat [B, 3, H, W],
            }
            All tensors are on CPU.
        """
        if self._mock_mode:
            return self._mock_probe(x0, timesteps=timesteps, noise_seed=noise_seed)

        if self._model is None:
            raise RuntimeError("Model not loaded.")

        timesteps = timesteps or self.probe_timesteps
        seed = noise_seed if noise_seed is not None else self.noise_seed

        x0 = x0.to(self.device)
        # Convert to float32 for forward noising (model may use fp16 internally)
        x0_f32 = x0.float()
        B = x0.shape[0]

        # Move alphas_cumprod to device
        acp = self._alphas_cumprod.to(self.device)

        results: Dict[int, Dict[str, torch.Tensor]] = {}

        for t in timesteps:
            t_clamped = min(max(t, 0), 999)
            alpha_bar = acp[t_clamped]  # scalar

            # Deterministic noise per timestep (different seed per t)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed + t_clamped)
            eps = torch.randn(
                x0_f32.shape,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )

            # Forward noising: x_t = sqrt(αbar) * x0 + sqrt(1-αbar) * ε
            x_t = alpha_bar.sqrt() * x0_f32 + (1.0 - alpha_bar).sqrt() * eps

            # Single-step UNet forward (raw UNet call, not SpacedDiffusion)
            t_tensor = torch.full((B,), t_clamped, device=self.device, dtype=torch.long)
            model_output = self._model(x_t, t_tensor)

            # Handle learn_sigma=True: output has 6 channels (eps + variance)
            if model_output.shape[1] == 6:
                eps_hat, _ = model_output.float().chunk(2, dim=1)
            else:
                eps_hat = model_output.float()

            # Score residual
            residual = x_t - eps_hat

            results[t_clamped] = {
                "x_t": x_t.cpu(),
                "eps_hat": eps_hat.cpu(),
                "residual": residual.cpu(),
            }

        return results

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _mock_probe(
        self,
        x0: torch.Tensor,
        timesteps: Optional[List[int]] = None,
        noise_seed: Optional[int] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Mock probe for unit-testing without a real checkpoint.

        Returns random tensors with the same shape as real probe outputs.
        Features computed from these will be random (not meaningful),
        but all downstream code runs without error.
        """
        timesteps = timesteps or self.probe_timesteps
        seed = noise_seed if noise_seed is not None else self.noise_seed
        results: Dict[int, Dict[str, torch.Tensor]] = {}

        acp = self._alphas_cumprod  # on CPU for mock mode

        for t in timesteps:
            t_clamped = min(max(t, 0), 999)
            alpha_bar = acp[t_clamped]

            generator = torch.Generator()
            generator.manual_seed(seed + t_clamped)

            # Simulate forward noising with Gaussian noise
            x0_cpu = x0.cpu().float()
            eps = torch.randn(x0_cpu.shape, generator=generator)
            x_t = alpha_bar.sqrt() * x0_cpu + (1.0 - alpha_bar).sqrt() * eps

            # Mock UNet: return different Gaussian noise as eps_hat
            generator2 = torch.Generator()
            generator2.manual_seed(seed + t_clamped + 10000)
            eps_hat = torch.randn(x0_cpu.shape, generator=generator2)

            residual = x_t - eps_hat

            results[t_clamped] = {
                "x_t": x_t,
                "eps_hat": eps_hat,
                "residual": residual,
            }

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    def __repr__(self) -> str:
        return (
            f"MSSPBackbone("
            f"model_path={self.model_path!r}, "
            f"device={str(self.device)!r}, "
            f"image_size={self.image_size}, "
            f"probe_timesteps={self.probe_timesteps}, "
            f"mock={self._mock_mode}"
            f")"
        )
