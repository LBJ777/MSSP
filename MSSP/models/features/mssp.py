"""
models/features/mssp.py
-----------------------
MSSP Feature Extractor: 185-dimensional descriptor from multi-scale
single-step score residuals.

Feature layout (185D total):
    F_norm  [5D]  : relative residual norm at each of 5 noise scales
    F_psd  [120D] : radial PSD in 8 frequency bands × 3 channels × 5 scales
    F_stat  [60D] : per-channel moments (mean, std, skew, kurt) × 3 channels × 5 scales

Algorithm:
    for t_i in T_set = {200, 400, 600, 800, 999}:
        r_i = probe_results[t_i]['residual']   # [B, 3, H, W]
        x_t = probe_results[t_i]['x_t']        # [B, 3, H, W]

        F_norm[i]  = ||r_i||₂ / ||x_t||₂              # 1D → total 5D
        F_psd[i]   = radial_PSD(r_i, n_bands=8)        # 24D → total 120D
        F_stat[i]  = moments(r_i, per_channel=True)    # 12D → total 60D

VAE 8px artifact mapping (for Test 2 in validation script):
    For H=256, the f=H/8=32 spatial frequency corresponds to
    normalized radial frequency 32/256 = 0.125.
    With 8 bands over [0, 0.5], the band edges are:
        [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5]
    f=0.125 is the start of band index 2 (0-based), so channels at t=400-600
    in F_psd[bands 2-3] should show elevated power for SD images.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from .base import FeatureExtractor


class MSSPFeatureExtractor(FeatureExtractor):
    """185-dimensional MSSP feature extractor.

    Extracts features from probe() output without any additional UNet calls.

    Args:
        backbone: MSSPBackbone instance used to compute probe() residuals.
        probe_timesteps: Timesteps to probe. Must match backbone.probe_timesteps.
        n_freq_bands: Number of radial frequency bands for PSD features.
            Default 8 → 8 × 3 channels = 24D per scale.
        normalize_features: If True, L2-normalize the final feature vector.

    Feature dimension:
        5 + (n_freq_bands × 3 × 5) + (4 × 3 × 5)
        = 5 + 120 + 60 = 185D  (with n_freq_bands=8)
    """

    VAE_FREQ = 0.125        # f=H/8 normalized, maps to n_freq_bands=8 band index 2
    VAE_BAND_IDX = 2        # which band contains the VAE 8px artifact

    def __init__(
        self,
        backbone,
        probe_timesteps: Optional[List[int]] = None,
        n_freq_bands: int = 8,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.probe_timesteps = probe_timesteps or backbone.probe_timesteps
        self.n_freq_bands = n_freq_bands
        self.normalize_features = normalize_features

        # Pre-compute feature dimension
        n_scales = len(self.probe_timesteps)
        self._feature_dim = (
            n_scales                          # F_norm: 5D
            + n_scales * n_freq_bands * 3     # F_psd: 5×8×3=120D
            + n_scales * 4 * 3                # F_stat: 5×4×3=60D
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    # ------------------------------------------------------------------
    # Main extract interface
    # ------------------------------------------------------------------

    def extract(
        self,
        x0: torch.Tensor,
        intermediates=None,  # unused, kept for base class compatibility
    ) -> torch.Tensor:
        """Extract 185D MSSP features from input images.

        Args:
            x0: Input images [B, 3, H, W] in [-1, 1] range.
            intermediates: Ignored (MSSP does not use DDIM intermediates).

        Returns:
            Feature tensor [B, 185].
        """
        # Run probe: 5 single-step UNet forwards
        probe_results = self.backbone.probe(x0, timesteps=self.probe_timesteps)

        f_norm_list = []   # will be [B, 1] per scale
        f_psd_list = []    # will be [B, n_freq_bands*3] per scale
        f_stat_list = []   # will be [B, 4*3] per scale

        for t in self.probe_timesteps:
            t_key = min(max(t, 0), 999)
            r_i = probe_results[t_key]["residual"].float()  # [B, 3, H, W]
            x_t = probe_results[t_key]["x_t"].float()       # [B, 3, H, W]

            # F_norm: relative residual norm
            B_cur = r_i.shape[0]
            r_norm = r_i.reshape(B_cur, -1).norm(dim=1)    # [B]
            x_norm = x_t.reshape(B_cur, -1).norm(dim=1).clamp(min=1e-8)
            rel_norm = (r_norm / x_norm).unsqueeze(1)        # [B, 1]
            f_norm_list.append(rel_norm)

            # F_psd: radial PSD in n_freq_bands, per channel
            psd_feat = self._compute_psd_features(r_i)      # [B, n_freq_bands*3]
            f_psd_list.append(psd_feat)

            # F_stat: statistical moments per channel
            stat_feat = self._compute_statistical_moments(r_i)  # [B, 4*3]
            f_stat_list.append(stat_feat)

        # Concatenate all scales
        f_norm = torch.cat(f_norm_list, dim=1)   # [B, n_scales]
        f_psd = torch.cat(f_psd_list, dim=1)     # [B, n_scales * n_freq_bands * 3]
        f_stat = torch.cat(f_stat_list, dim=1)   # [B, n_scales * 4 * 3]

        features = torch.cat([f_norm, f_psd, f_stat], dim=1)  # [B, 185]

        # Clamp NaN/Inf to robust values
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        if self.normalize_features:
            norm = features.norm(dim=1, keepdim=True).clamp(min=1e-8)
            features = features / norm

        self.validate_output(features)
        return features

    # ------------------------------------------------------------------
    # PSD feature computation
    # ------------------------------------------------------------------

    def _compute_psd_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute radial PSD in n_freq_bands for each of 3 channels.

        Args:
            x: [B, 3, H, W] residual tensor.

        Returns:
            [B, n_freq_bands * 3] tensor of log-scaled band power.
        """
        B, C, H, W = x.shape
        channel_features = []

        # Pre-compute radial frequency map (same for all batches and channels)
        radial_map = self._get_radial_freq_map(H, W // 2 + 1, device=x.device)  # [H, W//2+1]

        for c in range(C):
            channel = x[:, c, :, :]  # [B, H, W]
            fft = torch.fft.rfft2(channel)  # [B, H, W//2+1] complex
            power = fft.real ** 2 + fft.imag ** 2  # [B, H, W//2+1]

            band_powers = self._band_average(power, radial_map)  # [B, n_freq_bands]
            channel_features.append(band_powers)

        return torch.cat(channel_features, dim=1)  # [B, n_freq_bands * 3]

    def _get_radial_freq_map(self, H: int, W_half: int, device) -> torch.Tensor:
        """Compute normalized radial frequency for each FFT bin.

        Returns:
            [H, W_half] float tensor with values in [0, 0.5].
        """
        h_idx = torch.arange(H, device=device, dtype=torch.float32)
        w_idx = torch.arange(W_half, device=device, dtype=torch.float32)

        # Fold frequency: u_eff = min(u, H-u) to get positive frequencies
        h_folded = torch.minimum(h_idx, H - h_idx)

        hh, ww = torch.meshgrid(h_folded, w_idx, indexing="ij")  # [H, W_half]
        radial = torch.sqrt(hh ** 2 + ww ** 2) / H  # normalize by H → [0, 0.5]
        return radial  # [H, W_half]

    def _band_average(
        self,
        power: torch.Tensor,
        radial_map: torch.Tensor,
    ) -> torch.Tensor:
        """Average log power in each radial frequency band.

        Args:
            power: [B, H, W_half] power spectrum.
            radial_map: [H, W_half] normalized radial frequencies.

        Returns:
            [B, n_freq_bands] log-averaged band powers.
        """
        B = power.shape[0]
        band_edges = torch.linspace(0.0, 0.5, self.n_freq_bands + 1, device=power.device)
        band_powers = []

        for i in range(self.n_freq_bands):
            lo, hi = band_edges[i], band_edges[i + 1]
            mask = (radial_map >= lo) & (radial_map < hi)  # [H, W_half]

            if mask.sum() == 0:
                band_powers.append(torch.zeros(B, device=power.device))
            else:
                # Mean power in band, then log-scale
                masked_power = power[:, mask]  # [B, n_pixels_in_band]
                mean_power = masked_power.mean(dim=1)  # [B]
                log_power = torch.log1p(mean_power)    # log(1+x) for stability
                band_powers.append(log_power)

        return torch.stack(band_powers, dim=1)  # [B, n_freq_bands]

    # ------------------------------------------------------------------
    # Statistical moment computation
    # ------------------------------------------------------------------

    def _compute_statistical_moments(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean, std, skewness, kurtosis per channel.

        Args:
            x: [B, 3, H, W] residual tensor.

        Returns:
            [B, 4 * 3 = 12] tensor of per-channel moments.
        """
        B, C, H, W = x.shape
        moments = []

        for c in range(C):
            channel = x[:, c, :, :].reshape(B, -1)  # [B, H*W]
            mean = channel.mean(dim=1)               # [B]
            std = channel.std(dim=1).clamp(min=1e-8) # [B]

            z = (channel - mean.unsqueeze(1)) / std.unsqueeze(1)  # standardized
            skew = z.pow(3).mean(dim=1)              # [B]
            kurt = z.pow(4).mean(dim=1) - 3.0        # excess kurtosis [B]

            moments.append(torch.stack([mean, std, skew, kurt], dim=1))  # [B, 4]

        return torch.cat(moments, dim=1)  # [B, 12]

    # ------------------------------------------------------------------
    # Utility: extract per-scale features for analysis
    # ------------------------------------------------------------------

    def extract_per_scale(self, x0: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract features separately for each probe timestep.

        Useful for ablation studies and visualization.

        Args:
            x0: Input images [B, 3, H, W] in [-1, 1] range.

        Returns:
            Dict mapping timestep → [B, 37] feature vector
            (1 norm + 24 PSD + 12 stat = 37D per scale).
        """
        probe_results = self.backbone.probe(x0, timesteps=self.probe_timesteps)
        per_scale = {}

        for t in self.probe_timesteps:
            t_key = min(max(t, 0), 999)
            r_i = probe_results[t_key]["residual"].float()
            x_t = probe_results[t_key]["x_t"].float()

            r_norm = r_i.norm(dim=[1, 2, 3])
            x_norm = x_t.norm(dim=[1, 2, 3]).clamp(min=1e-8)
            rel_norm = (r_norm / x_norm).unsqueeze(1)

            psd_feat = self._compute_psd_features(r_i)
            stat_feat = self._compute_statistical_moments(r_i)

            per_scale[t_key] = torch.cat([rel_norm, psd_feat, stat_feat], dim=1)

        return per_scale

    def get_vae_band_power(
        self,
        x0: torch.Tensor,
        vae_timesteps: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Extract PSD power at the VAE 8px artifact frequency band.

        The VAE 8px artifact (f=H/8) corresponds to normalized frequency 0.125,
        which falls in band index VAE_BAND_IDX=2 (for n_freq_bands=8).

        Args:
            x0: Input images [B, 3, H, W].
            vae_timesteps: Timesteps to check. Defaults to [400, 600].

        Returns:
            Dict mapping timestep → [B, 3] per-channel power at band VAE_BAND_IDX.
        """
        vae_timesteps = vae_timesteps or [400, 600]
        probe_results = self.backbone.probe(x0, timesteps=vae_timesteps)
        results = {}

        for t in vae_timesteps:
            t_key = min(max(t, 0), 999)
            r_i = probe_results[t_key]["residual"].float()
            B, C, H, W = r_i.shape

            radial_map = self._get_radial_freq_map(H, W // 2 + 1, device=r_i.device)
            band_edges = torch.linspace(0.0, 0.5, self.n_freq_bands + 1, device=r_i.device)

            lo = band_edges[self.VAE_BAND_IDX]
            hi = band_edges[self.VAE_BAND_IDX + 1]
            mask = (radial_map >= lo) & (radial_map < hi)

            per_channel = []
            for c in range(C):
                channel = r_i[:, c, :, :]
                fft = torch.fft.rfft2(channel)
                power = fft.real ** 2 + fft.imag ** 2
                masked_power = power[:, mask].mean(dim=1)  # [B]
                per_channel.append(masked_power)

            results[t_key] = torch.stack(per_channel, dim=1)  # [B, 3]

        return results
