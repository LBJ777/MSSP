"""
experiments/step_b_1a_psd_diagnostic.py
----------------------------------------
Experiment 1A: Fine-grained PSD analysis per timestep.

Goals:
  - For each t in {200, 400, 600, 800, 999}: plot 4-category residual PSD
    curves overlaid on the same axes (64 frequency bins vs 8 in step_b)
  - Compute ANOVA F-statistic for each (t, freq_bin) to find the most
    discriminative (timestep, frequency) combinations
  - Also compare: original image x0 PSD vs residual r_i PSD separability
    (does ADM probing add discriminative information beyond raw images?)

Outputs in output_dir:
  1a_psd_curves.png       -- 5-panel overlaid PSD curves per timestep
  1a_fstat_heatmap.png    -- heatmap of F-statistics (timestep × freq_bin)
  1a_top_discriminative.png -- top discriminative (t, freq) combinations
  1a_x0_vs_residual.png   -- x0 PSD vs residual PSD separability comparison
  1a_results.json         -- all numerical results
  1a_residuals_cache.npz  -- cached residuals (skip ADM if already computed)

Usage:
  # With ADM (needs GPU + checkpoint):
  conda run -n aigc python experiments/step_b_1a_psd_diagnostic.py \\
      --data_dir /path/to/step_a \\
      --model_path /path/to/256x256_diffusion_uncond.pt \\
      --num_samples 50 --device cuda

  # Re-use cached residuals (no GPU needed after first run):
  conda run -n aigc python experiments/step_b_1a_psd_diagnostic.py \\
      --cache_path results/step_b_real_1a/1a_residuals_cache.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_MSSP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MSSP_ROOT not in sys.path:
    sys.path.insert(0, _MSSP_ROOT)

from utils.logger import setup_logger

logger = logging.getLogger("MSSP.1a")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = ["Real", "ProGAN", "SD_v1.5", "Wukong"]
CATEGORY_COLORS = {
    "Real":    "#2196F3",   # blue
    "ProGAN":  "#F44336",   # red
    "SD_v1.5": "#4CAF50",   # green
    "Wukong":  "#FF9800",   # orange
}
PROBE_TIMESTEPS = [200, 400, 600, 800, 999]
N_FREQ_BINS = 64            # fine-grained (step_b used only 8 bands)
IMAGE_SIZE = 256

# Frequency landmarks (normalized: 0–0.5)
F_VAE = 1.0 / 8             # = 0.125  SD VAE 8px artifact
F_GAN_HF = 0.25             # boundary between mid and high freq


# ---------------------------------------------------------------------------
# Data loading (same as step_b, simplified)
# ---------------------------------------------------------------------------

def load_images(data_dir: str, num_samples: int) -> Dict[str, torch.Tensor]:
    """Load images from AIGCDetectBenchmark layout (entry/0_real, entry/1_fake)."""
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def _load_folder(folder: Path, max_n: int):
        if not folder.is_dir():
            return None
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)[:max_n]
        imgs = []
        for p in paths:
            try:
                imgs.append(transform(Image.open(str(p)).convert("RGB")))
            except Exception as e:
                logger.warning("Failed to load %s: %s", p, e)
        return torch.stack(imgs) if imgs else None

    data: Dict[str, torch.Tensor] = {}
    real_loaded = False
    for entry in sorted(Path(data_dir).iterdir()):
        if not entry.is_dir():
            continue
        real_dir = entry / "0_real"
        fake_dir = entry / "1_fake"
        if real_dir.is_dir() and not real_loaded:
            t = _load_folder(real_dir, num_samples)
            if t is not None:
                data["Real"] = t
                real_loaded = True
                logger.info("Real: %d images from %s", len(t), real_dir)
        if fake_dir.is_dir():
            t = _load_folder(fake_dir, num_samples)
            if t is not None:
                data[entry.name] = t
                logger.info("%s: %d images from %s", entry.name, len(t), fake_dir)
    return data


# ---------------------------------------------------------------------------
# Radial PSD computation (fine-grained)
# ---------------------------------------------------------------------------

def compute_radial_psd(
    images: np.ndarray,   # [N, C, H, W], values arbitrary
    n_bins: int = N_FREQ_BINS,
) -> np.ndarray:
    """Compute radial power spectrum averaged over channels.

    Returns:
        psd: [N, n_bins] log-power per frequency bin, averaged over channels.
        freq_centers: [n_bins] center frequencies (normalized 0–0.5).
    """
    N, C, H, W = images.shape
    bin_edges = np.linspace(0.0, 0.5, n_bins + 1)
    freq_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Build radial frequency map once
    h_idx = np.arange(H, dtype=np.float32)
    w_idx = np.arange(W // 2 + 1, dtype=np.float32)
    h_folded = np.minimum(h_idx, H - h_idx)
    hh, ww = np.meshgrid(h_folded, w_idx, indexing="ij")
    radial = np.sqrt(hh ** 2 + ww ** 2) / H   # [H, W//2+1]

    psd_all = np.zeros((N, n_bins), dtype=np.float32)
    for n in range(N):
        per_channel = np.zeros(n_bins, dtype=np.float32)
        for c in range(C):
            ch = images[n, c].astype(np.float32)
            fft = np.fft.rfft2(ch)
            power = np.abs(fft) ** 2
            for b in range(n_bins):
                mask = (radial >= bin_edges[b]) & (radial < bin_edges[b + 1])
                if mask.any():
                    per_channel[b] += np.log1p(power[mask].mean())
                else:
                    per_channel[b] += 0.0
        psd_all[n] = per_channel / C   # average over channels
    return psd_all, freq_centers


# ---------------------------------------------------------------------------
# Residual extraction via ADM backbone
# ---------------------------------------------------------------------------

def extract_residuals(
    data: Dict[str, torch.Tensor],
    backbone,
    timesteps: List[int],
    batch_size: int,
    device: str,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Extract residuals r_i = x_t - eps_hat for each category and timestep.

    Returns:
        Dict[category → Dict[timestep → ndarray [N, 3, H, W]]]
    """
    results: Dict[str, Dict[int, List]] = {
        cat: {t: [] for t in timesteps} for cat in data
    }
    for cat, imgs in data.items():
        N = len(imgs)
        logger.info("Extracting residuals for '%s' (%d images)…", cat, N)
        for i in tqdm(range(0, N, batch_size), desc=cat, leave=False):
            batch = imgs[i: i + batch_size].to(device)
            probe_out = backbone.probe(batch, timesteps=timesteps)
            for t_key, tensors in probe_out.items():
                results[cat][t_key].append(tensors["residual"].cpu().numpy())

    final: Dict[str, Dict[int, np.ndarray]] = {}
    for cat in data:
        final[cat] = {}
        for t in timesteps:
            arrs = results[cat][t]
            if arrs:
                final[cat][t] = np.concatenate(arrs, axis=0)   # [N, 3, H, W]
    return final


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_fstat_matrix(
    residuals: Dict[str, Dict[int, np.ndarray]],
    n_bins: int = N_FREQ_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ANOVA F-statistic for each (timestep, freq_bin) combination.

    Returns:
        fstat_matrix: [n_timesteps, n_bins]  F values
        pval_matrix:  [n_timesteps, n_bins]  p values
        psd_per_cat:  Dict[cat → [n_timesteps, n_bins]] mean PSD arrays
        freq_centers: [n_bins]
    """
    from scipy import stats as sp_stats

    timesteps = sorted(next(iter(residuals.values())).keys())
    cats = list(residuals.keys())

    # Collect PSD per category per timestep
    psd_per_cat: Dict[str, np.ndarray] = {}
    psd_raw_per_cat: Dict[str, List[np.ndarray]] = {cat: [] for cat in cats}
    freq_centers = None

    for t_idx, t in enumerate(timesteps):
        for cat in cats:
            r = residuals[cat][t]   # [N, 3, H, W]
            psd, fc = compute_radial_psd(r, n_bins)
            if freq_centers is None:
                freq_centers = fc
            psd_raw_per_cat[cat].append(psd)  # [N, n_bins]

    for cat in cats:
        psd_per_cat[cat] = np.stack(psd_raw_per_cat[cat], axis=0)  # [n_t, N, n_bins]

    # F-statistic matrix
    n_t = len(timesteps)
    fstat = np.zeros((n_t, n_bins), dtype=np.float32)
    pval  = np.ones((n_t, n_bins), dtype=np.float32)

    for t_idx in range(n_t):
        for b in range(n_bins):
            groups = [psd_per_cat[cat][t_idx, :, b] for cat in cats]
            try:
                f, p = sp_stats.f_oneway(*groups)
                fstat[t_idx, b] = float(f) if np.isfinite(f) else 0.0
                pval[t_idx, b]  = float(p) if np.isfinite(p) else 1.0
            except Exception:
                pass

    return fstat, pval, psd_per_cat, freq_centers, timesteps


def compute_x0_psd(
    data: Dict[str, torch.Tensor],
    n_bins: int = N_FREQ_BINS,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute PSD of original images x0 (no ADM probing).

    Returns:
        psd_x0: Dict[cat → [N, n_bins]]
        freq_centers: [n_bins]
    """
    psd_x0 = {}
    freq_centers = None
    for cat, imgs in data.items():
        arr = imgs.cpu().numpy()   # [N, 3, H, W], range [-1, 1]
        psd, fc = compute_radial_psd(arr, n_bins)
        psd_x0[cat] = psd
        if freq_centers is None:
            freq_centers = fc
    return psd_x0, freq_centers


def separability_score(psd_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-bin ANOVA F-statistic across categories.

    Args:
        psd_dict: Dict[cat → [N, n_bins]]
    Returns:
        [n_bins] F-statistics
    """
    from scipy import stats as sp_stats
    cats = list(psd_dict.keys())
    n_bins = next(iter(psd_dict.values())).shape[1]
    fstat = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        groups = [psd_dict[cat][:, b] for cat in cats]
        try:
            f, _ = sp_stats.f_oneway(*groups)
            fstat[b] = float(f) if np.isfinite(f) else 0.0
        except Exception:
            pass
    return fstat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_psd_curves(
    psd_per_cat: Dict[str, np.ndarray],   # Dict[cat → [n_t, N, n_bins]]
    freq_centers: np.ndarray,
    timesteps: List[int],
    output_path: str,
):
    """5-panel figure: one subplot per timestep, 4 category curves overlaid."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_t = len(timesteps)
    fig, axes = plt.subplots(1, n_t, figsize=(4 * n_t, 4), sharey=False)
    if n_t == 1:
        axes = [axes]

    cats = list(psd_per_cat.keys())

    for t_idx, t in enumerate(timesteps):
        ax = axes[t_idx]
        for cat in cats:
            psd = psd_per_cat[cat][t_idx]   # [N, n_bins]
            mean_psd = psd.mean(axis=0)
            std_psd  = psd.std(axis=0)
            color = CATEGORY_COLORS.get(cat, "#9E9E9E")
            ax.plot(freq_centers, mean_psd, color=color, linewidth=1.5, label=cat)
            ax.fill_between(freq_centers,
                            mean_psd - std_psd,
                            mean_psd + std_psd,
                            color=color, alpha=0.12)

        # Mark key frequencies
        ax.axvline(F_VAE, color="purple", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="f=H/8 (VAE)")
        ax.axvline(F_GAN_HF, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
                   label="f=H/4")
        ax.set_title(f"t = {t}", fontsize=11)
        ax.set_xlabel("Normalized Frequency")
        if t_idx == 0:
            ax.set_ylabel("Mean Log-Power")
        ax.grid(alpha=0.25)

    # Legend on last panel
    handles = [mpatches.Patch(color=CATEGORY_COLORS.get(c, "#9E9E9E"), label=c)
               for c in cats]
    handles += [
        plt.Line2D([0], [0], color="purple", linestyle="--", linewidth=0.8,
                   label="f=H/8 (VAE 8px)"),
        plt.Line2D([0], [0], color="gray", linestyle=":", linewidth=0.8,
                   label="f=H/4"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("1A: Residual PSD per Timestep — Category Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PSD curves: %s", output_path)


def plot_fstat_heatmap(
    fstat: np.ndarray,        # [n_t, n_bins]
    freq_centers: np.ndarray,
    timesteps: List[int],
    output_path: str,
):
    """Heatmap: rows=timesteps, cols=freq_bins, values=F-statistic."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(fstat, aspect="auto", origin="upper",
                   cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="ANOVA F-statistic")

    # Axes
    ax.set_yticks(range(len(timesteps)))
    ax.set_yticklabels([str(t) for t in timesteps])
    ax.set_ylabel("Probe Timestep t")

    # X ticks: every 8 bins
    tick_step = max(1, N_FREQ_BINS // 8)
    xtick_idx = list(range(0, N_FREQ_BINS, tick_step))
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([f"{freq_centers[i]:.3f}" for i in xtick_idx], fontsize=7)
    ax.set_xlabel("Normalized Frequency")

    # Mark VAE and GAN HF lines
    vae_bin = int(np.argmin(np.abs(freq_centers - F_VAE)))
    gan_bin = int(np.argmin(np.abs(freq_centers - F_GAN_HF)))
    ax.axvline(vae_bin, color="blue", linestyle="--", linewidth=1, alpha=0.7,
               label=f"f=H/8 (bin {vae_bin})")
    ax.axvline(gan_bin, color="green", linestyle=":", linewidth=1, alpha=0.7,
               label=f"f=H/4 (bin {gan_bin})")
    ax.legend(fontsize=8, loc="upper right")

    ax.set_title("1A: ANOVA F-statistic — Which (timestep, freq) separates categories best?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F-stat heatmap: %s", output_path)


def plot_top_discriminative(
    fstat: np.ndarray,
    freq_centers: np.ndarray,
    timesteps: List[int],
    top_k: int,
    output_path: str,
):
    """Bar chart of top-K (timestep, freq_bin) pairs by F-statistic."""
    import matplotlib.pyplot as plt

    flat = fstat.flatten()
    top_flat_idx = np.argsort(flat)[::-1][:top_k]
    rows, cols = np.unravel_index(top_flat_idx, fstat.shape)

    labels = [f"t={timesteps[r]}\nf={freq_centers[c]:.3f}" for r, c in zip(rows, cols)]
    values = flat[top_flat_idx]

    fig, ax = plt.subplots(figsize=(min(top_k * 0.7, 14), 4))
    bars = ax.bar(range(top_k), values, color="#E57373", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("ANOVA F-statistic")
    ax.set_title(f"1A: Top {top_k} Most Discriminative (Timestep × Frequency) Combinations")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved top discriminative bar chart: %s", output_path)


def plot_x0_vs_residual(
    f_x0: np.ndarray,        # [n_bins] F-stat for x0 PSD
    f_res: np.ndarray,       # [n_t, n_bins] F-stat for residual PSD
    freq_centers: np.ndarray,
    timesteps: List[int],
    output_path: str,
):
    """Compare per-freq separability: original image x0 vs residuals at each t."""
    import matplotlib.pyplot as plt

    n_t = len(timesteps)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(freq_centers, f_x0, color="black", linewidth=2,
            linestyle="-", label="Original x₀", zorder=5)

    cmap = plt.cm.get_cmap("Blues", n_t + 2)
    for t_idx, t in enumerate(timesteps):
        ax.plot(freq_centers, f_res[t_idx],
                color=cmap(t_idx + 2), linewidth=1.2,
                linestyle="--", label=f"Residual t={t}", alpha=0.85)

    ax.axvline(F_VAE, color="purple", linestyle=":", linewidth=0.8, alpha=0.6,
               label="f=H/8 (VAE)")
    ax.axvline(F_GAN_HF, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
               label="f=H/4")
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("ANOVA F-statistic (category separability)")
    ax.set_title("1B: Does ADM Probing Add Information? — x₀ PSD vs Residual PSD")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved x0 vs residual comparison: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MSSP 1A PSD diagnostic")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Root data directory (AIGCDetectBenchmark layout)")
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to ADM checkpoint (.pt)")
    p.add_argument("--num_samples", type=int, default=50,
                   help="Images per category")
    p.add_argument("--output_dir", type=str, default="./results/step_b_1a")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_freq_bins", type=int, default=N_FREQ_BINS,
                   help="Number of frequency bins (default 64, step_b uses 8)")
    p.add_argument("--top_k", type=int, default=20,
                   help="Top-K (t, freq) pairs to highlight")
    p.add_argument("--cache_path", type=str, default=None,
                   help="If set, load cached residuals from this .npz (skip ADM inference)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger("MSSP", log_dir=args.output_dir, log_filename="1a_diagnostic.log")
    logger.info("=== MSSP 1A PSD Diagnostic ===")
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Freq bins: %d (step_b used 8)", args.n_freq_bins)

    t0_total = time.time()

    # ------------------------------------------------------------------
    # Step 1: Obtain residuals (load from cache or run ADM inference)
    # ------------------------------------------------------------------
    cache_path = os.path.join(args.output_dir, "1a_residuals_cache.npz")

    if args.cache_path and os.path.isfile(args.cache_path):
        logger.info("Loading cached residuals from %s", args.cache_path)
        cache = np.load(args.cache_path, allow_pickle=True)
        # Reconstruct nested dict
        residuals: Dict[str, Dict[int, np.ndarray]] = {}
        data_x0: Dict[str, np.ndarray] = {}
        for key in cache.files:
            if key.startswith("res__"):
                _, cat, t_str = key.split("__")
                t = int(t_str)
                residuals.setdefault(cat, {})[t] = cache[key]
            elif key.startswith("x0__"):
                _, cat = key.split("__", 1)
                data_x0[cat] = cache[key]
        logger.info("Loaded residuals for %d categories", len(residuals))
    else:
        if not args.data_dir or not args.model_path:
            print("[ERROR] Provide --data_dir and --model_path (or --cache_path to skip inference)")
            sys.exit(1)

        logger.info("Loading images from %s …", args.data_dir)
        data = load_images(args.data_dir, args.num_samples)
        logger.info("Categories loaded: %s", list(data.keys()))

        logger.info("Initialising MSSPBackbone …")
        from models.backbone.adm_wrapper import MSSPBackbone
        backbone = MSSPBackbone(
            model_path=args.model_path,
            device=args.device,
        )

        logger.info("Extracting residuals (5 timesteps × %d categories)…", len(data))
        residuals = extract_residuals(data, backbone, PROBE_TIMESTEPS,
                                      args.batch_size, args.device)

        # Store x0 as numpy for PSD comparison
        data_x0 = {cat: imgs.numpy() for cat, imgs in data.items()}

        # Save cache
        logger.info("Saving residuals cache → %s", cache_path)
        save_dict = {}
        for cat, tdict in residuals.items():
            for t, arr in tdict.items():
                save_dict[f"res__{cat}__{t}"] = arr
        for cat, arr in data_x0.items():
            save_dict[f"x0__{cat}"] = arr
        np.savez_compressed(cache_path, **save_dict)

    # ------------------------------------------------------------------
    # Step 2: Compute PSD per category per timestep
    # ------------------------------------------------------------------
    logger.info("Computing fine-grained PSD (%d bins) …", args.n_freq_bins)
    fstat, pval, psd_per_cat, freq_centers, timesteps = compute_fstat_matrix(
        residuals, n_bins=args.n_freq_bins)

    # ------------------------------------------------------------------
    # Step 3: Compute x0 PSD separability (for 1B comparison)
    # ------------------------------------------------------------------
    logger.info("Computing original image (x₀) PSD separability …")
    psd_x0_dict = {}
    for cat, arr in data_x0.items():
        psd, _ = compute_radial_psd(arr, n_bins=args.n_freq_bins)
        psd_x0_dict[cat] = psd

    f_x0 = separability_score(psd_x0_dict)

    # ------------------------------------------------------------------
    # Step 4: Print key findings
    # ------------------------------------------------------------------
    logger.info("=== Key findings ===")
    logger.info("F-stat matrix shape: %s", fstat.shape)

    # Top (t, bin) overall
    flat = fstat.flatten()
    top_flat = np.argsort(flat)[::-1][:args.top_k]
    rows, cols = np.unravel_index(top_flat, fstat.shape)
    logger.info("Top %d (timestep, freq_bin) by F-statistic:", args.top_k)
    top_results = []
    for rank, (r, c) in enumerate(zip(rows, cols)):
        t_val = timesteps[r]
        f_val = float(freq_centers[c])
        F_val = float(fstat[r, c])
        p_val = float(pval[r, c])
        logger.info("  #%d  t=%d  f=%.4f  F=%.2f  p=%.2e", rank + 1, t_val, f_val, F_val, p_val)
        top_results.append({"rank": rank + 1, "t": t_val, "freq": f_val,
                             "F": F_val, "p": p_val})

    # Per-timestep max F
    logger.info("Per-timestep maximum F-statistic:")
    per_t_max = []
    for t_idx, t in enumerate(timesteps):
        max_f = float(fstat[t_idx].max())
        best_bin = int(fstat[t_idx].argmax())
        best_freq = float(freq_centers[best_bin])
        logger.info("  t=%d: max F=%.2f at f=%.4f (bin %d)", t, max_f, best_freq, best_bin)
        per_t_max.append({"t": t, "max_F": max_f, "best_freq": best_freq, "best_bin": best_bin})

    # x0 vs residual
    logger.info("x₀ PSD separability: max F=%.2f at f=%.4f",
                float(f_x0.max()), float(freq_centers[f_x0.argmax()]))
    logger.info("Residual PSD separability (max across all t): max F=%.2f",
                float(fstat.max()))
    adm_adds_info = fstat.max() > f_x0.max()
    logger.info("→ ADM probing %s information beyond raw x₀ PSD",
                "ADDS" if adm_adds_info else "does NOT add")

    # VAE band analysis
    vae_bin = int(np.argmin(np.abs(freq_centers - F_VAE)))
    logger.info("At f=H/8 (bin %d, f≈%.4f):", vae_bin, freq_centers[vae_bin])
    for t_idx, t in enumerate(timesteps):
        logger.info("  t=%d: F=%.3f, p=%.2e", t,
                    float(fstat[t_idx, vae_bin]), float(pval[t_idx, vae_bin]))

    # ------------------------------------------------------------------
    # Step 5: Generate plots
    # ------------------------------------------------------------------
    logger.info("Generating plots …")
    plot_psd_curves(psd_per_cat, freq_centers, timesteps,
                    os.path.join(args.output_dir, "1a_psd_curves.png"))
    plot_fstat_heatmap(fstat, freq_centers, timesteps,
                       os.path.join(args.output_dir, "1a_fstat_heatmap.png"))
    plot_top_discriminative(fstat, freq_centers, timesteps, args.top_k,
                            os.path.join(args.output_dir, "1a_top_discriminative.png"))
    plot_x0_vs_residual(f_x0, fstat, freq_centers, timesteps,
                        os.path.join(args.output_dir, "1a_x0_vs_residual.png"))

    # ------------------------------------------------------------------
    # Step 6: Save JSON results
    # ------------------------------------------------------------------
    results_json = {
        "n_freq_bins": args.n_freq_bins,
        "timesteps": timesteps,
        "top_discriminative": top_results,
        "per_timestep_max_F": per_t_max,
        "x0_max_F": float(f_x0.max()),
        "x0_best_freq": float(freq_centers[f_x0.argmax()]),
        "residual_max_F": float(fstat.max()),
        "adm_adds_information": bool(adm_adds_info),
        "vae_band": {
            "freq": float(freq_centers[vae_bin]),
            "bin_idx": vae_bin,
            "F_per_timestep": [
                {"t": timesteps[i], "F": float(fstat[i, vae_bin]),
                 "p": float(pval[i, vae_bin])}
                for i in range(len(timesteps))
            ],
        },
        "total_time_s": round(time.time() - t0_total, 1),
    }
    json_path = os.path.join(args.output_dir, "1a_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info("Results JSON: %s", json_path)
    logger.info("Total time: %.1fs", time.time() - t0_total)

    print("\n" + "=" * 60)
    print("  1A PSD Diagnostic Complete")
    print("=" * 60)
    print(f"  Top discriminative position:  t={top_results[0]['t']}, "
          f"f={top_results[0]['freq']:.4f}, F={top_results[0]['F']:.2f}")
    print(f"  Best per-timestep F:  ", end="")
    print("  |  ".join(f"t={r['t']}: {r['max_F']:.1f}" for r in per_t_max))
    print(f"  x₀ PSD max F: {f_x0.max():.2f}  |  Residual PSD max F: {fstat.max():.2f}")
    info_str = "YES — ADM adds info" if adm_adds_info else "NO  — x₀ PSD already encodes signal"
    print(f"  ADM information gain: {info_str}")
    print(f"  Outputs: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
