"""
experiments/step_b_mssp_validation.py
--------------------------------------
MSSP Step B: Statistical validation of multi-scale single-step probing.

Verifies three core hypotheses from 新方法构思.md §3.4:

  Test 1 — GAN High-Frequency Artifacts (t=200):
      ProGAN images should have significantly higher high-frequency PSD
      in r_{t=200} compared to real images (checkerboard artifacts).
      Expected: ProGAN high-freq power > 2× real high-freq power.

  Test 2 — SD VAE 8px Artifacts (t=400-600):
      SD images should show elevated power at f=H/8 frequency band
      in r_{t=400} and r_{t=600} residuals.
      Compare against DRIFT G5 failure (z-score=0.0013 on x_T).
      Expected: z-score > 2.0 in residual space.

  Test 3 — k-NN Classification Accuracy:
      185D MSSP features should achieve k-NN accuracy ≥ 84%
      (the DRIFT step_a baseline with 128D endpoint features).

  Test 4 — Multi-Scale PSD Visualization:
      Generate 5×4 comparison plot showing PSD at each noise scale
      for each image category. Mark key frequency bands.

Usage:
    # Mock mode (no GPU/checkpoint required, ~1 min)
    python experiments/step_b_mssp_validation.py --mock --num_samples 20

    # Real data mode
    python experiments/step_b_mssp_validation.py \\
        --data_dir /path/to/data \\
        --model_path /path/to/256x256_diffusion_uncond.pt \\
        --num_samples 200 --device cuda

Output files in output_dir:
    step_b_report.txt         — PASS/FAIL summary per hypothesis
    test1_gan_hf_psd.png      — Test 1 visualization
    test2_vae_artifact.png    — Test 2 visualization
    test3_knn_confusion.png   — Test 3 confusion matrix
    test4_multiscale_psd.png  — Test 4 multi-scale overview
    mssp_features.npz         — Cached feature matrix [N, 185]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: ensure MSSP package is importable
# ---------------------------------------------------------------------------
_MSSP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MSSP_ROOT not in sys.path:
    sys.path.insert(0, _MSSP_ROOT)

from models.backbone.adm_wrapper import MSSPBackbone
from models.features.mssp import MSSPFeatureExtractor
from utils.logger import setup_logger

logger = logging.getLogger("MSSP.validation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = ["Real", "ProGAN", "SD_v1.5", "Wukong"]
CATEGORY_COLORS = {
    "Real": "#2196F3",
    "ProGAN": "#F44336",
    "SD_v1.5": "#4CAF50",
    "Wukong": "#FF9800",
}
PROBE_TIMESTEPS = [200, 400, 600, 800, 999]
N_FREQ_BANDS = 8
IMAGE_SIZE = 256

# Frequency band index corresponding to f=H/8 (VAE 8px artifact)
# band_edges = linspace(0, 0.5, 9) → [0, 0.0625, 0.125, 0.1875, ...]
# f=H/8=32 → normalized = 32/256 = 0.125 → band index 2
VAE_BAND_IDX = 2
# High-frequency bands for GAN test: bands 4-7 (f > 0.25)
HF_BAND_START = 4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_images_mock(num_samples: int, image_size: int = 256) -> Dict[str, torch.Tensor]:
    """Generate random mock images for unit testing.

    Returns:
        Dict mapping category name → [N, 3, H, W] tensor in [-1, 1] range.
    """
    logger.info("Mock mode: generating %d random images per category", num_samples)
    data = {}
    for cat in CATEGORIES:
        # Each category gets slightly different statistics to allow feature variation
        torch.manual_seed(hash(cat) % (2**31))
        imgs = torch.randn(num_samples, 3, image_size, image_size)
        # Clamp to [-1, 1]
        imgs = imgs.clamp(-1.0, 1.0)
        data[cat] = imgs
    return data


def load_images_from_dir(
    data_dir: str,
    num_samples: int,
    image_size: int = 256,
    category_map: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """Load images from AIGCDetectBenchmark-style directory.

    Expected structure:
        data_dir/
        ├── <gen_A>/0_real/*.png
        ├── <gen_A>/1_fake/*.png
        ├── <gen_B>/1_fake/*.png
        └── ...
    OR flat:
        data_dir/
        ├── real/*.png
        ├── ProGAN/*.png
        └── ...

    Real images are loaded once from the first generator's 0_real/ folder.
    Fake images are loaded per-generator from each 1_fake/ folder.

    Args:
        data_dir: Root dataset directory.
        num_samples: Max images to load per category.
        image_size: Target image size.
        category_map: Optional mapping from directory names to category labels.

    Returns:
        Dict mapping category name → [N, 3, H, W] in [-1, 1].
    """
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])

    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def _load_folder(folder: Path, max_n: int) -> Optional[torch.Tensor]:
        if not folder.is_dir():
            return None
        paths = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in SUPPORTED_EXT
        ])[:max_n]
        if not paths:
            return None
        imgs = []
        for p in paths:
            try:
                img = Image.open(str(p)).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                logger.warning("Failed to load %s: %s", p, e)
        if not imgs:
            return None
        return torch.stack(imgs)  # [N, 3, H, W]

    data_path = Path(data_dir)
    data: Dict[str, torch.Tensor] = {}
    real_loaded = False

    for entry in sorted(data_path.iterdir()):
        if not entry.is_dir():
            continue

        # Check AIGCDetectBenchmark layout: entry/0_real, entry/1_fake
        real_dir = entry / "0_real"
        fake_dir = entry / "1_fake"

        if real_dir.is_dir() and not real_loaded:
            imgs = _load_folder(real_dir, num_samples)
            if imgs is not None:
                data["Real"] = imgs
                real_loaded = True
                logger.info("Loaded Real: %d images from %s", len(imgs), real_dir)

        if fake_dir.is_dir():
            cat_name = category_map.get(entry.name, entry.name) if category_map else entry.name
            imgs = _load_folder(fake_dir, num_samples)
            if imgs is not None:
                data[cat_name] = imgs
                logger.info("Loaded %s: %d images from %s", cat_name, len(imgs), fake_dir)

    if not data:
        # Fallback: flat directory (each subdir is a category)
        for entry in sorted(data_path.iterdir()):
            if not entry.is_dir():
                continue
            imgs = _load_folder(entry, num_samples)
            if imgs is not None:
                cat_name = category_map.get(entry.name, entry.name) if category_map else entry.name
                data[cat_name] = imgs
                logger.info("Loaded %s: %d images from %s", cat_name, len(imgs), entry)

    if not data:
        raise RuntimeError(f"No images found in '{data_dir}'. Check directory structure.")

    return data


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(
    data: Dict[str, torch.Tensor],
    extractor: MSSPFeatureExtractor,
    batch_size: int = 4,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract MSSP features for all categories.

    Args:
        data: Dict mapping category → [N, 3, H, W] images in [-1, 1].
        extractor: MSSPFeatureExtractor instance.
        batch_size: Images processed per forward pass.
        device: PyTorch device string.

    Returns:
        (features, labels, category_names):
            features: [N_total, 185] float32 ndarray
            labels: [N_total] int32 ndarray (category index)
            category_names: list of category names (index → name)
    """
    all_features = []
    all_labels = []
    category_names = sorted(data.keys())

    for cat_idx, cat_name in enumerate(category_names):
        imgs = data[cat_name]  # [N, 3, H, W]
        N = imgs.shape[0]
        cat_features = []

        logger.info("Extracting features for '%s' (%d images)...", cat_name, N)
        t0 = time.time()

        for i in tqdm(range(0, N, batch_size), desc=cat_name, leave=False):
            batch = imgs[i: i + batch_size].to(device)
            with torch.no_grad():
                feats = extractor.extract(batch)  # [B, 185]
            cat_features.append(feats.cpu().numpy())

        cat_features = np.concatenate(cat_features, axis=0)  # [N, 185]
        all_features.append(cat_features)
        all_labels.append(np.full(N, cat_idx, dtype=np.int32))

        elapsed = time.time() - t0
        logger.info(
            "  Done: %d images in %.1fs (%.2f img/s)",
            N, elapsed, N / (elapsed + 1e-6)
        )

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, category_names


def extract_residuals_per_scale(
    data: Dict[str, torch.Tensor],
    backbone: MSSPBackbone,
    timesteps: List[int],
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, Dict[int, np.ndarray]]:
    """Extract raw residuals per timestep for PSD analysis.

    Returns:
        Dict[category → Dict[timestep → [N, 3, H, W] residuals]].
    """
    results: Dict[str, Dict[int, List[np.ndarray]]] = {
        cat: {t: [] for t in timesteps}
        for cat in data
    }

    for cat_name, imgs in data.items():
        N = imgs.shape[0]
        logger.info("Extracting residuals for '%s' (%d images)...", cat_name, N)

        for i in tqdm(range(0, N, batch_size), desc=f"residuals/{cat_name}", leave=False):
            batch = imgs[i: i + batch_size].to(device)
            probe_out = backbone.probe(batch, timesteps=timesteps)
            for t_key, tensors in probe_out.items():
                residual = tensors["residual"].cpu().numpy()  # [B, 3, H, W]
                results[cat_name][t_key].append(residual)

    # Concatenate along batch dim
    final: Dict[str, Dict[int, np.ndarray]] = {}
    for cat_name in data:
        final[cat_name] = {}
        for t in timesteps:
            if results[cat_name][t]:
                final[cat_name][t] = np.concatenate(results[cat_name][t], axis=0)
    return final


# ---------------------------------------------------------------------------
# Test 1: GAN high-frequency artifacts at t=200
# ---------------------------------------------------------------------------

def run_test1_gan_hf_psd(
    residuals: Dict[str, Dict[int, np.ndarray]],
    output_dir: str,
    n_freq_bands: int = N_FREQ_BANDS,
) -> Dict:
    """Test 1: GAN images should have higher high-freq PSD at t=200.

    Returns result dict with PASS/FAIL and statistics.
    """
    import matplotlib.pyplot as plt

    logger.info("=== Test 1: GAN High-Frequency PSD (t=200) ===")

    t_target = 200
    results = {}

    # Compute mean high-frequency PSD per category
    hf_powers = {}  # category → [N] array of mean high-freq power
    for cat_name, scale_data in residuals.items():
        if t_target not in scale_data:
            logger.warning("t=%d not in residuals for '%s'", t_target, cat_name)
            continue

        r = scale_data[t_target]  # [N, 3, H, W]
        N, C, H, W = r.shape
        band_edges = np.linspace(0.0, 0.5, n_freq_bands + 1)

        # Build radial freq map
        h_idx = np.arange(H, dtype=np.float32)
        w_idx = np.arange(W // 2 + 1, dtype=np.float32)
        h_folded = np.minimum(h_idx, H - h_idx)
        hh, ww = np.meshgrid(h_folded, w_idx, indexing="ij")
        radial = np.sqrt(hh ** 2 + ww ** 2) / H  # [H, W//2+1]

        # High-frequency mask: bands HF_BAND_START to n_freq_bands-1
        hf_mask = radial >= band_edges[HF_BAND_START]  # [H, W//2+1]

        cat_hf = []
        for n in range(N):
            img_hf = []
            for c in range(C):
                channel = r[n, c]  # [H, W]
                fft = np.fft.rfft2(channel)
                power = np.abs(fft) ** 2
                hf_power = np.log1p(power[hf_mask].mean())
                img_hf.append(hf_power)
            cat_hf.append(np.mean(img_hf))  # average over channels

        hf_powers[cat_name] = np.array(cat_hf)
        results[cat_name] = {
            "mean_hf_power": float(np.mean(cat_hf)),
            "std_hf_power": float(np.std(cat_hf)),
            "n": int(N),
        }
        logger.info(
            "  %s: mean high-freq power = %.4f ± %.4f",
            cat_name, results[cat_name]["mean_hf_power"], results[cat_name]["std_hf_power"]
        )

    # Compute ProGAN vs Real ratio
    test_pass = False
    ratio = None
    if "ProGAN" in hf_powers and "Real" in hf_powers:
        real_mean = results["Real"]["mean_hf_power"]
        progan_mean = results["ProGAN"]["mean_hf_power"]
        ratio = progan_mean / (real_mean + 1e-8)
        test_pass = ratio > 2.0
        logger.info(
            "  ProGAN/Real high-freq ratio = %.3f (threshold > 2.0) → %s",
            ratio, "PASS" if test_pass else "FAIL"
        )
        results["ratio_ProGAN_vs_Real"] = float(ratio)

    results["PASS"] = test_pass

    # Visualization
    if hf_powers:
        fig, ax = plt.subplots(figsize=(8, 5))
        cats_sorted = sorted(hf_powers.keys())
        means = [results[c]["mean_hf_power"] for c in cats_sorted]
        stds = [results[c]["std_hf_power"] for c in cats_sorted]
        colors = [CATEGORY_COLORS.get(c, "#9E9E9E") for c in cats_sorted]

        bars = ax.bar(cats_sorted, means, yerr=stds, capsize=4,
                      color=colors, alpha=0.8, edgecolor="black")
        ax.set_ylabel("Mean Log High-Freq Power (f > 0.25)")
        ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        pass_str = "PASS" if test_pass else "FAIL"
        ax.set_title(
            f"Test 1: GAN High-Frequency Artifacts at t=200\n"
            f"ProGAN/Real ratio = {ratio_str} ({pass_str})"
        )
        ax.axhline(y=results.get("Real", {}).get("mean_hf_power", 0) * 2,
                   color="red", linestyle="--", alpha=0.5, label="2× Real threshold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        out_path = os.path.join(output_dir, "test1_gan_hf_psd.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved: %s", out_path)

    return results


# ---------------------------------------------------------------------------
# Test 2: SD VAE 8px artifact at t=400-600
# ---------------------------------------------------------------------------

def run_test2_vae_artifact(
    residuals: Dict[str, Dict[int, np.ndarray]],
    output_dir: str,
    n_freq_bands: int = N_FREQ_BANDS,
) -> Dict:
    """Test 2: SD images should have elevated f=H/8 power in residuals.

    Compare against DRIFT G5 baseline: z-score = 0.0013 on x_T.
    Expected: z-score > 2.0 in residual space.
    """
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    logger.info("=== Test 2: SD VAE 8px Artifact (t=400, t=600) ===")

    band_edges = np.linspace(0.0, 0.5, n_freq_bands + 1)
    vae_lo = band_edges[VAE_BAND_IDX]
    vae_hi = band_edges[VAE_BAND_IDX + 1]

    logger.info(
        "  VAE artifact band: f ∈ [%.4f, %.4f] (band index %d/%d)",
        vae_lo, vae_hi, VAE_BAND_IDX, n_freq_bands
    )

    results = {}
    vae_band_powers: Dict[str, Dict[int, np.ndarray]] = {}  # cat → t → [N]

    for cat_name, scale_data in residuals.items():
        vae_band_powers[cat_name] = {}
        for t in [400, 600]:
            if t not in scale_data:
                continue
            r = scale_data[t]  # [N, 3, H, W]
            N, C, H, W = r.shape

            # Build radial freq map
            h_idx = np.arange(H, dtype=np.float32)
            w_idx = np.arange(W // 2 + 1, dtype=np.float32)
            h_folded = np.minimum(h_idx, H - h_idx)
            hh, ww = np.meshgrid(h_folded, w_idx, indexing="ij")
            radial = np.sqrt(hh ** 2 + ww ** 2) / H
            vae_mask = (radial >= vae_lo) & (radial < vae_hi)

            img_vae = []
            for n in range(N):
                chan_vae = []
                for c in range(C):
                    channel = r[n, c]
                    fft = np.fft.rfft2(channel)
                    power = np.abs(fft) ** 2
                    vae_power = np.log1p(power[vae_mask].mean())
                    chan_vae.append(vae_power)
                img_vae.append(np.mean(chan_vae))

            vae_band_powers[cat_name][t] = np.array(img_vae)

    # Compute z-score: SD vs Real
    sd_key = next((k for k in vae_band_powers if "SD" in k or "sd" in k), None)
    test_pass = False

    for t in [400, 600]:
        if sd_key and "Real" in vae_band_powers and t in vae_band_powers[sd_key] and t in vae_band_powers["Real"]:
            sd_vals = vae_band_powers[sd_key][t]
            real_vals = vae_band_powers["Real"][t]

            # z-score: how many std above Real mean is SD mean
            real_mean = real_vals.mean()
            real_std = real_vals.std() + 1e-8
            sd_mean = sd_vals.mean()
            z_score = (sd_mean - real_mean) / real_std

            t_stat, p_val = scipy_stats.ttest_ind(sd_vals, real_vals)
            pass_t = z_score > 2.0

            results[f"t{t}"] = {
                "sd_mean": float(sd_mean),
                "real_mean": float(real_mean),
                "z_score": float(z_score),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "PASS": pass_t,
            }

            if pass_t:
                test_pass = True

            logger.info(
                "  t=%d: SD mean=%.4f, Real mean=%.4f, z-score=%.3f "
                "(threshold > 2.0) → %s",
                t, sd_mean, real_mean, z_score,
                "PASS" if pass_t else "FAIL"
            )

    # Comparison with DRIFT G5 (z=0.0013 on x_T)
    results["drift_g5_reference_zscore"] = 0.0013
    results["PASS"] = test_pass

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, t in enumerate([400, 600]):
        ax = axes[ax_idx]
        cats_with_data = [c for c in sorted(vae_band_powers) if t in vae_band_powers[c]]
        if not cats_with_data:
            ax.set_title(f"t={t}: No data")
            continue

        means = [vae_band_powers[c][t].mean() for c in cats_with_data]
        stds = [vae_band_powers[c][t].std() for c in cats_with_data]
        colors = [CATEGORY_COLORS.get(c, "#9E9E9E") for c in cats_with_data]

        ax.bar(cats_with_data, means, yerr=stds, capsize=4,
               color=colors, alpha=0.8, edgecolor="black")
        ax.set_ylabel("Log Power at f=H/8 Band")
        z = results.get(f"t{t}", {}).get("z_score", None)
        p = results.get(f"t{t}", {}).get("PASS", False)
        z_str = f"{z:.3f}" if z is not None else "N/A"
        p_str = "PASS" if p else "FAIL"
        ax.set_title(
            f"t={t}: VAE 8px Artifact Band\n"
            f"z-score = {z_str} ({p_str})\n"
            f"[DRIFT G5 baseline: z=0.0013]"
        )
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Test 2: SD VAE 8px Artifact at t=400-600 vs DRIFT G5 (z=0.0013 on x_T)")
    out_path = os.path.join(output_dir, "test2_vae_artifact.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    return results


# ---------------------------------------------------------------------------
# Test 3: k-NN classification accuracy
# ---------------------------------------------------------------------------

def run_test3_knn_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    category_names: List[str],
    output_dir: str,
    k: int = 5,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """Test 3: k-NN classification accuracy on 185D MSSP features.

    Uses stratified k-fold cross-validation to match DRIFT step_a Test5.
    Expected: accuracy ≥ 84% (DRIFT baseline).

    Args:
        features: [N, 185] feature matrix.
        labels: [N] integer category labels.
        category_names: List mapping label index → category name.
        k: Number of nearest neighbors.
        n_splits: Number of CV folds.
        seed: Random seed for reproducibility.

    Returns:
        Result dict with accuracy statistics and confusion matrix.
    """
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix

    logger.info("=== Test 3: k-NN Classification (k=%d, %d-fold CV) ===", k, n_splits)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = labels

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accuracies = []
    all_preds = np.zeros_like(y)
    all_true = np.zeros_like(y)
    idx = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        acc = accuracy_score(y_val, preds)
        fold_accuracies.append(acc)

        # Collect predictions for confusion matrix
        n_val = len(val_idx)
        all_preds[idx: idx + n_val] = preds
        all_true[idx: idx + n_val] = y_val
        idx += n_val

        logger.info("  Fold %d/%d: accuracy = %.1f%%", fold + 1, n_splits, acc * 100)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    test_pass = mean_acc >= 0.84

    logger.info(
        "  Overall: %.1f%% ± %.1f%% (threshold ≥ 84%%) → %s",
        mean_acc * 100, std_acc * 100,
        "PASS" if test_pass else "FAIL"
    )
    logger.info("  DRIFT baseline (step_a Test5): 84.0% ± 3.6%")

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Fold accuracy bar chart
    ax1.bar(
        [f"Fold {i+1}" for i in range(len(fold_accuracies))],
        [a * 100 for a in fold_accuracies],
        color="#2196F3", alpha=0.8, edgecolor="black"
    )
    ax1.axhline(y=84.0, color="red", linestyle="--", linewidth=2, label="DRIFT baseline 84%")
    ax1.axhline(y=mean_acc * 100, color="green", linestyle="-", linewidth=2,
                label=f"MSSP mean {mean_acc*100:.1f}%")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"k-NN Fold Accuracies (k={k})")
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    # Normalized confusion matrix
    import matplotlib.colors as mcolors
    n_cats = len(category_names)
    tick_labels = [category_names[i] if i < len(category_names) else str(i)
                   for i in range(n_cats)]
    im = ax2.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2)
    ax2.set_xticks(range(n_cats))
    ax2.set_yticks(range(n_cats))
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax2.set_yticklabels(tick_labels)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Normalized Confusion Matrix")
    for i in range(n_cats):
        for j in range(n_cats):
            val = cm_normalized[i, j]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val > 0.5 else "black", fontsize=9)

    fig.suptitle(
        f"Test 3: k-NN Accuracy = {mean_acc*100:.1f}% ± {std_acc*100:.1f}% "
        f"({'PASS ✓' if test_pass else 'FAIL ✗'}, threshold ≥ 84%)"
    )
    out_path = os.path.join(output_dir, "test3_knn_accuracy.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "fold_accuracies": fold_accuracies,
        "drift_baseline": 0.84,
        "confusion_matrix": cm.tolist(),
        "PASS": test_pass,
    }


# ---------------------------------------------------------------------------
# Test 4: Multi-scale PSD visualization
# ---------------------------------------------------------------------------

def run_test4_multiscale_psd(
    residuals: Dict[str, Dict[int, np.ndarray]],
    output_dir: str,
    n_freq_bands: int = N_FREQ_BANDS,
) -> Dict:
    """Test 4: Multi-scale PSD overview — 5 timesteps × N categories.

    Generates a grid showing radial PSD profiles for each (category, scale).
    """
    import matplotlib.pyplot as plt

    logger.info("=== Test 4: Multi-Scale PSD Visualization ===")

    categories_present = sorted(residuals.keys())
    timesteps_present = sorted(set(
        t for cat_data in residuals.values() for t in cat_data
    ))

    n_scales = len(timesteps_present)
    n_cats = len(categories_present)

    if n_scales == 0 or n_cats == 0:
        logger.warning("No data available for Test 4")
        return {"PASS": True, "note": "No data"}

    fig, axes = plt.subplots(
        n_cats, n_scales,
        figsize=(3 * n_scales, 3 * n_cats),
        squeeze=False,
    )

    band_centers = np.linspace(0.5 / n_freq_bands, 0.5 - 0.5 / n_freq_bands, n_freq_bands)

    for row_idx, cat_name in enumerate(categories_present):
        for col_idx, t in enumerate(timesteps_present):
            ax = axes[row_idx][col_idx]

            if t not in residuals[cat_name]:
                ax.set_visible(False)
                continue

            r = residuals[cat_name][t]  # [N, 3, H, W]
            N, C, H, W = r.shape

            # Compute mean radial PSD across all images and channels
            h_idx = np.arange(H, dtype=np.float32)
            w_idx = np.arange(W // 2 + 1, dtype=np.float32)
            h_folded = np.minimum(h_idx, H - h_idx)
            hh, ww = np.meshgrid(h_folded, w_idx, indexing="ij")
            radial = np.sqrt(hh ** 2 + ww ** 2) / H

            band_edges = np.linspace(0.0, 0.5, n_freq_bands + 1)
            mean_band_power = np.zeros(n_freq_bands)

            for bi in range(n_freq_bands):
                mask = (radial >= band_edges[bi]) & (radial < band_edges[bi + 1])
                if mask.sum() > 0:
                    all_vals = []
                    for n in range(min(N, 20)):  # limit for speed
                        for c in range(C):
                            fft = np.fft.rfft2(r[n, c])
                            power = np.abs(fft) ** 2
                            all_vals.append(power[mask].mean())
                    mean_band_power[bi] = np.log1p(np.mean(all_vals)) if all_vals else 0.0

            color = CATEGORY_COLORS.get(cat_name, "#9E9E9E")
            ax.bar(range(n_freq_bands), mean_band_power,
                   color=color, alpha=0.8, edgecolor="black", width=0.8)

            # Mark VAE artifact band
            ax.axvline(x=VAE_BAND_IDX, color="purple", linestyle="--",
                       alpha=0.7, label="f=H/8" if col_idx == 0 else "")
            # Mark high-frequency start
            ax.axvline(x=HF_BAND_START - 0.5, color="orange", linestyle=":",
                       alpha=0.7, label="HF start" if col_idx == 0 else "")

            ax.set_xticks(range(n_freq_bands))
            ax.set_xticklabels([f"{b:.2f}" for b in band_centers], rotation=45, fontsize=6)

            if row_idx == 0:
                ax.set_title(f"t={t}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(cat_name, fontsize=9)

            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Test 4: Multi-Scale PSD (rows=category, cols=timestep)\n"
        "Purple dashed=f=H/8 (VAE), Orange dotted=HF boundary",
        fontsize=10
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "test4_multiscale_psd.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    return {"PASS": True, "n_categories": n_cats, "n_timesteps": n_scales}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    test_results: Dict,
    output_dir: str,
    args: argparse.Namespace,
    elapsed_total: float,
) -> None:
    """Write a human-readable PASS/FAIL report to step_b_report.txt."""
    report_lines = [
        "=" * 70,
        "MSSP Step B Validation Report",
        "=" * 70,
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {'MOCK' if args.mock else 'REAL'}",
        f"Samples per category: {args.num_samples}",
        f"Device: {args.device}",
        f"Total time: {elapsed_total:.1f}s",
        "",
        "=" * 70,
        "Hypothesis Verification",
        "=" * 70,
        "",
    ]

    overall_pass = True

    # Test 1
    t1 = test_results.get("test1", {})
    t1_pass = t1.get("PASS", False)
    if not t1_pass:
        overall_pass = False
    ratio = t1.get("ratio_ProGAN_vs_Real", "N/A")
    ratio_str = f"{ratio:.3f}" if isinstance(ratio, float) else str(ratio)
    t1_str = "PASS" if t1_pass else "FAIL"
    report_lines += [
        f"Test 1 — GAN High-Frequency Artifacts (t=200)",
        f"  Hypothesis: ProGAN high-freq PSD > 2x Real",
        f"  ProGAN/Real ratio: {ratio_str}",
        f"  Result: {t1_str}",
        "",
    ]

    # Test 2
    t2 = test_results.get("test2", {})
    t2_pass = t2.get("PASS", False)
    if not t2_pass:
        overall_pass = False
    drift_ref = t2.get("drift_g5_reference_zscore", 0.0013)
    report_lines += [
        f"Test 2 — SD VAE 8px Artifact (t=400-600)",
        f"  Hypothesis: SD f=H/8 z-score > 2.0 in residual space",
        f"  Reference (DRIFT G5 on x_T): z = {drift_ref}",
    ]
    for t_key in ["t400", "t600"]:
        t_data = t2.get(t_key, {})
        if t_data:
            z = t_data.get("z_score", "N/A")
            p = t_data.get("PASS", False)
            z_str = f"{z:.3f}" if isinstance(z, float) else str(z)
            p_str = "PASS" if p else "FAIL"
            report_lines.append(f"  {t_key}: z-score = {z_str} ({p_str})")
    t2_str = "PASS" if t2_pass else "FAIL"
    report_lines += [f"  Overall Test 2: {t2_str}", ""]

    # Test 3
    t3 = test_results.get("test3", {})
    t3_pass = t3.get("PASS", False)
    if not t3_pass:
        overall_pass = False
    acc = t3.get("mean_accuracy", "N/A")
    std = t3.get("std_accuracy", "N/A")
    drift_base = t3.get("drift_baseline", 0.84)
    acc_str = f"{acc*100:.1f}% +/- {std*100:.1f}%" if isinstance(acc, float) else str(acc)
    t3_str = "PASS" if t3_pass else "FAIL"
    report_lines += [
        f"Test 3 -- k-NN Classification Accuracy",
        f"  Hypothesis: MSSP k-NN >= 84% (DRIFT baseline)",
        f"  MSSP accuracy: {acc_str}",
        f"  DRIFT baseline: {drift_base*100:.1f}%",
        f"  Result: {t3_str}",
        "",
    ]

    # Test 4
    t4 = test_results.get("test4", {})
    t4_pass = t4.get("PASS", True)
    report_lines += [
        "Test 4 -- Multi-Scale PSD Visualization",
        "  Result: Generated (see test4_multiscale_psd.png)",
        "",
    ]

    # Summary
    overall_str = "ALL TESTS PASSED" if overall_pass else "SOME TESTS FAILED"
    report_lines += [
        "=" * 70,
        f"OVERALL: {overall_str}",
        "=" * 70,
        "",
        "Note: In MOCK mode, results are random and expected to fail.",
        "Run with real data and ADM checkpoint for meaningful results.",
    ]

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "step_b_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Also save JSON
    json_path = os.path.join(output_dir, "step_b_results.json")
    with open(json_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    print("\n" + report_text)
    logger.info("Report saved: %s", report_path)
    logger.info("JSON results: %s", json_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MSSP Step B: Statistical validation of multi-scale probing."
    )
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode (no GPU or checkpoint required).")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset root (AIGCDetectBenchmark layout).")
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to ADM checkpoint .pt file.")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Max images to load per category.")
    parser.add_argument("--output_dir", type=str, default="./results/step_b_mssp",
                        help="Directory for output files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="PyTorch device (cuda/cpu).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for feature extraction.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--skip_tests", type=str, default="",
                        help="Comma-separated list of tests to skip (e.g. '1,2').")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(
        name="MSSP",
        log_level="INFO",
        log_dir=args.output_dir,
        log_filename="step_b_mssp.log",
    )
    logger.info("MSSP Step B Validation")
    logger.info("Args: %s", vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    skip_tests = set(args.skip_tests.split(",")) if args.skip_tests else set()

    # ---------------------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------------------
    if args.mock or args.data_dir is None:
        logger.info("Loading mock data...")
        args.mock = True
        data = load_images_mock(args.num_samples, args.image_size)
    else:
        logger.info("Loading data from %s...", args.data_dir)
        data = load_images_from_dir(args.data_dir, args.num_samples, args.image_size)

    logger.info(
        "Data loaded: %d categories, %s",
        len(data),
        {k: v.shape[0] for k, v in data.items()},
    )

    # ---------------------------------------------------------------------------
    # 2. Initialise backbone and extractor
    # ---------------------------------------------------------------------------
    model_path = "" if args.mock else args.model_path
    logger.info("Initialising MSSPBackbone (mock=%s)...", args.mock or not model_path)

    backbone = MSSPBackbone(
        model_path=model_path,
        device=args.device,
        image_size=args.image_size,
        probe_timesteps=PROBE_TIMESTEPS,
        noise_seed=args.seed,
    )

    extractor = MSSPFeatureExtractor(
        backbone=backbone,
        probe_timesteps=PROBE_TIMESTEPS,
        n_freq_bands=N_FREQ_BANDS,
        normalize_features=True,
    )

    logger.info("Feature extractor: dim=%d", extractor.feature_dim)

    # ---------------------------------------------------------------------------
    # 3. Extract features
    # ---------------------------------------------------------------------------
    logger.info("Extracting 185D MSSP features...")
    features, labels, category_names = extract_all_features(
        data, extractor, batch_size=args.batch_size, device=args.device
    )

    # Cache features
    npz_path = os.path.join(args.output_dir, "mssp_features.npz")
    np.savez(npz_path, features=features, labels=labels,
             category_names=np.array(category_names))
    logger.info("Features cached: %s (shape=%s)", npz_path, features.shape)

    # ---------------------------------------------------------------------------
    # 4. Extract raw residuals for PSD analysis (Tests 1, 2, 4)
    # ---------------------------------------------------------------------------
    logger.info("Extracting raw residuals for PSD analysis...")
    residuals = extract_residuals_per_scale(
        data, backbone, PROBE_TIMESTEPS,
        batch_size=args.batch_size, device=args.device
    )

    # ---------------------------------------------------------------------------
    # 5. Run tests
    # ---------------------------------------------------------------------------
    test_results = {}

    if "1" not in skip_tests:
        test_results["test1"] = run_test1_gan_hf_psd(residuals, args.output_dir)

    if "2" not in skip_tests:
        test_results["test2"] = run_test2_vae_artifact(residuals, args.output_dir)

    if "3" not in skip_tests:
        test_results["test3"] = run_test3_knn_accuracy(
            features, labels, category_names, args.output_dir,
            k=5, n_splits=5, seed=args.seed
        )

    if "4" not in skip_tests:
        test_results["test4"] = run_test4_multiscale_psd(residuals, args.output_dir)

    # ---------------------------------------------------------------------------
    # 6. Generate report
    # ---------------------------------------------------------------------------
    elapsed = time.time() - t_start
    generate_report(test_results, args.output_dir, args, elapsed)
    logger.info("Total time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
