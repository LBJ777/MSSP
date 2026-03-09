"""
step_b_1c_progan_freq.py
========================
Three frequency-domain ProGAN detection methods applied to cached MSSP data.

Methods:
  A - Directional PSD   : H/V/diagonal spectral energy + anisotropy ratios
  B - 2D Peak Detection : stride-2/4/8 periodic spot intensity vs background
  C - NNF               : Gaussian-blur residual radial PSD + directional + autocorr

No GPU required — operates entirely on 1A residual cache.
"""

import argparse
import json
import os
import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def method_a_features(images: np.ndarray, inner: int = 8, outer: int = 48) -> np.ndarray:
    """Directional PSD features.

    images : [N, C, H, W] float32
    Returns: [N, 10]  — horiz, vert, diag1, diag2 (mean over channels),
                         H/V ratio, diag1/horiz ratio,
                         + std of each 4-dir energy (anisotropy spread)
    """
    N, C, H, W = images.shape
    feats = np.zeros((N, 10), dtype=np.float32)

    for i in range(N):
        h_vals, v_vals, d1_vals, d2_vals = [], [], [], []
        for c in range(C):
            ch = images[i, c].astype(np.float64)
            fft2 = np.fft.fftshift(np.fft.fft2(ch))
            power = np.abs(fft2) ** 2
            cx, cy = H // 2, W // 2

            # ensure bounds
            o = min(outer, cx - 1, cy - 1)
            ins = min(inner, o - 1)

            horiz = power[cx, cy + ins: cy + o].mean()
            vert  = power[cx + ins: cx + o, cy].mean()
            diag1 = np.array([power[cx + k, cy + k] for k in range(ins, o)]).mean()
            diag2 = np.array([power[cx - k, cy + k] for k in range(ins, o)]).mean()

            h_vals.append(horiz); v_vals.append(vert)
            d1_vals.append(diag1); d2_vals.append(diag2)

        h = np.mean(h_vals); v = np.mean(v_vals)
        d1 = np.mean(d1_vals); d2 = np.mean(d2_vals)

        feats[i, 0] = np.log1p(h)
        feats[i, 1] = np.log1p(v)
        feats[i, 2] = np.log1p(d1)
        feats[i, 3] = np.log1p(d2)
        feats[i, 4] = h / (v + EPS)       # H/V anisotropy
        feats[i, 5] = d1 / (h + EPS)      # diag/horiz
        # std across 4 directions (log scale)
        dirs_log = np.log1p([h, v, d1, d2])
        feats[i, 6] = np.std(dirs_log)
        # symmetry: |h-v| / (h+v) and |d1-d2|/(d1+d2)
        feats[i, 7] = abs(h - v) / (h + v + EPS)
        feats[i, 8] = abs(d1 - d2) / (d1 + d2 + EPS)
        # total HF energy
        feats[i, 9] = np.log1p(h + v + d1 + d2)

    return feats


def method_b_features(images: np.ndarray, bg_radius: int = 3) -> np.ndarray:
    """2D spectral peak detection for periodic artifacts.

    Checks 3 candidate strides (2, 4, 8) → offsets H//4, H//8, H//16.
    For each stride: mean of 8 symmetric peak positions vs local background.

    images : [N, C, H, W] float32
    Returns: [N, 9]  — 3 strides × (abs_peak_log, peak/bg_ratio, 8-pos std)
    """
    N, C, H, W = images.shape
    feats = np.zeros((N, 9), dtype=np.float32)
    strides = [2, 4, 8]

    for i in range(N):
        avg_power = np.zeros((H, W), dtype=np.float64)
        for c in range(C):
            ch = images[i, c].astype(np.float64)
            fft2 = np.fft.fftshift(np.abs(np.fft.fft2(ch)))
            avg_power += fft2 ** 2
        avg_power /= C
        cx, cy = H // 2, W // 2

        for si, stride in enumerate(strides):
            offset = H // (2 * stride)   # H//4 for stride=2, H//8 for stride=4 …
            offset = max(offset, 1)

            positions = [
                avg_power[cx + offset, cy],
                avg_power[cx - offset, cy],
                avg_power[cx, cy + offset],
                avg_power[cx, cy - offset],
                avg_power[min(cx + offset, H-1), min(cy + offset, W-1)],
                avg_power[max(cx - offset, 0), max(cy - offset, 0)],
                avg_power[min(cx + offset, H-1), max(cy - offset, 0)],
                avg_power[max(cx - offset, 0), min(cy + offset, W-1)],
            ]
            peak_mean = np.mean(positions)

            # background: annulus around DC, avoiding the 8 peak positions
            r = bg_radius
            bg_patch = avg_power[cx - r: cx + r + 1, cy - r: cy + r + 1].mean()

            feats[i, si * 3 + 0] = np.log1p(peak_mean)
            feats[i, si * 3 + 1] = peak_mean / (bg_patch + EPS)
            feats[i, si * 3 + 2] = np.std(np.log1p(positions))

    return feats


def _radial_psd(channel: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute 1D radial PSD for a single channel [H, W]."""
    H, W = channel.shape
    fft2 = np.fft.fft2(channel)
    power = np.abs(fft2) ** 2
    power = np.fft.fftshift(power)

    cx, cy = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = min(cx, cy)

    bins = np.linspace(0, max_r, n_bins + 1)
    psd = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (R >= bins[b]) & (R < bins[b + 1])
        if mask.sum() > 0:
            psd[b] = power[mask].mean()
    return np.log1p(psd)


def method_c_features(
    images: np.ndarray,
    sigma: float = 2.0,
    n_psd_bins: int = 64,
    max_lag: int = 16,
) -> np.ndarray:
    """NNF (noise-noise fingerprint) features on x₀.

    n = x₀ - gaussian_blur(x₀, sigma)
    Features per image:
      - radial PSD of n (n_psd_bins)
      - directional PSD of n (4 energies + 2 ratios = 6)
      - mean autocorrelation profile lags 1..max_lag (max_lag dims)

    images : [N, C, H, W]
    Returns: [N, n_psd_bins + 6 + max_lag]
    """
    N, C, H, W = images.shape
    feat_dim = n_psd_bins + 6 + max_lag
    feats = np.zeros((N, feat_dim), dtype=np.float32)

    inner, outer = 8, 48
    outer = min(outer, H // 2 - 1)

    for i in range(N):
        radial_acc = np.zeros(n_psd_bins)
        h_acc = v_acc = d1_acc = d2_acc = 0.0
        acorr_acc = np.zeros(max_lag)

        for c in range(C):
            ch = images[i, c].astype(np.float64)
            smooth = gaussian_filter(ch, sigma=sigma)
            n = ch - smooth                          # noise residual

            # radial PSD
            radial_acc += _radial_psd(n, n_psd_bins)

            # directional PSD
            fft2 = np.fft.fftshift(np.fft.fft2(n))
            power = np.abs(fft2) ** 2
            cx, cy = H // 2, W // 2
            ins = min(inner, outer - 1)
            h_acc  += power[cx, cy + ins: cy + outer].mean()
            v_acc  += power[cx + ins: cx + outer, cy].mean()
            d1_acc += np.array([power[cx + k, cy + k] for k in range(ins, outer)]).mean()
            d2_acc += np.array([power[cx - k, cy + k] for k in range(ins, outer)]).mean()

            # autocorrelation along rows (mean over rows and cols)
            row_acorr = np.zeros(max_lag)
            for row in range(0, H, 8):   # sample every 8 rows for speed
                r = n[row]
                ac = np.correlate(r, r, mode='full')
                ac = ac[len(r) - 1:]     # lags 0, 1, 2, ...
                ac = ac / (ac[0] + EPS)
                row_acorr += ac[1: max_lag + 1]
            row_acorr /= (H // 8)
            acorr_acc += row_acorr

        radial_acc /= C
        h = h_acc / C; v = v_acc / C; d1 = d1_acc / C; d2 = d2_acc / C
        acorr_acc /= C

        feats[i, :n_psd_bins] = radial_acc
        feats[i, n_psd_bins]     = np.log1p(h)
        feats[i, n_psd_bins + 1] = np.log1p(v)
        feats[i, n_psd_bins + 2] = np.log1p(d1)
        feats[i, n_psd_bins + 3] = np.log1p(d2)
        feats[i, n_psd_bins + 4] = h / (v + EPS)
        feats[i, n_psd_bins + 5] = d1 / (h + EPS)
        feats[i, n_psd_bins + 6: n_psd_bins + 6 + max_lag] = acorr_acc

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def binary_auc_knn(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    k: int = 5,
    n_splits: int = 5,
) -> tuple:
    """5-fold stratified k-NN AUC for binary classification (pos vs neg)."""
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))

    # replace nan/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_va)[:, 1]
        try:
            aucs.append(roc_auc_score(y_va, proba))
        except ValueError:
            aucs.append(0.5)

    return float(np.mean(aucs)), float(np.std(aucs))


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(cache_path: str):
    """Return dict of arrays: x0__{cat}, res__{cat}__{t}"""
    data = np.load(cache_path, allow_pickle=True)
    return dict(data)


def get_space_images(data: dict, space: str, cat: str) -> np.ndarray:
    """Return [N,C,H,W] for given (space, category)."""
    if space == "x0":
        key = f"x0__{cat}"
    else:
        t = space.split("_")[1]   # "res_200" → "200"
        key = f"res__{cat}__{t}"
    arr = data[key].astype(np.float32)
    # Normalize residuals (may be raw differences; x0 already in [-1,1] or [0,1])
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_a(data, categories, output_path):
    """4-panel: mean directional PSD spectra per category × 4 spaces."""
    spaces = ["x0", "res_200", "res_400", "res_600", "res_800"]
    space_labels = ["x₀", "r(t=200)", "r(t=400)", "r(t=600)", "r(t=800)"]
    dirs = ["Horizontal", "Vertical", "Diag 45°", "Diag 135°"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    colors = {"Real": "steelblue", "ProGAN": "crimson",
              "SD_v1.5": "darkorange", "Wukong": "forestgreen"}

    for ax_idx, (space, slabel) in enumerate(zip(spaces[:6], space_labels[:6])):
        ax = axes[ax_idx] if ax_idx < len(axes) else None
        if ax is None:
            break
        # compute mean directional energies per category
        for cat in categories:
            imgs = get_space_images(data, space, cat)
            feats = method_a_features(imgs)   # [N,10]
            # log-energies: cols 0-3
            means = feats[:, :4].mean(axis=0)
            ax.bar(
                np.arange(4) + list(categories).index(cat) * 0.2,
                means,
                width=0.18,
                label=cat,
                color=colors.get(cat, "grey"),
                alpha=0.8,
            )
        ax.set_xticks(np.arange(4) + 0.3)
        ax.set_xticklabels(dirs, fontsize=8)
        ax.set_title(slabel, fontsize=10)
        ax.set_ylabel("log(1+energy)")
        if ax_idx == 0:
            ax.legend(fontsize=7)

    # remove unused axes
    for ax in axes[len(spaces):]:
        ax.set_visible(False)

    fig.suptitle("Method A — Directional PSD (per space)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_method_b(data, categories, output_path):
    """Show 2D FFT of mean image per category for a quick visual."""
    spaces_to_show = ["x0", "res_200", "res_600"]
    space_labels = ["x₀", "r(t=200)", "r(t=600)"]

    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats, len(spaces_to_show), figsize=(14, 3 * n_cats))

    for ci, cat in enumerate(categories):
        for si, (space, slabel) in enumerate(zip(spaces_to_show, space_labels)):
            imgs = get_space_images(data, space, cat)
            mean_img = imgs.mean(axis=0)   # [C,H,W]
            # mean power spectrum over channels
            H, W = mean_img.shape[1], mean_img.shape[2]
            avg_ps = np.zeros((H, W))
            for c in range(mean_img.shape[0]):
                fft2 = np.fft.fftshift(np.abs(np.fft.fft2(mean_img[c])))
                avg_ps += fft2 ** 2
            avg_ps /= mean_img.shape[0]
            log_ps = np.log1p(avg_ps)

            ax = axes[ci, si] if n_cats > 1 else axes[si]
            ax.imshow(log_ps, cmap="inferno", origin="lower")
            ax.set_title(f"{cat} | {slabel}", fontsize=8)
            ax.axis("off")

    fig.suptitle("Method B — 2D FFT Power Spectrum (mean image per category)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_method_c(data, categories, output_path, n_psd_bins=64):
    """NNF radial PSD + autocorrelation per category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Real": "steelblue", "ProGAN": "crimson",
              "SD_v1.5": "darkorange", "Wukong": "forestgreen"}

    for cat in categories:
        imgs = get_space_images(data, "x0", cat)
        feats = method_c_features(imgs, n_psd_bins=n_psd_bins)
        # radial PSD: cols 0..n_psd_bins
        mean_psd = feats[:, :n_psd_bins].mean(axis=0)
        axes[0].plot(mean_psd, label=cat, color=colors.get(cat, "grey"), linewidth=2)
        # autocorr: last 16 cols
        mean_ac = feats[:, n_psd_bins + 6:].mean(axis=0)
        axes[1].plot(mean_ac, label=cat, color=colors.get(cat, "grey"), linewidth=2)

    axes[0].set_title("NNF Radial PSD (x₀ only)", fontsize=11)
    axes[0].set_xlabel("Frequency bin"); axes[0].set_ylabel("log(1+power)")
    axes[0].legend()

    axes[1].set_title("NNF Autocorrelation (lag 1–16)", fontsize=11)
    axes[1].set_xlabel("Lag"); axes[1].set_ylabel("Mean autocorr")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[1].legend()

    fig.suptitle("Method C — NNF (Gaussian-blur noise residual)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_summary_heatmap(auc_table: dict, spaces: list, output_path: str):
    """AUC heatmap: rows=methods, cols=spaces."""
    methods = ["A_directional", "B_peak", "C_nnf"]
    method_labels = ["A: Directional PSD", "B: 2D Peak", "C: NNF"]

    nrows = len(methods)
    ncols = len(spaces)
    mat = np.zeros((nrows, ncols))
    for ri, method in enumerate(methods):
        for ci, space in enumerate(spaces):
            key = f"{method}__{space}"
            mat[ri, ci] = auc_table.get(key, {}).get("auc", 0.5)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="AUC (ProGAN vs Real)")

    space_labels = []
    for s in spaces:
        if s == "x0":
            space_labels.append("x₀")
        else:
            t = s.split("_")[1]
            space_labels.append(f"r(t={t})")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(space_labels, fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(method_labels, fontsize=10)
    ax.set_title("ProGAN vs Real — AUC by Method & Input Space", fontsize=12, fontweight="bold")

    for ri in range(nrows):
        for ci in range(ncols):
            val = mat[ri, ci]
            std_key = f"{methods[ri]}__{spaces[ci]}"
            std = auc_table.get(std_key, {}).get("std", 0.0)
            txt = f"{val:.3f}\n±{std:.3f}"
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=8, color="black" if 0.6 < val < 0.9 else "white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="1C ProGAN frequency detection")
    p.add_argument("--cache_path", type=str,
                   default="results/step_b_1a/1a_residuals_cache.npz")
    p.add_argument("--output_dir", type=str, default="results/step_b_1c")
    p.add_argument("--n_bins", type=int, default=64)
    p.add_argument("--k_nn", type=int, default=5)
    p.add_argument("--n_splits", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1C] Loading cache from {args.cache_path}")
    data = load_cache(args.cache_path)

    # Determine available categories and spaces
    categories = []
    for key in data.keys():
        if key.startswith("x0__"):
            cat = key[4:]
            categories.append(cat)
    categories = sorted(categories)
    print(f"[1C] Categories: {categories}")

    timesteps = [200, 400, 600, 800, 999]
    spaces = ["x0"] + [f"res_{t}" for t in timesteps]

    # Check which spaces are actually in cache
    available_spaces = ["x0"]
    for t in timesteps:
        if f"res__Real__{t}" in data:
            available_spaces.append(f"res_{t}")
    print(f"[1C] Available spaces: {available_spaces}")

    # ── Method A ──────────────────────────────────────────────────────────────
    print("\n[1C] === Method A: Directional PSD ===")
    auc_table = {}

    for space in available_spaces:
        X_progan = method_a_features(get_space_images(data, space, "ProGAN"))
        X_real   = method_a_features(get_space_images(data, space, "Real"))
        auc, std = binary_auc_knn(X_progan, X_real, k=args.k_nn, n_splits=args.n_splits)
        key = f"A_directional__{space}"
        auc_table[key] = {"auc": auc, "std": std}
        print(f"  {space:12s}  AUC={auc:.4f} ± {std:.4f}")

    # ── Method B ──────────────────────────────────────────────────────────────
    print("\n[1C] === Method B: 2D Peak Detection ===")
    for space in available_spaces:
        X_progan = method_b_features(get_space_images(data, space, "ProGAN"))
        X_real   = method_b_features(get_space_images(data, space, "Real"))
        auc, std = binary_auc_knn(X_progan, X_real, k=args.k_nn, n_splits=args.n_splits)
        key = f"B_peak__{space}"
        auc_table[key] = {"auc": auc, "std": std}
        print(f"  {space:12s}  AUC={auc:.4f} ± {std:.4f}")

    # ── Method C (x0 only) ────────────────────────────────────────────────────
    print("\n[1C] === Method C: NNF (x0 only) ===")
    X_progan = method_c_features(
        get_space_images(data, "x0", "ProGAN"),
        n_psd_bins=args.n_bins,
    )
    X_real = method_c_features(
        get_space_images(data, "x0", "Real"),
        n_psd_bins=args.n_bins,
    )
    auc, std = binary_auc_knn(X_progan, X_real, k=args.k_nn, n_splits=args.n_splits)
    auc_table["C_nnf__x0"] = {"auc": auc, "std": std}
    print(f"  {'x0':12s}  AUC={auc:.4f} ± {std:.4f}")
    # Fill C for other spaces as 0.5 (not applicable)
    for space in available_spaces:
        if space != "x0":
            auc_table[f"C_nnf__{space}"] = {"auc": 0.5, "std": 0.0}

    # ── Best results summary ──────────────────────────────────────────────────
    print("\n[1C] === Summary (ProGAN vs Real AUC) ===")
    best_auc, best_key = 0.0, ""
    for key, v in sorted(auc_table.items()):
        marker = ""
        if v["auc"] > 0.7:
            marker = " ◀ GOOD"
        if v["auc"] > best_auc:
            best_auc = v["auc"]
            best_key = key
        print(f"  {key:35s}  {v['auc']:.4f} ± {v['std']:.4f}{marker}")
    print(f"\n  Best: {best_key} → AUC={best_auc:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[1C] Generating plots …")
    plot_method_a(data, categories, os.path.join(args.output_dir, "1c_progan_method_a.png"))
    print("  Method A plot done.")
    plot_method_b(data, categories, os.path.join(args.output_dir, "1c_progan_method_b.png"))
    print("  Method B plot done.")
    plot_method_c(data, categories, os.path.join(args.output_dir, "1c_progan_method_c.png"),
                  n_psd_bins=args.n_bins)
    print("  Method C plot done.")

    summary_path = os.path.join(args.output_dir, "1c_progan_summary.png")
    mat = plot_summary_heatmap(auc_table, available_spaces, summary_path)
    print("  Summary heatmap done.")

    # ── JSON results ──────────────────────────────────────────────────────────
    results = {
        "categories": categories,
        "spaces": available_spaces,
        "auc_table": auc_table,
        "best": {"key": best_key, "auc": best_auc},
        "notes": {
            "A_directional": "4-dir energy + anisotropy ratios on x0 and residuals",
            "B_peak":        "stride-2/4/8 periodic spot peak/bg ratio on 2D FFT",
            "C_nnf":         "Gaussian-blur noise residual PSD+dir+autocorr on x0 only",
        },
    }
    json_path = os.path.join(args.output_dir, "1c_progan_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[1C] Results saved to {json_path}")
    print("[1C] Done.")


if __name__ == "__main__":
    main()
