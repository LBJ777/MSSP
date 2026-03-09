"""
utils/visualization.py
-----------------------
Visualization utilities for MSSP experiments.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_multiscale_psd_comparison(
    category_psd: Dict[str, Dict[int, np.ndarray]],
    n_freq_bands: int = 8,
    output_path: Optional[str] = None,
    title: str = "MSSP: Multi-Scale PSD Comparison",
) -> plt.Figure:
    """Plot PSD profiles for all categories across all probe timesteps.

    Args:
        category_psd: Dict[cat_name → Dict[timestep → [n_bands] array]].
        n_freq_bands: Number of frequency bands.
        output_path: If set, save the figure to this path.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    categories = sorted(category_psd.keys())
    all_timesteps = sorted(set(
        t for cat_data in category_psd.values() for t in cat_data
    ))

    band_centers = np.linspace(
        0.5 / n_freq_bands, 0.5 - 0.5 / n_freq_bands, n_freq_bands
    )

    fig, axes = plt.subplots(
        1, len(all_timesteps),
        figsize=(4 * len(all_timesteps), 4),
        squeeze=False,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    for col_idx, t in enumerate(all_timesteps):
        ax = axes[0][col_idx]
        for cat_idx, cat_name in enumerate(categories):
            if t not in category_psd[cat_name]:
                continue
            psd = category_psd[cat_name][t]  # [n_bands]
            ax.plot(band_centers, psd, color=colors[cat_idx],
                    label=cat_name, linewidth=2, marker="o", markersize=4)

        # Mark VAE and high-freq boundaries
        vae_freq = 0.125  # f=H/8
        ax.axvline(x=vae_freq, color="purple", linestyle="--", alpha=0.5, label="f=H/8")
        ax.axvline(x=0.25, color="orange", linestyle=":", alpha=0.5, label="HF start")

        ax.set_xlabel("Normalized Radial Frequency")
        ax.set_ylabel("Log Power")
        ax.set_title(f"t={t}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    category_names: List[str],
    output_path: Optional[str] = None,
    title: str = "MSSP: t-SNE of 185D Features",
    seed: int = 42,
) -> plt.Figure:
    """2D t-SNE visualization of MSSP feature space.

    Args:
        features: [N, 185] feature matrix.
        labels: [N] integer labels.
        category_names: List mapping label index → name.
        output_path: Save path.
        title: Plot title.
        seed: Random seed for t-SNE.

    Returns:
        matplotlib Figure.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(features) - 1))
    emb = tsne.fit_transform(features)  # [N, 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(category_names)))

    for cat_idx, cat_name in enumerate(category_names):
        mask = labels == cat_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[colors[cat_idx]], label=cat_name,
            alpha=0.7, s=20, edgecolors="none",
        )

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
