# MSSP: Multi-Scale Single-Step Probing

**Multi-Scale Single-Step Probing** (MSSP) is a deepfake detection method that probes images at multiple noise levels using a single UNet forward pass per level, preserving frequency-domain fingerprints that are lost in full DDIM inversion.

## Core Idea

DRIFT performs 20-step DDIM inversion (x₀ → x_T), which "washes away" VAE quantization artifacts (DRIFT G5 z-score = 0.0013). MSSP instead:

1. Adds fixed noise at 5 levels: T_set = {200, 400, 600, 800, 999}
2. Makes **one** UNet forward pass per level (5 total vs DRIFT's 20)
3. Extracts the score residual r_i = x_t - ε̂_t
4. Computes 185D features from residuals across all scales

```
for t_i in [200, 400, 600, 800, 999]:
    x_{t_i} = √ᾱ_{t_i} · x₀ + √(1-ᾱ_{t_i}) · ε_i    # forward noising
    ε̂_{t_i} = UNet_ADM(x_{t_i}, t_i)                  # single forward pass
    r_i      = x_{t_i} - ε̂_{t_i}                       # score residual

f_MSSP = Concat([F_norm(5D), F_psd(120D), F_stat(60D)])  → 185D
```

## Method Advantages

| | DIRE | DRIFT (F1) | **MSSP** |
|--|------|-----------|---------|
| UNet calls | 80 | 20 | **5** |
| Inference speed | ~12s/img | ~3s/img | **~0.75s/img** |
| VAE artifact detection | Low | Failed (washed away) | **Direct at t=400-600** |
| GAN artifact detection | Medium | Weak | **Direct at t=200** |

## Project Structure

```
MSSP/
├── configs/mssp_default.yaml          # Hyperparameters
├── models/
│   ├── backbone/adm_wrapper.py        # MSSPBackbone with probe() method
│   ├── features/mssp.py               # 185D feature extractor
│   └── heads/binary.py                # Binary classification head
├── data/
│   ├── dataloader.py                  # AIGCDetectBenchmark data loader
│   └── transforms.py                  # Image transforms ([-1,1] for ADM)
├── utils/
│   ├── logger.py                      # Structured logging
│   └── visualization.py              # PSD plots, t-SNE
├── experiments/
│   ├── step_b_mssp_validation.py      # Core validation script
│   └── step_b_mssp.sh                 # One-click runner
└── results/                           # Output directory
```

## Quick Start

### Mock Mode (No GPU/Checkpoint Required)

```bash
cd /path/to/AIGCDetectBenchmark-main/MSSP
bash experiments/step_b_mssp.sh --mock --num_samples 20
```

This runs in ~1 minute on CPU with random data. Validates:
- Level 0: Syntax check (py_compile)
- Level 1: Import check
- Level 2: Full pipeline with mock data

### Real Data Mode

```bash
bash experiments/step_b_mssp.sh \
    --data_dir /path/to/data \
    --model_path /path/to/256x256_diffusion_uncond.pt \
    --num_samples 200 \
    --device cuda
```

**ADM Checkpoint:** Download from:
```
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

**Data Structure:**
```
data_dir/
├── ProGAN/
│   ├── 0_real/*.png
│   └── 1_fake/*.png
├── SD_v1.5/
│   └── 1_fake/*.png
└── Wukong/
    └── 1_fake/*.png
```

## Validation Hypotheses (Step B)

| Test | Hypothesis | Threshold | Status |
|------|-----------|-----------|--------|
| Test 1 | GAN high-freq PSD ratio (t=200) | ProGAN/Real > 2× | TBD |
| Test 2 | SD VAE 8px artifact z-score (t=400-600) | > 2.0 (vs DRIFT G5: 0.0013) | TBD |
| Test 3 | k-NN classification accuracy | ≥ 84% (DRIFT baseline) | TBD |
| Test 4 | Multi-scale PSD visualization | Generated | TBD |

## Feature Dimensions

```
F_norm  [5D]   : ||r_i||₂ / ||x_t||₂ for each of 5 noise scales
F_psd   [120D] : radial PSD in 8 freq bands × 3 channels × 5 scales
F_stat  [60D]  : mean, std, skew, kurtosis × 3 channels × 5 scales
─────────────────────────────────────────────────────────────────
Total   [185D]
```

## Outputs

Running `step_b_mssp_validation.py` produces:
- `step_b_report.txt` — PASS/FAIL per hypothesis
- `step_b_results.json` — Numerical results
- `test1_gan_hf_psd.png` — GAN high-frequency analysis
- `test2_vae_artifact.png` — SD VAE artifact detection
- `test3_knn_accuracy.png` — k-NN confusion matrix
- `test4_multiscale_psd.png` — Multi-scale PSD overview
- `mssp_features.npz` — Cached 185D features
- `step_b_mssp.log` — Detailed log

## Citation Context

Based on experimental results from DRIFT_new (AIGCDetectBenchmark-main/DRIFT_new/).
Key insights driving MSSP design:
- DRIFT G5 VAE z-score = 0.0013 (failure of endpoint detection)
- Wasserstein inter/intra ratio = 4.07 (signal exists but spread across dimensions)
- hardcross k-NN = 84.0% ± 3.6% (MSSP target to beat)
