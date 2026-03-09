#!/usr/bin/env bash
# experiments/step_b_mssp.sh
# One-click runner for MSSP Step B validation.
#
# Usage:
#   # Mock mode (no GPU/checkpoint, ~1-5 min):
#   bash experiments/step_b_mssp.sh --mock
#
#   # Real data mode (requires ADM checkpoint and data):
#   bash experiments/step_b_mssp.sh \
#       --data_dir /path/to/AIGCDetectBenchmark \
#       --model_path /path/to/256x256_diffusion_uncond.pt \
#       --num_samples 200
#
#   # With custom output dir:
#   bash experiments/step_b_mssp.sh --mock --output_dir ./my_results
#
# The script must be run from the MSSP/ root directory.

set -e

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MOCK=""
DATA_DIR=""
MODEL_PATH=""
NUM_SAMPLES=50
OUTPUT_DIR="./results/step_b_mssp"
DEVICE="cpu"
BATCH_SIZE=4
IMAGE_SIZE=256
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock)           MOCK="--mock"; shift ;;
        --data_dir)       DATA_DIR="$2"; shift 2 ;;
        --model_path)     MODEL_PATH="$2"; shift 2 ;;
        --num_samples)    NUM_SAMPLES="$2"; shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --image_size)     IMAGE_SIZE="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Python interpreter: use aigc conda env if available
# ---------------------------------------------------------------------------
if conda run -n aigc python --version &>/dev/null 2>&1; then
    PYTHON="conda run -n aigc python"
    echo "[step_b_mssp.sh] Using conda env: aigc"
else
    PYTHON="python"
    echo "[step_b_mssp.sh] Using system python"
fi

# ---------------------------------------------------------------------------
# Auto-detect CUDA
# ---------------------------------------------------------------------------
if [[ -z "$MOCK" && "$DEVICE" == "cpu" ]]; then
    if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="cuda"
        echo "[step_b_mssp.sh] CUDA available -> using GPU"
    else
        echo "[step_b_mssp.sh] CUDA not available -> using CPU"
    fi
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  MSSP Step B: Multi-Scale Single-Step Probing Validation"
echo "============================================================"
echo "  Mode:        ${MOCK:+MOCK}${MOCK:-REAL}"
echo "  Samples:     $NUM_SAMPLES per category"
echo "  Output:      $OUTPUT_DIR"
echo "  Device:      $DEVICE"
echo ""

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Level 0: Syntax check
# ---------------------------------------------------------------------------
echo "[Level 0] Syntax check..."
$PYTHON -m py_compile experiments/step_b_mssp_validation.py
$PYTHON -m py_compile models/backbone/adm_wrapper.py
$PYTHON -m py_compile models/features/mssp.py
$PYTHON -m py_compile models/features/base.py
$PYTHON -m py_compile models/heads/binary.py
$PYTHON -m py_compile data/dataloader.py
$PYTHON -m py_compile data/transforms.py
$PYTHON -m py_compile utils/logger.py
echo "  [PASS] All files pass syntax check."

# ---------------------------------------------------------------------------
# Level 1: Import check
# ---------------------------------------------------------------------------
echo "[Level 1] Import check..."
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from models.backbone.adm_wrapper import MSSPBackbone
from models.features.mssp import MSSPFeatureExtractor
from models.features.base import FeatureExtractor
from models.heads.binary import BinaryDetectionHead
from data.dataloader import DRIFTDataLoader
from data.transforms import get_transforms, denormalize, get_adm_transforms
from utils.logger import setup_logger, get_logger
print('  [PASS] All imports successful.')
print('  MSSPBackbone:', MSSPBackbone)
print('  MSSPFeatureExtractor:', MSSPFeatureExtractor)
print('  Feature dim:', MSSPFeatureExtractor(
    MSSPBackbone(model_path='mock', device='cpu'),
    n_freq_bands=8
).feature_dim)
"

# ---------------------------------------------------------------------------
# Level 2: Run validation
# ---------------------------------------------------------------------------
echo "[Level 2] Running validation..."

if [[ -n "$MOCK" ]]; then
    # Mock mode
$PYTHON experiments/step_b_mssp_validation.py \
        --mock \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --device cpu \
        --batch_size "$BATCH_SIZE" \
        --image_size "$IMAGE_SIZE" \
        --seed "$SEED"
else
    # Real mode
    EXTRA_ARGS=""
    if [[ -n "$DATA_DIR" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --data_dir $DATA_DIR"
    fi
    if [[ -n "$MODEL_PATH" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --model_path $MODEL_PATH"
    fi

$PYTHON experiments/step_b_mssp_validation.py \
        $EXTRA_ARGS \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --image_size "$IMAGE_SIZE" \
        --seed "$SEED"
fi

echo ""
echo "============================================================"
echo "  Results saved to: $OUTPUT_DIR"
echo "  Report: $OUTPUT_DIR/step_b_report.txt"
echo "============================================================"
