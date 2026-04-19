#!/usr/bin/env bash
# Run UNet+CBS, RRT-Connect+CBS, and UNet-only on the held-out test set,
# then print a metrics table and reorganise videos per method
# (viz_by_method/<method>/{SUCCESS,FAIL}_<episode>.mp4).
#
# Usage:
#   scripts/run_test_benchmark.sh [TEST_DIR] [OUTPUT_DIR] [CHECKPOINT] [N_EVAL]
#
# Defaults:
#   TEST_DIR    = assets/data_test/ep100_pts3-5_seed12345
#   OUTPUT_DIR  = eval_output/unet_benchmark
#   CHECKPOINT  = checkpoints/unet_best.pt
#   N_EVAL      = 50

set -euo pipefail

TEST_DIR="${1:-assets/data_test/ep100_pts3-5_seed12345}"
OUTPUT_DIR="${2:-eval_output/unet_benchmark}"
CHECKPOINT="${3:-checkpoints/unet_best.pt}"
N_EVAL="${4:-50}"

if [[ ! -d "$TEST_DIR" ]]; then
    echo "ERROR: test dir not found: $TEST_DIR" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "==========================================================="
echo "Benchmark config"
echo "  TEST_DIR   : $TEST_DIR"
echo "  OUTPUT_DIR : $OUTPUT_DIR"
echo "  CHECKPOINT : $CHECKPOINT"
echo "  N_EVAL     : $N_EVAL"
echo "  METHODS    : diff_cbs (UNet+CBS), rrt_cbs (RRT+CBS), diff (UNet only)"
echo "==========================================================="

python scripts/eval/compare_methods.py \
    --data_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --methods diff_cbs rrt_cbs diff \
    --model unet \
    --checkpoint "$CHECKPOINT" \
    --n_eval "$N_EVAL" \
    --max_nodes 500 \
    --rrt_max_samples 5000 \
    --rrt_max_time 10.0 \
    --save_viz

echo
echo "==========================================================="
echo "Aggregated results"
echo "==========================================================="
python scripts/eval/summarize_benchmark.py --output_dir "$OUTPUT_DIR"

echo
echo "Done. Browse videos at: $OUTPUT_DIR/viz_by_method/"
echo "  diff_cbs/  — UNet + CBS"
echo "  rrt_cbs/   — RRT-Connect + CBS"
echo "  diff/      — UNet only"
echo "Side-by-side comparison videos: $OUTPUT_DIR/viz/*_comparison.mp4"
