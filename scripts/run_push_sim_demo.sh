#!/usr/bin/env bash
# Generate push trajectory dataset for ML training.
#
# 10,000 environments × 4 pieces each = 40,000 trajectories
# across 40,000 unique polygon shapes.
#
# Usage:
#   bash scripts/run_push_sim_demo.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="assets/push_data"

# ── Hardcoded hyperparameters ──────────────────────────────────────
NUM_ENVS=10000          # number of environments
PIECES_PER_ENV=4        # pieces per environment
N_VERTS_MIN=4           # polygon vertex count range
N_VERTS_MAX=10
SCALE=0.05              # polygon size (radius in metres)
STEPS_MIN=30            # trajectory length range (biased long)
STEPS_MAX=150
PUSH_DIST_MIN=0.01      # push distance range (metres)
PUSH_DIST_MAX=0.08
START_POSE_RANGE=0.2    # start xy sampling bound (metres)
SEED=0
# ───────────────────────────────────────────────────────────────────

TOTAL_TRAJS=$(( NUM_ENVS * PIECES_PER_ENV ))

echo "=== Push dataset generation ==="
echo "  ${NUM_ENVS} envs × ${PIECES_PER_ENV} pieces = ${TOTAL_TRAJS} trajectories"
echo "  Step lengths: [${STEPS_MIN}, ${STEPS_MAX}] (Beta-biased long)"
echo "  Push distances: [${PUSH_DIST_MIN}, ${PUSH_DIST_MAX}] m"
echo "  Polygon verts: [${N_VERTS_MIN}, ${N_VERTS_MAX}]"
echo "  Output: ${OUTPUT_DIR}/"
echo ""

python scripts/generate_push_dataset.py \
    --num_envs "$NUM_ENVS" \
    --pieces_per_env "$PIECES_PER_ENV" \
    --n_verts_min "$N_VERTS_MIN" \
    --n_verts_max "$N_VERTS_MAX" \
    --scale "$SCALE" \
    --steps_min "$STEPS_MIN" \
    --steps_max "$STEPS_MAX" \
    --push_dist_min "$PUSH_DIST_MIN" \
    --push_dist_max "$PUSH_DIST_MAX" \
    --start_pose_range "$START_POSE_RANGE" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

TOTAL_FILES=$(find "$OUTPUT_DIR" -name "env_*.npz" | wc -l)
echo ""
echo "=== Done: ${TOTAL_FILES} env files in ${OUTPUT_DIR}/ ==="
