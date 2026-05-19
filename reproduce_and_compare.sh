#!/usr/bin/env bash
# ============================================================
# Reproduce & Compare: SI2E vs Baselines (A2C+MiniGrid and DrQv2+DMControl)
#
# Paper: "Effective Exploration Based on Structural Information Principles"
#        NeurIPS 2024 — Zeng, Peng, Li
# SI2E repo: (cloned to SI2E/)
# Base repos: base-rl-starter-files/  base-torch-ac/  base-drqv2/
#
# Usage:
#   bash reproduce_and_compare.sh [a2c|drqv2|all]
#
# Outputs land in results/a2c/ and results/drqv2/
# Run compare_results.py afterwards to plot/tabulate.
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK="${1:-all}"
SEED=1
RESULTS="$SCRIPT_DIR/results"
mkdir -p "$RESULTS/a2c" "$RESULTS/drqv2"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
log() { echo -e "\n\033[1;32m>>> $*\033[0m"; }

# ------------------------------------------------------------------
# A2C / MiniGrid
# ------------------------------------------------------------------
run_a2c() {
    log "=== A2C / MiniGrid Experiments ==="
    log "Task: DoorKey-8x8  Frames: 500_000 (shortened for local reproduce)"
    A2C_DIR="$SCRIPT_DIR/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
    MODEL_BASE="$RESULTS/a2c"
    FRAMES=500000   # paper uses 3M; reduce here for local sanity-check

    cd "$A2C_DIR"
    export PYTHONPATH="$A2C_DIR:$PYTHONPATH"

    # --- Baseline: plain A2C ---
    log "[1/3] A2C baseline (no intrinsic reward)"
    python3 -m scripts.train \
        --algo a2c \
        --env MiniGrid-DoorKey-8x8-v0 \
        --model "$MODEL_BASE/doorkey8x8-a2c-seed$SEED" \
        --save-interval 100 \
        --frames "$FRAMES" \
        --seed "$SEED" \
        --use_batch \
        2>&1 | tee "$MODEL_BASE/run_a2c_seed$SEED.log"

    # --- Baseline: VCSE ---
    log "[2/3] A2C + VCSE (plain kNN state entropy, no value-conditioning — run_vcse.sh)"
    python3 -m scripts.train \
        --algo a2c \
        --env MiniGrid-DoorKey-8x8-v0 \
        --model "$MODEL_BASE/doorkey8x8-vcse-seed$SEED" \
        --save-interval 100 \
        --frames "$FRAMES" \
        --use_entropy_reward \
        --seed "$SEED" \
        --beta 0.005 \
        --use_batch \
        2>&1 | tee "$MODEL_BASE/run_vcse_seed$SEED.log"

    # --- SI2E ---
    log "[3/3] A2C + SI2E (value-conditional structural entropy — run_si2e.sh)"
    python3 -m scripts.train \
        --algo a2c \
        --env MiniGrid-DoorKey-8x8-v0 \
        --model "$MODEL_BASE/doorkey8x8-si2e-seed$SEED" \
        --save-interval 100 \
        --frames "$FRAMES" \
        --use_entropy_reward \
        --use_value_condition \
        --seed "$SEED" \
        --beta 0.005 \
        --use_batch \
        2>&1 | tee "$MODEL_BASE/run_si2e_seed$SEED.log"

    log "A2C runs done. Logs in $MODEL_BASE/"
}

# ------------------------------------------------------------------
# DrQv2 / DMControl
# ------------------------------------------------------------------
run_drqv2() {
    log "=== DrQv2 / DMControl Experiments ==="
    log "Task: cartpole_swingup  Steps: 50_000 (paper uses 250K; shortened here)"
    DRQ_DIR="$SCRIPT_DIR/SI2E/SI2E_DrQv2"
    MODEL_BASE="$RESULTS/drqv2"

    cd "$DRQ_DIR"
    export MUJOCO_GL=egl   # headless rendering (no display required)

    # Reduced steps for local reproduce
    TOTAL_FRAMES=50000

    # --- Baseline: plain DrQv2 (no intrinsic reward) ---
    log "[1/3] DrQv2 baseline (no intrinsic reward)"
    python3 train.py \
        "agent._target_=drqv2.DrQV2Agent" \
        "task@_global_=cartpole_swingup" \
        seed="$SEED" \
        num_train_frames="$TOTAL_FRAMES" \
        "hydra.run.dir=$MODEL_BASE/drqv2-seed$SEED" \
        2>&1 | tee "$MODEL_BASE/run_drqv2_seed$SEED.log"

    # --- SE baseline: SI2EAgent without value-conditioning ---
    log "[2/3] DrQv2 + SE (state entropy, no value-conditioning — ablation baseline)"
    python3 train.py \
        "agent._target_=si2e.SI2EAgent" \
        "agent.do_vcse=false" \
        "task@_global_=cartpole_swingup" \
        seed="$SEED" \
        num_train_frames="$TOTAL_FRAMES" \
        "hydra.run.dir=$MODEL_BASE/se-seed$SEED" \
        2>&1 | tee "$MODEL_BASE/run_se_seed$SEED.log"

    # --- SI2E: value-conditional structural entropy (paper's method) ---
    log "[3/3] DrQv2 + SI2E (value-conditional structural entropy — paper's method)"
    python3 train.py \
        "agent._target_=si2e.SI2EAgent" \
        "agent.do_vcse=true" \
        "task@_global_=cartpole_swingup" \
        seed="$SEED" \
        num_train_frames="$TOTAL_FRAMES" \
        "hydra.run.dir=$MODEL_BASE/si2e-seed$SEED" \
        2>&1 | tee "$MODEL_BASE/run_si2e_seed$SEED.log"

    log "DrQv2 runs done. Logs in $MODEL_BASE/"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
case "$TASK" in
    a2c)   run_a2c ;;
    drqv2) run_drqv2 ;;
    all)   run_a2c; run_drqv2 ;;
    *) echo "Usage: $0 [a2c|drqv2|all]"; exit 1 ;;
esac

log "All done! Run:  python3 compare_results.py  to plot."
