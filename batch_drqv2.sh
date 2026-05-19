#!/usr/bin/env bash
# =============================================================================
# batch_drqv2.sh  —  Fill SI2E Table 2 local results (skip SI2E, skip MADE)
#
# Runs: 5 remaining tasks × (baseline + SE + VCSE) × seed=1
#       + cartpole_swingup × (baseline + SE) × seeds 2-3
#
# Estimated total: ~13 h  (sequential, 1 GPU)
# Skips: DrQv2+SI2E (14 FPS, ~5 h per run)
#
# Launch: nohup bash batch_drqv2.sh > results/batch.log 2>&1 &
# Monitor: tail -f /workspace/learn-si2e/results/batch.log
# =============================================================================
set -euo pipefail

ROOT=/workspace/learn-si2e
SI2E_DIR=$ROOT/SI2E/SI2E_DrQv2
VCSE_DIR=$ROOT/base-vcse/VCSE_DrQv2
RESULTS=$ROOT/results/drqv2-full

mkdir -p "$RESULTS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_si2e() {
    local task=$1 method=$2 seed=$3
    local agent_target agent_do_vcse tag dir

    case $method in
        baseline) agent_target=drqv2.DrQV2Agent; agent_do_vcse=""; tag=baseline ;;
        se)       agent_target=si2e.SI2EAgent;   agent_do_vcse="agent.do_vcse=false"; tag=se ;;
        *)        log "Unknown method: $method"; return 1 ;;
    esac

    dir="$RESULTS/${task}_${tag}_seed${seed}"
    if [[ -f "$dir/eval.csv" ]]; then
        log "SKIP (exists): $dir"
        return 0
    fi

    mkdir -p "$dir"
    log "START: $task / $method / seed=$seed  →  $dir"
    cd "$SI2E_DIR"
    MUJOCO_GL=egl python3 train.py \
        agent._target_="$agent_target" \
        ${agent_do_vcse} \
        "task@_global_=${task}" \
        seed="$seed" num_train_frames=250000 device=cuda:0 \
        hydra.run.dir="$dir" 2>&1 | tee -a "$dir/stdout.log" | tail -1
    log "DONE:  $task / $method / seed=$seed  →  final eval: $(tail -1 $dir/eval.csv)"
}

run_vcse() {
    local task=$1 seed=$2
    local dir="$RESULTS/${task}_vcse_seed${seed}"

    if [[ -f "$dir/eval.csv" ]]; then
        log "SKIP (exists): $dir"
        return 0
    fi

    mkdir -p "$dir"
    log "START: $task / VCSE / seed=$seed  →  $dir"
    cd "$VCSE_DIR"
    MUJOCO_GL=egl python3 train.py \
        agent.do_vcse=true \
        "task@_global_=${task}" \
        seed="$seed" num_train_frames=250000 device=cuda:0 \
        hydra.run.dir="$dir" 2>&1 | tee -a "$dir/stdout.log" | tail -1
    log "DONE:  $task / VCSE / seed=$seed  →  final eval: $(tail -1 $dir/eval.csv)"
}

# ---------------------------------------------------------------------------
# Wait for SI2E nohup (PID 55737) to finish first
# ---------------------------------------------------------------------------
if ps -p 55737 > /dev/null 2>&1; then
    log "Waiting for DrQv2+SI2E (PID 55737) to finish..."
    while ps -p 55737 > /dev/null 2>&1; do sleep 60; done
    log "DrQv2+SI2E finished. Starting batch."
else
    log "PID 55737 already done. Starting immediately."
fi

# ---------------------------------------------------------------------------
# Phase 1: 5 remaining tasks × baseline + SE, seed=1
# (cartpole_swingup seed=1 baseline+SE already done — skip via exists check)
# ---------------------------------------------------------------------------
TASKS=(hopper_stand cheetah_run quadruped_walk pendulum_swingup cartpole_balance)

log "=== Phase 1: 5 tasks × baseline+SE × seed=1 ==="
for task in "${TASKS[@]}"; do
    run_si2e "$task" baseline 1
    run_si2e "$task" se      1
done

# ---------------------------------------------------------------------------
# Phase 2: cartpole_swingup × seeds 2-3 × baseline+SE
# ---------------------------------------------------------------------------
log "=== Phase 2: cartpole_swingup × seeds 2-3 ==="
for seed in 2 3; do
    run_si2e cartpole_swingup baseline "$seed"
    run_si2e cartpole_swingup se       "$seed"
done

# ---------------------------------------------------------------------------
# Phase 3: VCSE for all 6 tasks × seed=1
# ---------------------------------------------------------------------------
log "=== Phase 3: VCSE × all 6 tasks × seed=1 ==="
ALL_TASKS=(hopper_stand cheetah_run quadruped_walk pendulum_swingup cartpole_balance cartpole_swingup)
for task in "${ALL_TASKS[@]}"; do
    run_vcse "$task" 1
done

log "=== ALL DONE ==="
