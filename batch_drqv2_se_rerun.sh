#!/usr/bin/env bash
# =============================================================================
# batch_drqv2_se_rerun.sh  —  SE-only rerun with larger replay buffer
#
# Purpose: Fix SE failures (cheetah_run=0, hopper_stand=8, pendulum_swingup=89)
#          caused by 100K buffer being too small for KNN entropy estimation.
#
# Fix: num_workers=0 (single-process replay) + replay_buffer_size=130K
#   Memory: 130 eps × 63 MB = 8.2 GB data + 1.5 GB overhead + 3 GB OS ≈ 12.7 GB
#   Diversity: full 130 episodes sampled per batch (vs 50/worker × 2 before)
#
# Tasks: only the 3 SE-failed tasks (cheetah_run, hopper_stand, pendulum_swingup)
#   cartpole_swingup (870), quadruped_walk (312) already have good SE — skip
#   cartpole_balance (running in batch_drqv2_official.sh) — skip
#
# Launch:  nohup bash batch_drqv2_se_rerun.sh >> results/batch_se_rerun.log 2>&1 &
# Monitor: tail -f /workspace/learn-si2e/results/batch_se_rerun.log
# =============================================================================
set -euo pipefail

ROOT=/workspace/learn-si2e
SI2E_DIR=$ROOT/SI2E/SI2E_DrQv2
RESULTS=$ROOT/results/drqv2-se-rerun

mkdir -p "$RESULTS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

TASKS=(cheetah_run hopper_stand pendulum_swingup)

run_se() {
    local task=$1 seed=$2
    local dir="$RESULTS/${task}_se_seed${seed}"
    local logfile="$dir/stdout.log"

    if [[ -f "$dir/eval.csv" ]] && tail -1 "$dir/eval.csv" 2>/dev/null | grep -q "^[0-9]"; then
        local last_f; last_f=$(tail -1 "$dir/eval.csv" | awk -F',' '{print $4}')
        if [[ "${last_f:-0}" -ge 500000 ]]; then
            log "SKIP (complete): $dir"
            return 0
        fi
    fi

    mkdir -p "$dir"
    log "LAUNCH: $task / SE / seed=$seed"

    cd "$SI2E_DIR"
    MUJOCO_GL=egl python3 train.py \
        agent._target_=si2e.SI2EAgent \
        agent.do_vcse=false \
        "task@_global_=${task}" \
        seed="$seed" num_train_frames=510000 device=cuda:0 \
        replay_buffer_size=130000 replay_buffer_num_workers=0 \
        hydra.run.dir="$dir" > "$logfile" 2>&1 &
    local pid=$!

    log "  PID=$pid  log=$logfile"

    if wait "$pid"; then
        local final_er; final_er=$(tail -1 "$dir/eval.csv" 2>/dev/null | awk -F',' '{printf "%.1f",$3}')
        log "DONE:  $task / SE / seed=$seed  ER=$final_er  →  $dir"
    else
        log "FAILED (exit=$?): $task / SE / seed=$seed"
    fi
}

log "=== batch_drqv2_se_rerun: 3 SE-failed tasks × seed=1, buf=130K, workers=0 ==="
log "=== Results → $RESULTS ==="

for task in "${TASKS[@]}"; do
    run_se "$task" 1
done

log "=== ALL DONE ==="
