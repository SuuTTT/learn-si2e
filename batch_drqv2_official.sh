#!/usr/bin/env bash
# =============================================================================
# batch_drqv2_official.sh  —  510K frames (= paper's "250K env steps")
#
# Runs: 6 tasks × (baseline + SE + VCSE) × seed=1
# Budget: num_train_frames=510000  (default in easy.yaml — matches paper)
# Parallel: safe to run alongside batch_drqv2.sh (DrQv2 uses ~900 MiB VRAM)
#
# Launch:  nohup bash batch_drqv2_official.sh > results/batch_official.log 2>&1 &
# Monitor: tail -f /workspace/learn-si2e/results/batch_official.log
# =============================================================================
set -euo pipefail

ROOT=/workspace/learn-si2e
SI2E_DIR=$ROOT/SI2E/SI2E_DrQv2
VCSE_DIR=$ROOT/base-vcse/VCSE_DrQv2
RESULTS=$ROOT/results/drqv2-official

mkdir -p "$RESULTS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# FPS estimates (measured on RTX 3070 Laptop, 510K frames)
fps_for() {
    case $1 in
        baseline) echo 135 ;;
        se|vcse)  echo 95  ;;
        *)        echo 100 ;;
    esac
}

# Print the run info table and return once step-0 eval is captured
print_info_table() {
    local task=$1 method=$2 seed=$3 dir=$4 pid=$5 logfile=$6

    local fps; fps=$(fps_for "$method")
    local frames=510000
    local est_secs=$(( frames / fps ))
    local est_hrs; est_hrs=$(awk "BEGIN{printf \"%.1f\", $est_secs/3600}")

    # Wait up to 90s for step-0 eval to appear
    local step0_er="pending"
    local t=0
    while [[ $t -lt 90 ]]; do
        local line; line=$(grep "eval.*F: 0 " "$logfile" 2>/dev/null | head -1 || true)
        if [[ -n "$line" ]]; then
            step0_er=$(echo "$line" | grep -oP 'R: [0-9.]+' | head -1 | cut -d' ' -f2)
            break
        fi
        sleep 3; (( t += 3 ))
    done

    printf "\n%-20s %s\n"  "Log"           "$logfile"
    printf "%-20s %s\n"    "Description"   "Reproduce SI2E paper — DrQv2+${method^^} / ${task} / 510K frames (full budget)"
    printf "%-20s %s\n"    "Hypothesis"    "SI2E > VCSE > SE > baseline (paper order); target ER ~ paper mean"
    printf "%-20s %s\n"    "PID"           "$pid"
    printf "%-20s %s\n"    "Est. duration" "~${est_hrs}h (${fps} FPS)"
    printf "%-20s %s\n"    "Step-0 ER"     "$step0_er"
    printf "\n"
}

run_si2e() {
    local task=$1 method=$2 seed=$3
    local agent_target agent_do_vcse tag dir logfile

    case $method in
        baseline) agent_target=drqv2.DrQV2Agent; agent_do_vcse=""; tag=baseline ;;
        se)       agent_target=si2e.SI2EAgent;   agent_do_vcse="agent.do_vcse=false"; tag=se ;;
        *)        log "Unknown method: $method"; return 1 ;;
    esac

    dir="$RESULTS/${task}_${tag}_seed${seed}"
    logfile="$dir/stdout.log"

    if [[ -f "$dir/eval.csv" ]] && tail -1 "$dir/eval.csv" 2>/dev/null | grep -q "^[0-9]"; then
        local last_f; last_f=$(tail -1 "$dir/eval.csv" | awk -F',' '{print $4}')
        if [[ "${last_f:-0}" -ge 500000 ]]; then
            log "SKIP (complete): $dir"
            return 0
        fi
    fi

    mkdir -p "$dir"
    log "LAUNCH: $task / $method / seed=$seed"

    cd "$SI2E_DIR"
    MUJOCO_GL=egl python3 train.py \
        agent._target_="$agent_target" \
        ${agent_do_vcse} \
        "task@_global_=${task}" \
        seed="$seed" num_train_frames=510000 device=cuda:0 \
        hydra.run.dir="$dir" > "$logfile" 2>&1 &
    local pid=$!

    print_info_table "$task" "$method" "$seed" "$dir" "$pid" "$logfile"

    if wait "$pid"; then
        local final_er; final_er=$(tail -1 "$dir/eval.csv" 2>/dev/null | awk -F',' '{printf "%.1f",$3}')
        log "DONE:  $task / $method / seed=$seed  ER=$final_er  →  $dir"
    else
        log "FAILED (exit=$?): $task / $method / seed=$seed"
    fi
}

run_vcse() {
    local task=$1 seed=$2
    local dir="$RESULTS/${task}_vcse_seed${seed}"
    local logfile="$dir/stdout.log"

    if [[ -f "$dir/eval.csv" ]] && tail -1 "$dir/eval.csv" 2>/dev/null | grep -q "^[0-9]"; then
        local last_f; last_f=$(tail -1 "$dir/eval.csv" | awk -F',' '{print $4}')
        if [[ "${last_f:-0}" -ge 500000 ]]; then
            log "SKIP (complete): $dir"
            return 0
        fi
    fi

    mkdir -p "$dir"
    log "LAUNCH: $task / VCSE / seed=$seed"

    cd "$VCSE_DIR"
    MUJOCO_GL=egl python3 train.py \
        agent.do_vcse=true \
        "task@_global_=${task}" \
        seed="$seed" num_train_frames=510000 device=cuda:0 \
        hydra.run.dir="$dir" > "$logfile" 2>&1 &
    local pid=$!

    print_info_table "$task" "vcse" "$seed" "$dir" "$pid" "$logfile"

    if wait "$pid"; then
        local final_er; final_er=$(tail -1 "$dir/eval.csv" 2>/dev/null | awk -F',' '{printf "%.1f",$3}')
        log "DONE:  $task / VCSE / seed=$seed  ER=$final_er  →  $dir"
    else
        log "FAILED (exit=$?): $task / VCSE / seed=$seed"
    fi
}

TASKS=(cartpole_swingup hopper_stand cheetah_run quadruped_walk pendulum_swingup cartpole_balance)

log "=== batch_drqv2_official: 510K frames × 6 tasks × (baseline+SE+VCSE) × seed=1 ==="
log "=== Results → $RESULTS ==="

for task in "${TASKS[@]}"; do
    run_si2e "$task" baseline 1
    run_si2e "$task" se      1
    run_vcse  "$task" 1
done

log "=== ALL DONE ==="
