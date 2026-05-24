#!/usr/bin/env bash
# sweep_vcse_difficulty.sh
# Run VCSE-kNN (original, Kim et al.) on a ladder of tasks at 1M frames / 1 seed
# to find the ~50% win-rate "sweet spot" between DoorKey-8x8 (97%) and DK-16x16 (0%).
#
# Usage:
#   nohup ./sweep_vcse_difficulty.sh > logs/vcse_sweep.log 2>&1 &

set -e

VCSE_DIR="/workspace/learn-si2e/base-vcse/VCSE_A2C/rl-starter-files/rl-starter-files"
VCSE_TORCH_AC="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac"
RESULTS="/workspace/learn-si2e/results/vcse_difficulty_sweep"
FRAMES=1000000
SEED=1

mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.csv"
[[ ! -f "$SUMMARY" ]] && echo "task,seed,frames,success_rate_pct,mean_return" > "$SUMMARY"

run_vcse() {
    local task="$1"          # e.g. KeyCorridorS4R3
    local env="MiniGrid-${task}-v0"
    local model="sweep-vcse-${task}-s${SEED}"
    local out_dir="$RESULTS/${task}"
    mkdir -p "$out_dir"

    local log_csv="${VCSE_DIR}/storage/${model}/log.csv"
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 950000 )); then
            echo "[SKIP] ${model} already at ${last_f}"
            grep "^${task}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "============================================================"
    echo "[RUN] $task  seed=$SEED  frames=$FRAMES"
    echo "============================================================"
    cd "$VCSE_DIR"
    PYTHONPATH="${VCSE_TORCH_AC}:${PYTHONPATH}" \
    python3 -m scripts.train \
        --algo a2c --env "$env" \
        --model "$model" \
        --frames "$FRAMES" \
        --use_batch \
        --save-interval 100 --log-interval 500 \
        --seed "$SEED" \
        --use_entropy_reward --use_value_condition --beta 0.005 \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    echo "[EVAL] 200 episodes..."
    local eval_out
    eval_out=$(PYTHONPATH="${VCSE_TORCH_AC}:${PYTHONPATH}" \
        python3 -m scripts.eval_success \
        --env "$env" --model "$model" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr mr
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d= -f2)
    mr=$(echo "$eval_out" | grep "^Mean return:"  | awk '{print $3}')
    echo "${task},${SEED},${FRAMES},${sr},${mr}" >> "$SUMMARY"
    echo "[DONE] $task → success=${sr}%"
}

# ── Difficulty ladder (KeyCorridor: S=size, R=rooms; higher = harder) ──────
# Paper (VCSE, 1M frames): S3R1=86%, S3R2=?, S3R3=?, S4R3=?, S5R3=?, S6R3=?
run_vcse "KeyCorridorS3R2"
run_vcse "KeyCorridorS3R3"
run_vcse "KeyCorridorS4R3"
run_vcse "KeyCorridorS5R3"
run_vcse "KeyCorridorS6R3"
# Bonus: other hard tasks
run_vcse "BlockedUnlockPickup"
run_vcse "MultiRoom-N6"

echo ""
echo "============================================================"
echo "SWEEP COMPLETE"
echo "============================================================"
cat "$SUMMARY"
