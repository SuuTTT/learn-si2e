#!/usr/bin/env bash
# batch_fastsi2e_dk3m.sh
# Run FastSI2E on DoorKey-8x8 at 3M frames (resumes from 1M checkpoints if present).
# Gives paper-comparable results vs original SI2E (3M frames, 5 seeds, 100%±0%).
#
# Run after batch_fast_si2e.sh completes DK-8x8 1M runs.
# Usage: nohup ./batch_fastsi2e_dk3m.sh > logs/fast_si2e_dk3m.log 2>&1 &

set -e

SEEDS=(1 2 3)
FRAMES=3000000
A2C_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/fast-si2e"

mkdir -p "$RESULTS_BASE" /workspace/learn-si2e/logs

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,env,seed,success_rate_pct,fps" > "$SUMMARY"
fi

FAST_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005 --fast_se"

run_and_eval() {
    local method="$1"
    local env="$2"
    local seed="$3"
    local algo_flags="$4"
    local frames="$5"
    local env_short="${env//MiniGrid-/}"; env_short="${env_short//-v0/}"

    local model_name="fastse-${method}-${env_short}-s${seed}"
    local out_dir="$RESULTS_BASE/${method}-${env_short}-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${A2C_DIR}/storage/${model_name}/log.csv"
    local status_pt="${A2C_DIR}/storage/${model_name}/status.pt"
    local threshold=$(( frames - 100000 ))

    if [[ -f "$log_csv" ]] && [[ -f "$status_pt" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= threshold )); then
            echo "[SKIP-TRAIN] ${model_name} already at ${last_f} frames"
            if grep -q "^${method},${env},${seed}," "$SUMMARY" 2>/dev/null; then
                echo "[SKIP-EVAL] ${model_name} already in summary"
                return
            fi
            echo "[EVAL-ONLY] Running eval on checkpoint..."
            local eval_out
            eval_out=$(cd "$A2C_DIR" && python3 -m scripts.eval_success \
                --env "$env" --model "$model_name" \
                --episodes 200 --argmax --seed 999 2>&1)
            echo "$eval_out"
            local sr; sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
            local fps; fps=$(grep "FPS" "${out_dir}/train.log" 2>/dev/null | awk -F'FPS ' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
            echo "${method},${env},${seed},${sr},${fps}" >> "$SUMMARY"
            echo "[DONE] ${model_name}: SR=${sr}%"
            return
        fi
    fi

    echo "========================================================"
    local resume_msg="from scratch"
    [[ -f "$status_pt" ]] && resume_msg="resuming from checkpoint"
    echo "[RUN] method=${method}  env=${env_short}  seed=${seed}  frames=${frames}  (${resume_msg})"
    echo "========================================================"
    cd "$A2C_DIR"
    python3 -m scripts.train \
        --env "$env" \
        --model "$model_name" \
        --frames "$frames" \
        --use_batch \
        --save-interval 0 \
        --log-interval 500 \
        --seed "$seed" \
        $algo_flags \
        2>&1 | tee "${out_dir}/train_3m.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log_3m.csv"

    local eval_out
    eval_out=$(cd "$A2C_DIR" && python3 -m scripts.eval_success \
        --env "$env" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr; sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    local fps; fps=$(grep "FPS" "${out_dir}/train_3m.log" | awk -F'FPS ' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
    echo "${method},${env},${seed},${sr},${fps}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: SR=${sr}%  FPS=${fps}"
}

run_pairs() {
    local method="$1" env="$2" flags="$3" frames="$4"
    shift 4
    local pids=()
    for seed in "$@"; do
        run_and_eval "$method" "$env" "$seed" "$flags" "$frames" &
        pids+=($!)
        if (( ${#pids[@]} >= 2 )); then
            wait "${pids[@]}"; pids=()
        fi
    done
    (( ${#pids[@]} > 0 )) && wait "${pids[@]}"
}

echo "=== DK-8x8: FastSI2E at 3M frames (paper-comparable) ==="
run_pairs "fast-si2e" "MiniGrid-DoorKey-8x8-v0" "$FAST_FLAGS" "$FRAMES" "${SEEDS[@]}"

echo ""
echo "=== FAST-SI2E DK-8x8 3M SUMMARY ==="
python3 - <<'PYEOF'
import csv, os, numpy as np
from collections import defaultdict
p = "/workspace/learn-si2e/results/fast-si2e/summary.csv"
data = defaultdict(lambda: defaultdict(list))
with open(p) as f:
    for row in csv.DictReader(f):
        try: data[row["env"]][row["method"]].append(float(row["success_rate_pct"]))
        except: pass
for env, methods in sorted(data.items()):
    print(f"\n{env}:")
    for method, vals in sorted(methods.items()):
        mean = np.mean(vals); std = np.std(vals)
        print(f"  {method:24s} N={len(vals):2d}  {mean:.1f}% ± {std:.1f}")
PYEOF
