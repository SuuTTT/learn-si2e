#!/usr/bin/env bash
# batch_fast_si2e.sh
# FastSI2E speed/performance validation.
# Confirms glass-jax clustering (fast_se) matches PartitionTree accuracy
# while being ~1.5x faster.
#
# Usage:
#   chmod +x batch_fast_si2e.sh
#   nohup ./batch_fast_si2e.sh > logs/fast_si2e.log 2>&1 &

set -e

SEEDS=(1 2 3)
FRAMES=3000000
A2C_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/fast-si2e"

mkdir -p "$RESULTS_BASE" /workspace/learn-si2e/logs

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,env,seed,success_rate_pct,frames" > "$SUMMARY"
fi

run_and_eval() {
    local method="$1"
    local env="$2"
    local seed="$3"
    local algo_flags="$4"
    local env_short="${env//MiniGrid-/}"; env_short="${env_short//-v0/}"

    local model_name="fastse-${method}-${env_short}-s${seed}"
    local out_dir="$RESULTS_BASE/${method}-${env_short}-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${A2C_DIR}/storage/${model_name}/log.csv"
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP] ${model_name} already at ${last_f} frames"
            grep "^${method},${env},${seed}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "========================================================"
    echo "[RUN] method=${method}  env=${env_short}  seed=${seed}"
    echo "========================================================"
    cd "$A2C_DIR"
    python3 -m scripts.train \
        --env "$env" \
        --model "$model_name" \
        --frames "$FRAMES" \
        --use_batch \
        --save-interval 0 \
        --log-interval 500 \
        --seed "$seed" \
        $algo_flags \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    local eval_out
    eval_out=$(python3 -m scripts.eval_success \
        --env "$env" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    local fps
    fps=$(grep "FPS" "${out_dir}/train.log" | awk -F'FPS ' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
    echo "${method},${env},${seed},${sr},${fps}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: SR=${sr}%  FPS=${fps}"
}

SI2E_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005"
FAST_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005 --fast_se"
PPO_FAST_FLAGS="--algo ppo --use_entropy_reward --use_value_condition --beta 0.005 --fast_se"

echo "=== DK-8x8: original SI2E vs FastSI2E (3 seeds each) ==="
for seed in "${SEEDS[@]}"; do
    run_and_eval "si2e" "MiniGrid-DoorKey-8x8-v0" "$seed" "$SI2E_FLAGS"
    run_and_eval "fast-si2e" "MiniGrid-DoorKey-8x8-v0" "$seed" "$FAST_FLAGS"
done

echo "=== KC-S3R2: FastSI2E (3 seeds) ==="
for seed in "${SEEDS[@]}"; do
    run_and_eval "fast-si2e" "MiniGrid-KeyCorridorS3R2-v0" "$seed" "$FAST_FLAGS"
    run_and_eval "ppo-fast-si2e" "MiniGrid-KeyCorridorS3R2-v0" "$seed" "$PPO_FAST_FLAGS"
done

echo ""
echo "=== FAST-SI2E SUMMARY ==="
python3 - <<'PYEOF'
import csv, os
from collections import defaultdict
p = "/workspace/learn-si2e/results/fast-si2e/summary.csv"
if not os.path.exists(p): exit()
data = defaultdict(lambda: defaultdict(list))
with open(p) as f:
    for row in csv.DictReader(f):
        try: data[row["env"]][row["method"]].append(float(row["success_rate_pct"]))
        except: pass
for env, methods in sorted(data.items()):
    print(f"\n{env}:")
    for method, vals in sorted(methods.items()):
        mean = sum(vals)/len(vals)
        std = (sum((v-mean)**2 for v in vals)/len(vals))**0.5
        print(f"  {method:24s} N={len(vals):2d}  {mean:.1f}% ± {std:.1f}")
PYEOF
