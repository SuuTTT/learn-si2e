#!/usr/bin/env bash
# batch_ppo_si2e.sh
# SI2E-PPO experiments: test if PPO backbone improves over A2C backbone.
# Runs 5 seeds on DK-8x8 (validation) and 5 seeds on KC-S3R2 (target task).
#
# Expected:
#   DK-8x8: 100% ±0 (matches A2C-SI2E — validates port)
#   KC-S3R2: >75% mean (vs A2C-SI2E 67.5%) — the performance gain
#
# Usage:
#   chmod +x batch_ppo_si2e.sh
#   nohup ./batch_ppo_si2e.sh > logs/ppo_si2e.log 2>&1 &

set -e

SEEDS=(1 2 3 4 5)
FRAMES=3000000
A2C_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/ppo-si2e"

mkdir -p "$RESULTS_BASE" /workspace/learn-si2e/logs

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,env,seed,success_rate_pct,frames" > "$SUMMARY"
fi

run_and_eval() {
    local method="$1"    # ppo-si2e | ppo-baseline | a2c-si2e
    local env="$2"       # MiniGrid-DoorKey-8x8-v0 | MiniGrid-KeyCorridorS3R2-v0
    local seed="$3"
    local algo_flags="$4"
    local env_short="${env//MiniGrid-/}"
    env_short="${env_short//-v0/}"

    local model_name="ppo-exp-${method}-${env_short}-s${seed}"
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
        --save-interval 100 \
        --log-interval 500 \
        --seed "$seed" \
        $algo_flags \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    echo "[EVAL] 200-episode greedy evaluation..."
    local eval_out
    eval_out=$(python3 -m scripts.eval_success \
        --env "$env" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    local final_frames
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES")
    echo "${method},${env},${seed},${sr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: SR=${sr}%"
}

PPO_SI2E_FLAGS="--algo ppo --use_entropy_reward --use_value_condition --beta 0.005"
A2C_SI2E_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005"
PPO_BASE_FLAGS="--algo ppo"

# ── Phase 1: DK-8x8 validation (PPO-SI2E vs A2C-SI2E) ────────────────────
echo "=== DK-8x8: PPO-SI2E (5 seeds) ==="
for seed in "${SEEDS[@]}"; do
    run_and_eval "ppo-si2e" "MiniGrid-DoorKey-8x8-v0" "$seed" "$PPO_SI2E_FLAGS"
done

# ── Phase 2: KC-S3R2 performance test ────────────────────────────────────
echo "=== KC-S3R2: PPO-SI2E (5 seeds) ==="
for seed in "${SEEDS[@]}"; do
    run_and_eval "ppo-si2e" "MiniGrid-KeyCorridorS3R2-v0" "$seed" "$PPO_SI2E_FLAGS"
done

echo "=== KC-S3R2: A2C-SI2E (5 seeds, comparison) ==="
for seed in "${SEEDS[@]}"; do
    run_and_eval "a2c-si2e" "MiniGrid-KeyCorridorS3R2-v0" "$seed" "$A2C_SI2E_FLAGS"
done

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== SUMMARY ==="
cat "$SUMMARY"
python3 - <<'PYEOF'
import csv, os
from collections import defaultdict
p = "/workspace/learn-si2e/results/ppo-si2e/summary.csv"
if not os.path.exists(p): exit()
data = defaultdict(lambda: defaultdict(list))
with open(p) as f:
    for row in csv.DictReader(f):
        try: data[row["env"]][row["method"]].append(float(row["success_rate_pct"]))
        except: pass
for env, methods in data.items():
    print(f"\n{env}:")
    for method, vals in sorted(methods.items()):
        mean = sum(vals)/len(vals)
        std = (sum((v-mean)**2 for v in vals)/len(vals))**0.5
        print(f"  {method:16s} N={len(vals):2d}  {mean:.1f}% ± {std:.1f}")
PYEOF
