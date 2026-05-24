#!/usr/bin/env bash
# batch_redbluedoors.sh
# Run baseline / SE / VCSE (kNN) / SI2E (H₂) on MiniGrid-RedBlueDoors-6x6-v0 × 3 seeds.
# Paper reports: VCSE=79.82%, SI2E=85.80% (baseline likely ~0% — not reported).
# Grid: 12×6, max_steps=720, 3M frames.
#
# Usage: nohup ./batch_redbluedoors.sh > logs/redbluedoors.log 2>&1 &

set -e

SEEDS=(1 2 3)
FRAMES=3000000
ENV="MiniGrid-RedBlueDoors-6x6-v0"
SI2E_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
VCSE_DIR="/workspace/learn-si2e/base-vcse/VCSE_A2C/rl-starter-files/rl-starter-files"
VCSE_TORCH_AC="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac"
RESULTS_BASE="/workspace/learn-si2e/results/redbluedoors"

mkdir -p "$RESULTS_BASE"

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,seed,success_rate_pct,mean_return,frames" > "$SUMMARY"
fi

run_si2e_method() {
    local method="$1" seed="$2" extra_flags="$3"
    local model_name="rb6-${method}-s${seed}"
    local out_dir="$RESULTS_BASE/${method}-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${SI2E_DIR}/storage/${model_name}/log.csv"
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP] ${model_name} already complete"
            grep "^${method},${seed}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "============================================================"
    echo "[RUN] RedBlueDoors-6x6  method=${method}  seed=${seed}"
    echo "============================================================"
    cd "$SI2E_DIR"
    python3 -m scripts.train \
        --algo a2c --env "$ENV" --model "$model_name" \
        --frames "$FRAMES" --use_batch \
        --save-interval 100 --log-interval 500 --seed "$seed" \
        $extra_flags \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    local eval_out
    eval_out=$(python3 -m scripts.eval_success \
        --env "$ENV" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr mr final_frames
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    mr=$(echo "$eval_out" | grep "^Mean return:" | awk '{print $3}')
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES")
    echo "${method},${seed},${sr},${mr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: success_rate=${sr}%"
}

run_vcse_original() {
    local seed="$1"
    local model_name="rb6-vcse-s${seed}"
    local out_dir="$RESULTS_BASE/vcse-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${VCSE_DIR}/storage/${model_name}/log.csv"
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP] ${model_name} already complete"
            grep "^vcse,${seed}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "============================================================"
    echo "[RUN] RedBlueDoors-6x6  method=vcse(kNN)  seed=${seed}"
    echo "============================================================"
    cd "$VCSE_DIR"
    PYTHONPATH="${VCSE_TORCH_AC}:${PYTHONPATH}" \
    python3 -m scripts.train \
        --algo a2c --env "$ENV" --model "$model_name" \
        --frames "$FRAMES" --use_batch \
        --save-interval 100 --log-interval 500 --seed "$seed" \
        --use_entropy_reward --use_value_condition --beta 0.005 \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    local eval_out
    eval_out=$(PYTHONPATH="${VCSE_TORCH_AC}:${PYTHONPATH}" \
        python3 -m scripts.eval_success \
        --env "$ENV" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr mr final_frames
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    mr=$(echo "$eval_out" | grep "^Mean return:" | awk '{print $3}')
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES")
    echo "vcse,${seed},${sr},${mr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: success_rate=${sr}%"
}

echo "=== RedBlueDoors-6x6: baseline ==="
for seed in "${SEEDS[@]}"; do run_si2e_method "baseline" "$seed" ""; done

echo "=== RedBlueDoors-6x6: SE ==="
for seed in "${SEEDS[@]}"; do run_si2e_method "se" "$seed" "--use_entropy_reward --beta 0.005"; done

echo "=== RedBlueDoors-6x6: VCSE (kNN) ==="
for seed in "${SEEDS[@]}"; do run_vcse_original "$seed"; done

echo "=== RedBlueDoors-6x6: SI2E (H₂) ==="
for seed in "${SEEDS[@]}"; do run_si2e_method "si2e" "$seed" "--use_entropy_reward --use_value_condition --beta 0.005"; done

echo ""
echo "============================================================"
echo "RedBlueDoors-6x6 SUMMARY"
echo "============================================================"
cat "$SUMMARY"
python3 - <<'PYEOF'
import csv, sys, os, statistics
from collections import defaultdict
summary = "/workspace/learn-si2e/results/redbluedoors/summary.csv"
if not os.path.exists(summary): sys.exit(0)
data = defaultdict(list)
with open(summary) as f:
    for row in csv.DictReader(f):
        data[row['method']].append(float(row['success_rate_pct']))
print("Method       | Seeds | Mean SR%  | Std")
print("-------------|-------|-----------|-----")
for method in ['baseline', 'se', 'vcse', 'si2e']:
    vals = data.get(method, [])
    if vals:
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{method:<12} | {len(vals):>5} | {mean:>9.2f} | {std:.2f}")
PYEOF
