#!/usr/bin/env bash
# batch_vcse_original.sh
# Run original VCSE (Kim et al. 2023, kNN value-conditional entropy) on
# MiniGrid-DoorKey-8x8 × N seeds.
#
# This is the ORIGINAL VCSE implementation from:
#   "Accelerating RL with Value-Conditional State Entropy Exploration"
#   Kim et al., NeurIPS 2023 — arXiv:2305.19476
#   code: base-vcse/VCSE_A2C/
#
# Unlike SI2E's "VCSE" label, this uses compute_value_condition_state_entropy
# (pure kNN + digamma estimator, GPU tensors) — NO PartitionTree / H₂.
#
# Paper targets (DoorKey-8x8, 10 seeds):
#   A2C + VCSE: 94.32 ± 11.09%
#
# Usage:
#   chmod +x batch_vcse_original.sh
#   ./batch_vcse_original.sh        # foreground
#   nohup ./batch_vcse_original.sh > logs/vcse_original.log 2>&1 &   # background
#
# Results appended to: results/a2c-multiseed/summary.csv  (method = "vcse")

set -e

SEEDS=(1 2 3 4 5)
FRAMES=3000000
ENV="MiniGrid-DoorKey-8x8-v0"
A2C_DIR="/workspace/learn-si2e/base-vcse/VCSE_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/a2c-multiseed"

mkdir -p "$RESULTS_BASE"

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,seed,success_rate_pct,mean_return,frames" > "$SUMMARY"
fi

run_vcse() {
    local seed="$1"
    local model_name="multiseed-vcse-orig-s${seed}"
    local out_dir="$RESULTS_BASE/vcse-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${A2C_DIR}/storage/${model_name}/log.csv"

    # Skip if already done
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP] ${model_name} already at ${last_f} frames"
            grep "^vcse,${seed}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "============================================================"
    echo "[RUN] original VCSE  seed=${seed}  model=${model_name}"
    echo "============================================================"

    cd "$A2C_DIR"
    # Prepend base-vcse's torch-ac so it doesn't import SI2E's PartitionTree-dependent version
    PYTHONPATH="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac:${PYTHONPATH}" \
    python3 -m scripts.train \
        --algo a2c \
        --env "$ENV" \
        --model "$model_name" \
        --frames "$FRAMES" \
        --use_batch \
        --save-interval 100 \
        --log-interval 500 \
        --seed "$seed" \
        --use_entropy_reward --use_value_condition --beta 0.005 \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    echo "[EVAL] Running 200-episode greedy evaluation..."
    local eval_out
    eval_out=$(PYTHONPATH="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac:${PYTHONPATH}" \
        python3 -m scripts.eval_success \
        --env "$ENV" \
        --model "$model_name" \
        --episodes 200 \
        --argmax \
        --seed 999 2>&1)
    echo "$eval_out"

    local sr mr final_frames
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    mr=$(echo "$eval_out" | grep "^Mean return:" | awk '{print $3}')
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES")

    echo "vcse,${seed},${sr},${mr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: success_rate=${sr}%"
}

for seed in "${SEEDS[@]}"; do
    run_vcse "$seed"
done

echo ""
echo "============================================================"
echo "VCSE (original kNN) DONE — summary:"
echo "============================================================"
python3 - <<'PYEOF'
import csv, sys, os
from collections import defaultdict

path = "/workspace/learn-si2e/results/a2c-multiseed/summary.csv"
if not os.path.exists(path): sys.exit(0)

data = defaultdict(list)
with open(path) as f:
    for row in csv.DictReader(f):
        try:
            data[row["method"]].append(float(row["success_rate_pct"]))
        except (ValueError, KeyError):
            pass

paper = {"baseline": "— (0%)", "se": "72.60 ± 20.32", "vcse": "94.32 ± 11.09", "si2e": "98.58 ± 3.11"}
print(f"{'Method':<12} {'N':>3}  {'Mean SR':>8}  {'Std SR':>8}  {'Paper target':>15}")
for method in ["baseline", "se", "vcse", "si2e"]:
    vals = data[method]
    if vals:
        mean = sum(vals)/len(vals)
        std = (sum((v-mean)**2 for v in vals)/len(vals))**0.5
        print(f"{method:<12} {len(vals):>3}  {mean:>7.1f}%  {std:>7.1f}%  {paper[method]:>15}")
    else:
        print(f"{method:<12}   0  {'—':>8}  {'—':>8}  {paper[method]:>15}")
PYEOF
