#!/usr/bin/env bash
# batch_doorkey16x16.sh
# Run baseline / SE / VCSE (kNN) / SI2E (H₂) on MiniGrid-DoorKey-16x16 × 3 seeds.
# Harder than DoorKey-8x8 — not in the SI2E paper, bonus experiment.
#
# Usage:
#   chmod +x batch_doorkey16x16.sh
#   nohup ./batch_doorkey16x16.sh > logs/doorkey16x16.log 2>&1 &

set -e

SEEDS=(1 2 3)
FRAMES=3000000
ENV="MiniGrid-DoorKey-16x16-v0"
SI2E_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
VCSE_DIR="/workspace/learn-si2e/base-vcse/VCSE_A2C/rl-starter-files/rl-starter-files"
VCSE_TORCH_AC="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac"
RESULTS_BASE="/workspace/learn-si2e/results/doorkey16x16"

mkdir -p "$RESULTS_BASE"

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,seed,success_rate_pct,mean_return,frames" > "$SUMMARY"
fi

# ── SI2E / baseline / SE methods (all from SI2E codebase) ────────────────────
run_si2e_method() {
    local method="$1"   # baseline | se | si2e
    local seed="$2"
    local extra_flags="$3"
    local model_name="dk16-${method}-s${seed}"
    local out_dir="$RESULTS_BASE/${method}-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${SI2E_DIR}/storage/${model_name}/log.csv"
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP] ${model_name} already at ${last_f} frames"
            grep "^${method},${seed}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "============================================================"
    echo "[RUN] DoorKey-16x16  method=${method}  seed=${seed}"
    echo "============================================================"
    cd "$SI2E_DIR"
    python3 -m scripts.train \
        --algo a2c \
        --env "$ENV" \
        --model "$model_name" \
        --frames "$FRAMES" \
        --use_batch \
        --save-interval 100 \
        --log-interval 500 \
        --seed "$seed" \
        $extra_flags \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    echo "[EVAL] 200-episode greedy evaluation..."
    local eval_out
    eval_out=$(python3 -m scripts.eval_success \
        --env "$ENV" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr mr final_frames
    sr=$(echo "$eval_out"  | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    mr=$(echo "$eval_out"  | grep "^Mean return:"  | awk '{print $3}')
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES")
    echo "${method},${seed},${sr},${mr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: success_rate=${sr}%"
}

# ── Original VCSE (kNN, Kim et al.) from base-vcse codebase ──────────────────
run_vcse_original() {
    local seed="$1"
    local model_name="dk16-vcse-s${seed}"
    local out_dir="$RESULTS_BASE/vcse-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${VCSE_DIR}/storage/${model_name}/log.csv"
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP] ${model_name} already at ${last_f} frames"
            grep "^vcse,${seed}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "============================================================"
    echo "[RUN] DoorKey-16x16  method=vcse(kNN)  seed=${seed}"
    echo "============================================================"
    cd "$VCSE_DIR"
    PYTHONPATH="${VCSE_TORCH_AC}:${PYTHONPATH}" \
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

    echo "[EVAL] 200-episode greedy evaluation..."
    local eval_out
    eval_out=$(PYTHONPATH="${VCSE_TORCH_AC}:${PYTHONPATH}" \
        python3 -m scripts.eval_success \
        --env "$ENV" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr mr final_frames
    sr=$(echo "$eval_out"  | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    mr=$(echo "$eval_out"  | grep "^Mean return:"  | awk '{print $3}')
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES")
    echo "vcse,${seed},${sr},${mr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: success_rate=${sr}%"
}

# ── Run all methods ───────────────────────────────────────────────────────────
echo "=== DoorKey-16x16: baseline ==="
for seed in "${SEEDS[@]}"; do
    run_si2e_method "baseline" "$seed" ""
done

echo "=== DoorKey-16x16: SE (kNN entropy) ==="
for seed in "${SEEDS[@]}"; do
    run_si2e_method "se" "$seed" "--use_entropy_reward --beta 0.005"
done

echo "=== DoorKey-16x16: VCSE (original kNN, Kim et al.) ==="
for seed in "${SEEDS[@]}"; do
    run_vcse_original "$seed"
done

echo "=== DoorKey-16x16: SI2E (PartitionTree H₂) ==="
for seed in "${SEEDS[@]}"; do
    run_si2e_method "si2e" "$seed" "--use_entropy_reward --use_value_condition --beta 0.005"
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "DoorKey-16x16 SUMMARY  ($SUMMARY)"
echo "============================================================"
cat "$SUMMARY"
echo ""
python3 - <<'PYEOF'
import csv, sys, os
from collections import defaultdict

path = "/workspace/learn-si2e/results/doorkey16x16/summary.csv"
if not os.path.exists(path): sys.exit(0)

data = defaultdict(list)
with open(path) as f:
    for row in csv.DictReader(f):
        try:
            data[row["method"]].append(float(row["success_rate_pct"]))
        except (ValueError, KeyError):
            pass

print(f"{'Method':<12} {'N':>3}  {'Mean SR':>8}  {'Std SR':>8}")
for method in ["baseline", "se", "vcse", "si2e"]:
    vals = data[method]
    if vals:
        mean = sum(vals)/len(vals)
        std = (sum((v-mean)**2 for v in vals)/len(vals))**0.5
        print(f"{method:<12} {len(vals):>3}  {mean:>7.1f}%  {std:>7.1f}%")
    else:
        print(f"{method:<12}   0  {'—':>8}  {'—':>8}")
PYEOF
