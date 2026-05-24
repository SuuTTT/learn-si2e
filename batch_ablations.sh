#!/usr/bin/env bash
# batch_ablations.sh
# Mechanistic ablations to test what makes SI2E outperform VCSE.
#
# Two ablations × 3 seeds × DoorKey-8x8 @ 1M frames (~20 min/run on GPU).
# Plus: 2 extra seeds for KC-S3R2 SI2E to confirm s3=35.5% was an outlier.
#
# HYPOTHESIS BEING TESTED:
#   H3 (no_norm):    Remove relative batch-max normalization from adj_matrix.
#                    Use absolute similarity 1/(1+dist) instead.
#                    Prediction: variance INCREASES back toward VCSE if H3 is true.
#   H4 (no_cluster): Skip cluster-level bonus (reward_1 term).
#                    PartitionTree still runs (with relative norm), but only leaf-level.
#                    Prediction: variance INCREASES if H4 is true; stays low if H3 is sufficient.
#
# INTERPRETATION GUIDE:
#   si2e       ~100% ± 0      (ground truth — all seeds converge)
#   vcse       ~98%  ± 3      (ground truth — occasional sub-100%)
#   no_cluster ≈ vcse → cluster-level bonus IS the key mechanism (H4 confirmed)
#   no_cluster ≈ si2e → cluster-level bonus NOT necessary; relative norm alone explains it
#   no_norm    ≈ vcse → relative normalization IS the key mechanism (H3 confirmed)
#   no_norm    ≈ si2e → relative normalization NOT necessary; cluster bonus alone explains it
#
# Usage: nohup ./batch_ablations.sh > logs/ablations.log 2>&1 &

set -e

SEEDS=(1 2 3)
FRAMES_QUICK=1000000    # 1M frames — enough to see convergence on DK-8x8
FRAMES_KC=3000000       # 3M for KC-S3R2 extra seeds
ENV_DK="MiniGrid-DoorKey-8x8-v0"
ENV_KC="MiniGrid-KeyCorridorS3R2-v0"
SI2E_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/ablations"
RESULTS_KC="/workspace/learn-si2e/results/keycorridor"

mkdir -p "$RESULTS_BASE"

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "ablation,seed,success_rate_pct,mean_return,frames" > "$SUMMARY"
fi

run_ablation() {
    local ablation="$1" seed="$2"
    local model_name="abl-${ablation}-s${seed}"
    local out_dir="$RESULTS_BASE/${ablation}-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${SI2E_DIR}/storage/${model_name}/log.csv"
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 900000 )); then
            echo "[SKIP] ${model_name} already complete"
            grep "^${ablation},${seed}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "============================================================"
    echo "[RUN] DK-8x8  ablation=${ablation}  seed=${seed}"
    echo "============================================================"
    cd "$SI2E_DIR"
    python3 -m scripts.train \
        --algo a2c --env "$ENV_DK" --model "$model_name" \
        --frames "$FRAMES_QUICK" --use_batch \
        --save-interval 100 --log-interval 500 --seed "$seed" \
        --use_entropy_reward --use_value_condition --beta 0.005 \
        --ablation "$ablation" \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    local eval_out
    eval_out=$(python3 -m scripts.eval_success \
        --env "$ENV_DK" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr mr final_frames
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    mr=$(echo "$eval_out" | grep "^Mean return:" | awk '{print $3}')
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES_QUICK")
    echo "${ablation},${seed},${sr},${mr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: success_rate=${sr}%"
}

run_si2e_kc_extra() {
    local seed="$1"
    local model_name="kc-si2e-s${seed}"
    local out_dir="$RESULTS_KC/si2e-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${SI2E_DIR}/storage/${model_name}/log.csv"
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP] ${model_name} already complete"
            return
        fi
    fi

    echo "============================================================"
    echo "[RUN] KC-S3R2 SI2E extra seed=${seed}"
    echo "============================================================"
    cd "$SI2E_DIR"
    python3 -m scripts.train \
        --algo a2c --env "$ENV_KC" --model "$model_name" \
        --frames "$FRAMES_KC" --use_batch \
        --save-interval 100 --log-interval 500 --seed "$seed" \
        --use_entropy_reward --use_value_condition --beta 0.005 \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    local eval_out
    eval_out=$(python3 -m scripts.eval_success \
        --env "$ENV_KC" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr mr
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    mr=$(echo "$eval_out" | grep "^Mean return:" | awk '{print $3}')
    local final_frames
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES_KC")
    echo "si2e,${seed},${sr},${mr},${final_frames}" >> "$RESULTS_KC/summary.csv"
    echo "[DONE] ${model_name}: success_rate=${sr}%"
}

echo "============================================================"
echo "ABLATION: no_cluster (H4 — remove cluster-level bonus)"
echo "============================================================"
for seed in "${SEEDS[@]}"; do run_ablation "no_cluster" "$seed"; done

echo "============================================================"
echo "ABLATION: no_norm (H3 — remove relative batch-max normalization)"
echo "============================================================"
for seed in "${SEEDS[@]}"; do run_ablation "no_norm" "$seed"; done

echo "============================================================"
echo "KC-S3R2 SI2E extra seeds (s4, s5) to confirm s3=35.5% outlier"
echo "============================================================"
run_si2e_kc_extra 4
run_si2e_kc_extra 5

echo ""
echo "============================================================"
echo "ABLATION SUMMARY"
echo "============================================================"
cat "$SUMMARY"
python3 - <<'PYEOF'
import csv, sys, os, statistics
from collections import defaultdict

print("\n=== DK-8x8 Ablation Results (1M frames, 3 seeds each) ===")
print("Baseline reference (from results/a2c-multiseed/summary.csv):")
print("  VCSE:  97.8% mean, std=3.1  (5 seeds, 3M frames)")
print("  SI2E: 100.0% mean, std=0.0  (5 seeds, 3M frames)")
print("")
print("Ablations (1M frames — expect slightly lower absolute values):")

summary = "/workspace/learn-si2e/results/ablations/summary.csv"
if not os.path.exists(summary):
    sys.exit(0)
data = defaultdict(list)
with open(summary) as f:
    for row in csv.DictReader(f):
        data[row['ablation']].append(float(row['success_rate_pct']))

print("Ablation      | Seeds | Mean SR%  | Std   | Interpretation")
print("--------------|-------|-----------|-------|---------------")
for ablation in ['no_cluster', 'no_norm']:
    vals = data.get(ablation, [])
    if vals:
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if std > 2.5:
            interp = "HIGH variance → this component matters"
        else:
            interp = "low variance  → this component NOT key"
        print(f"{ablation:<13} | {len(vals):>5} | {mean:>9.2f} | {std:>5.2f} | {interp}")

print("")
print("=== KC-S3R2 SI2E Extended Seeds ===")
kc_summary = "/workspace/learn-si2e/results/keycorridor/summary.csv"
if os.path.exists(kc_summary):
    si2e_vals = []
    with open(kc_summary) as f:
        for row in csv.DictReader(f):
            if row['method'] == 'si2e':
                si2e_vals.append((int(row['seed']), float(row['success_rate_pct'])))
    si2e_vals.sort()
    for seed, sr in si2e_vals:
        print(f"  seed={seed}: {sr:.1f}%")
    vals = [sr for _, sr in si2e_vals]
    if len(vals) > 1:
        print(f"  Mean: {statistics.mean(vals):.1f}%  Std: {statistics.stdev(vals):.1f}")
PYEOF
