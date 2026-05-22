#!/usr/bin/env bash
# batch_a2c_multiseed.sh
# Run A2C baseline / SE / VCSE on MiniGrid-DoorKey-8x8 × N seeds.
# SI2E is skipped (97 min/seed; not needed for paper comparison).
#
# Usage:
#   chmod +x batch_a2c_multiseed.sh
#   ./batch_a2c_multiseed.sh
#
# Results:
#   training logs  → results/a2c-multiseed/<method>-s<seed>/train.log
#   CSV logs       → results/a2c-multiseed/<method>-s<seed>/log.csv
#   success rates  → results/a2c-multiseed/summary.csv
#
# Paper targets (10 seeds, DoorKey-8x8):
#   baseline: — (0%)    SE: 72.60 ± 20.32%    VCSE: 94.32 ± 11.09%    SI2E: 98.58 ± 3.11%
# NOTE: what was labeled 'vcse' in this script is actually SI2E (PartitionTree H₂).
#       Original VCSE (kNN, Kim et al.) runs are in batch_vcse_original.sh.

set -e

SEEDS=(1 2 3 4 5)      # adjust as desired; 5 seeds × 3 methods ≈ 2.5 h
FRAMES=3000000
ENV="MiniGrid-DoorKey-8x8-v0"
A2C_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/a2c-multiseed"

mkdir -p "$RESULTS_BASE"

# Write summary CSV header if missing
SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,seed,success_rate_pct,mean_return,frames" > "$SUMMARY"
fi

run_and_eval() {
    local method="$1"      # baseline | se | vcse
    local seed="$2"
    local extra_flags="$3"

    local model_name="multiseed-${method}-s${seed}"
    local out_dir="$RESULTS_BASE/${method}-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${A2C_DIR}/storage/${model_name}/log.csv"
    local storage_log="${A2C_DIR}/storage/${model_name}/train.log"

    # Skip if already at 3M frames
    if [[ -f "$log_csv" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP] ${model_name} already at ${last_f} frames"
            # still re-run eval in case summary row is missing
            grep "^${method},${seed}," "$SUMMARY" > /dev/null 2>&1 && return
        fi
    fi

    echo "============================================================"
    echo "[RUN] method=${method} seed=${seed}  model=${model_name}"
    echo "      flags: ${extra_flags}"
    echo "============================================================"

    cd "$A2C_DIR"
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

    # Copy CSV to results dir for convenience
    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    # Evaluate success rate (200 episodes, greedy policy)
    echo "[EVAL] Running 200-episode greedy evaluation..."
    local eval_out
    eval_out=$(python3 -m scripts.eval_success \
        --env "$ENV" \
        --model "$model_name" \
        --episodes 200 \
        --argmax \
        --seed 999 2>&1)
    echo "$eval_out"

    local sr
    sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    local mr
    mr=$(echo "$eval_out" | grep "^Mean return:" | awk '{print $3}')
    local final_frames
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 || echo "$FRAMES")

    echo "${method},${seed},${sr},${mr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: success_rate=${sr}%"
}

# ── Baseline ─────────────────────────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    run_and_eval "baseline" "$seed" ""
done

# ── SE (kNN entropy reward) ────────────────────────────────────────────────
# beta=0.005 matches the paper's run_sent.sh (default beta=1e-05 is ~500x too small)
for seed in "${SEEDS[@]}"; do
    run_and_eval "se" "$seed" "--use_entropy_reward --beta 0.005"
done

# ── SI2E (structural entropy H₂ via PartitionTree + value conditioning) ────
# Labeled 'vcse' in run_si2e.sh but is SI2E's proposed method, NOT Kim et al. VCSE.
# beta=0.005 matches the paper's run_si2e.sh
# NOTE: model names on disk remain 'multiseed-vcse-s*' for seeds already trained;
#       the method label in summary.csv is fixed to 'si2e' by the cleanup below.
for seed in "${SEEDS[@]}"; do
    run_and_eval "si2e" "$seed" "--use_entropy_reward --use_value_condition --beta 0.005"
done

# Fix any rows written with old 'vcse' label (e.g. s4/s5 started before rename)
sed -i 's/^vcse,/si2e,/' "$SUMMARY"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "MULTI-SEED SUMMARY  ($SUMMARY)"
echo "============================================================"
cat "$SUMMARY"

echo ""
echo "Per-method statistics:"
python3 - <<'PYEOF'
import csv, sys, os
from collections import defaultdict

summary_path = os.environ.get("RESULTS_BASE", "/workspace/learn-si2e/results/a2c-multiseed") + "/summary.csv"
if not os.path.exists(summary_path):
    print("summary.csv not found"); sys.exit(0)

data = defaultdict(list)
with open(summary_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            sr = float(row["success_rate_pct"])
            data[row["method"]].append(sr)
        except (ValueError, KeyError):
            pass

print(f"{'Method':<12} {'N':>3}  {'Mean SR':>8}  {'Std SR':>8}  {'Paper SR':>14}")
paper = {"baseline": "— (0%)", "se": "72.60 ± 20.32", "vcse": "94.32 ± 11.09", "si2e": "98.58 ± 3.11"}
for method in ["baseline", "se", "vcse", "si2e"]:
    vals = data[method]
    if vals:
        mean = sum(vals) / len(vals)
        std = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5
        print(f"{method:<12} {len(vals):>3}  {mean:>7.1f}%  {std:>7.1f}%  {paper[method]:>14}")
    else:
        print(f"{method:<12}   0  {'—':>8}  {'—':>8}  {paper[method]:>14}")
PYEOF
