#!/usr/bin/env bash
# batch_phase2.sh
# Phase 2 overnight experiments:
#   (A) KC-S3R2 fast-si2e seeds 4,5  — round out 5-seed comparison
#   (B) RedBlueDoors-6x6 fast-si2e s1-3 — first test of FastSI2E on RBD
#   (C) fast-si2e + adaptive-β on KC-S3R2 s1-5 — combines speed & accuracy gains
#   (D) fast-si2e + adaptive-β on RBD s1-5
#
# Results written to results/phase2/summary.csv
# Runs pairs of seeds concurrently (2 at a time) to stay within 15GB RAM.
#
# Usage:
#   nohup ./batch_phase2.sh > logs/phase2.log 2>&1 &

set -e

A2C_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/phase2"
FAST_SI2E_BASE="/workspace/learn-si2e/results/fast-si2e"

mkdir -p "$RESULTS_BASE" /workspace/learn-si2e/logs

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,env,seed,success_rate_pct,fps" > "$SUMMARY"
fi

FRAMES_KC=3000000
FRAMES_RBD=3000000

BASE_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005 --fast_se"

run_and_eval() {
    local method="$1"
    local env="$2"
    local seed="$3"
    local frames="$4"
    local extra_flags="${5:-}"
    local summary_target="${6:-$SUMMARY}"
    local env_short="${env//MiniGrid-/}"; env_short="${env_short//-v0/}"

    local model_name="${method}-${env_short}-s${seed}"
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
            if grep -q "^${method},${env},${seed}," "$summary_target" 2>/dev/null; then
                echo "[SKIP-EVAL] ${model_name} already in summary"
                return
            fi
            local eval_out
            eval_out=$(cd "$A2C_DIR" && python3 -m scripts.eval_success \
                --env "$env" --model "$model_name" \
                --episodes 200 --argmax --seed 999 2>&1)
            echo "$eval_out"
            local sr; sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
            local fps; fps=$(grep "FPS" "${out_dir}/train.log" 2>/dev/null | awk -F'FPS ' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
            echo "${method},${env},${seed},${sr},${fps}" >> "$summary_target"
            echo "[DONE] ${model_name}: SR=${sr}%"
            return
        fi
    fi

    echo "========================================================"
    echo "[RUN] method=${method}  env=${env_short}  seed=${seed}  frames=${frames}"
    echo "========================================================"
    cd "$A2C_DIR"
    # shellcheck disable=SC2086
    python3 -m scripts.train \
        --env "$env" \
        --model "$model_name" \
        --frames "$frames" \
        --use_batch \
        --save-interval 0 \
        --log-interval 500 \
        --seed "$seed" \
        $BASE_FLAGS \
        $extra_flags \
        2>&1 | tee "${out_dir}/train.log"

    [[ -f "$log_csv" ]] && cp "$log_csv" "${out_dir}/log.csv"

    local eval_out
    eval_out=$(cd "$A2C_DIR" && python3 -m scripts.eval_success \
        --env "$env" --model "$model_name" \
        --episodes 200 --argmax --seed 999 2>&1)
    echo "$eval_out"

    local sr; sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
    local fps; fps=$(grep "FPS" "${out_dir}/train.log" | awk -F'FPS ' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
    echo "${method},${env},${seed},${sr},${fps}" >> "$summary_target"
    echo "[DONE] ${model_name}: SR=${sr}%  FPS=${fps}"
}

run_pairs() {
    local method="$1" env="$2" frames="$3" extra_flags="$4" summary_target="$5"
    shift 5
    local pids=()
    for seed in "$@"; do
        run_and_eval "$method" "$env" "$seed" "$frames" "$extra_flags" "$summary_target" &
        pids+=($!)
        if (( ${#pids[@]} >= 2 )); then
            wait "${pids[@]}"; pids=()
        fi
    done
    if (( ${#pids[@]} > 0 )); then wait "${pids[@]}"; fi
}

# ──────────────────────────────────────────────────────────────────────────
# (A) KC-S3R2 fast-si2e extra seeds — write into fast-si2e summary so
#     analyze_results.py picks them up with the existing s1-3 results
# ──────────────────────────────────────────────────────────────────────────
echo "=== (A) KC-S3R2 fast-si2e seeds 4,5 ==="
FAST_SUMMARY="$FAST_SI2E_BASE/summary.csv"
run_pairs "fastse-fast-si2e" "MiniGrid-KeyCorridorS3R2-v0" \
    "$FRAMES_KC" "" "$FAST_SUMMARY" 4 5

# ──────────────────────────────────────────────────────────────────────────
# (B) RedBlueDoors-6x6 fast-si2e s1-3 — first test on RBD
# ──────────────────────────────────────────────────────────────────────────
echo "=== (B) RedBlueDoors fast-si2e s1-3 ==="
run_pairs "fastse-rbd-fast-si2e" "MiniGrid-RedBlueDoors-6x6-v0" \
    "$FRAMES_RBD" "" "$SUMMARY" 1 2 3

# ──────────────────────────────────────────────────────────────────────────
# (C) KC-S3R2 fast-si2e + adaptive-β s1-5
# ──────────────────────────────────────────────────────────────────────────
echo "=== (C) KC-S3R2 fast-si2e + adaptive-β s1-5 ==="
run_pairs "fastse-adapt-kc" "MiniGrid-KeyCorridorS3R2-v0" \
    "$FRAMES_KC" "--beta_adaptive" "$SUMMARY" 1 2 3 4 5

# ──────────────────────────────────────────────────────────────────────────
# (D) RedBlueDoors fast-si2e + adaptive-β s1-5
# ──────────────────────────────────────────────────────────────────────────
echo "=== (D) RedBlueDoors fast-si2e + adaptive-β s1-5 ==="
run_pairs "fastse-adapt-rbd" "MiniGrid-RedBlueDoors-6x6-v0" \
    "$FRAMES_RBD" "--beta_adaptive" "$SUMMARY" 1 2 3 4 5

# ──────────────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "=== PHASE 2 SUMMARY ==="
python3 - <<'PYEOF'
import csv, os, numpy as np
from collections import defaultdict

files = {
    'fast-si2e':  '/workspace/learn-si2e/results/fast-si2e/summary.csv',
    'adaptive-β': '/workspace/learn-si2e/results/adaptive-beta/summary.csv',
    'phase2':     '/workspace/learn-si2e/results/phase2/summary.csv',
    'clustering': '/workspace/learn-si2e/results/clustering-methods/summary.csv',
}

data = defaultdict(lambda: defaultdict(list))
for src, path in files.items():
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                env = row['env'].replace('MiniGrid-','').replace('-v0','')
                m = row['method']
                data[env][m].append(float(row['success_rate_pct']))
            except: pass

envs = ['DoorKey-8x8', 'KeyCorridorS3R2', 'RedBlueDoors-6x6']
for env in envs:
    if not data[env]:
        continue
    print(f"\n{env}:")
    for m, vals in sorted(data[env].items()):
        print(f"  {m:38s} N={len(vals):2d}  {np.mean(vals):.1f}% ± {np.std(vals):.1f}")
PYEOF

echo ""
echo "=== Phase 2 complete ==="
