#!/usr/bin/env bash
# batch_phase3.sh
# Phase 3 experiments — fills remaining Table 1 gaps and extends N for KC-S3R2.
#
# (A) KC-S3R2  Leiden    s4,s5     — extend N=3 → N=5
# (B) KC-S3R2  Infomap   s4,s5     — extend N=3 → N=5
# (C) RBD-6x6  Leiden    s1-3      — fills Table 1 "---" cell
# (D) RBD-6x6  Infomap   s1-3      — fills Table 1 "---" cell
# (E) RBD-6x6  Infomap+adaptive-β  s1-3  — tests if community detection fixes
#                                           the adaptive-β incompatibility found in phase2
#
# Results appended to results/clustering-methods/summary.csv
# (same file as phase1 clustering; analyze_results.py picks it up automatically)
#
# Runs pairs of seeds concurrently (2 at a time, ~15GB RAM limit).
# Model names follow the cm-{method}-{env_short}-s{seed} convention from
# batch_clustering_methods.sh so skip logic works across sessions.
#
# Usage:
#   chmod +x batch_phase3.sh
#   nohup ./batch_phase3.sh > logs/phase3.log 2>&1 &

set -e

A2C_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/clustering-methods"

mkdir -p "$RESULTS_BASE" /workspace/learn-si2e/logs

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,env,seed,success_rate_pct,fps" > "$SUMMARY"
fi

FRAMES=3000000
BASE_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005 --fast_se"

# ──────────────────────────────────────────────────────────────────────────────
# run_and_eval method cluster env seed frames [extra_flags]
# ──────────────────────────────────────────────────────────────────────────────
run_and_eval() {
    local method="$1"
    local cluster="$2"
    local env="$3"
    local seed="$4"
    local frames="$5"
    local extra_flags="${6:-}"
    local env_short="${env//MiniGrid-/}"; env_short="${env_short//-v0/}"

    local model_name="cm-${method}-${env_short}-s${seed}"
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
    echo "[RUN] method=${method}  cluster=${cluster}  env=${env_short}  seed=${seed}"
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
        --cluster_method "$cluster" \
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
    echo "${method},${env},${seed},${sr},${fps}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: SR=${sr}%  FPS=${fps}"
}

# run_pairs method cluster env frames [extra_flags] seed [seed ...]
run_pairs() {
    local method="$1" cluster="$2" env="$3" frames="$4" extra_flags="$5"
    shift 5
    local pids=()
    for seed in "$@"; do
        run_and_eval "$method" "$cluster" "$env" "$seed" "$frames" "$extra_flags" &
        pids+=($!)
        if (( ${#pids[@]} >= 2 )); then
            wait "${pids[@]}"; pids=()
        fi
    done
    if (( ${#pids[@]} > 0 )); then wait "${pids[@]}"; fi
}

# ──────────────────────────────────────────────────────────────────────────────
# (A) KC-S3R2 Leiden s4,s5  — extend from N=3 to N=5
# ──────────────────────────────────────────────────────────────────────────────
echo "=== (A) KC-S3R2 Leiden s4,s5 ==="
run_pairs "leiden-si2e" "leiden" "MiniGrid-KeyCorridorS3R2-v0" "$FRAMES" "" 4 5

# ──────────────────────────────────────────────────────────────────────────────
# (B) KC-S3R2 Infomap s4,s5  — extend from N=3 to N=5
# ──────────────────────────────────────────────────────────────────────────────
echo "=== (B) KC-S3R2 Infomap s4,s5 ==="
run_pairs "infomap-si2e" "infomap" "MiniGrid-KeyCorridorS3R2-v0" "$FRAMES" "" 4 5

# ──────────────────────────────────────────────────────────────────────────────
# (C) RBD-6x6 Leiden s1-3  — fills Table 1 gap
# ──────────────────────────────────────────────────────────────────────────────
echo "=== (C) RBD-6x6 Leiden s1-3 ==="
run_pairs "leiden-si2e" "leiden" "MiniGrid-RedBlueDoors-6x6-v0" "$FRAMES" "" 1 2 3

# ──────────────────────────────────────────────────────────────────────────────
# (D) RBD-6x6 Infomap s1-3  — fills Table 1 gap
# ──────────────────────────────────────────────────────────────────────────────
echo "=== (D) RBD-6x6 Infomap s1-3 ==="
run_pairs "infomap-si2e" "infomap" "MiniGrid-RedBlueDoors-6x6-v0" "$FRAMES" "" 1 2 3

# ──────────────────────────────────────────────────────────────────────────────
# (E) RBD-6x6 Infomap+adaptive-β s1-3  — does community detection fix the
#     adaptive-β incompatibility seen in phase2 with k-means?
# ──────────────────────────────────────────────────────────────────────────────
echo "=== (E) RBD-6x6 Infomap+adaptive-β s1-3 ==="
run_pairs "infomap-si2e-adaptive" "infomap" "MiniGrid-RedBlueDoors-6x6-v0" "$FRAMES" "--beta_adaptive" 1 2 3

# ──────────────────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== PHASE 3 COMPLETE — FULL CLUSTERING SUMMARY ==="
python3 - <<'PYEOF'
import csv, os, numpy as np
from collections import defaultdict

p = "/workspace/learn-si2e/results/clustering-methods/summary.csv"
data = defaultdict(lambda: defaultdict(list))
with open(p) as f:
    for row in csv.DictReader(f):
        try:
            env = row['env'].replace('MiniGrid-', '').replace('-v0', '')
            data[env][row['method']].append(float(row['success_rate_pct']))
        except: pass

# Also load fast-si2e k-means as reference
kc_ref = "/workspace/learn-si2e/results/fast-si2e/summary.csv"
if os.path.exists(kc_ref):
    with open(kc_ref) as f:
        for row in csv.DictReader(f):
            if row.get('method', '').startswith('fastse-fast-si2e'):
                env = row['env'].replace('MiniGrid-', '').replace('-v0', '')
                data[env]['fast-si2e (k-means ref)'].append(float(row['success_rate_pct']))

envs = ['DoorKey-8x8', 'KeyCorridorS3R2', 'RedBlueDoors-6x6']
for env in envs:
    if not data[env]:
        continue
    print(f"\n{env}:")
    for m, vals in sorted(data[env].items()):
        print(f"  {m:38s} N={len(vals):2d}  {np.mean(vals):.1f}% ± {np.std(vals):.1f}")
PYEOF

echo ""
echo "=== Phase 3 complete ==="
