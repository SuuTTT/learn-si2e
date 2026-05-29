#!/usr/bin/env bash
# batch_clustering_methods.sh
# Compare k-means vs Leiden vs Infomap as the graph clustering step in FastSI2E.
#
# All methods use --fast_se; differ only in --cluster_method.
# Runs DK-8x8 (3M, 3 seeds) and KC-S3R2 (3M, 3 seeds) for each method.
# RedBlueDoors omitted for speed; can be added with run_pairs calls below.
#
# Usage:
#   chmod +x batch_clustering_methods.sh
#   nohup ./batch_clustering_methods.sh > logs/clustering_methods.log 2>&1 &

set -e

SEEDS=(1 2 3)
FRAMES=3000000
A2C_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/clustering-methods"

mkdir -p "$RESULTS_BASE" /workspace/learn-si2e/logs

SUMMARY="$RESULTS_BASE/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "method,env,seed,success_rate_pct,fps" > "$SUMMARY"
fi

BASE_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005 --fast_se"

run_and_eval() {
    local method="$1"   # e.g. leiden-si2e
    local cluster="$2"  # leiden | infomap | kmeans
    local env="$3"
    local seed="$4"
    local frames="$5"
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
            echo "[EVAL-ONLY] ${model_name}"
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

run_pairs() {
    local method="$1" cluster="$2" env="$3" frames="$4"
    shift 4
    local pids=()
    for seed in "$@"; do
        run_and_eval "$method" "$cluster" "$env" "$seed" "$frames" &
        pids+=($!)
        if (( ${#pids[@]} >= 2 )); then
            wait "${pids[@]}"; pids=()
        fi
    done
    (( ${#pids[@]} > 0 )) && wait "${pids[@]}"
}

echo "=== DoorKey-8x8: Leiden ==="
run_pairs "leiden-si2e" "leiden" "MiniGrid-DoorKey-8x8-v0" "$FRAMES" "${SEEDS[@]}"

echo "=== DoorKey-8x8: Infomap ==="
run_pairs "infomap-si2e" "infomap" "MiniGrid-DoorKey-8x8-v0" "$FRAMES" "${SEEDS[@]}"

echo "=== KC-S3R2: Leiden ==="
run_pairs "leiden-si2e" "leiden" "MiniGrid-KeyCorridorS3R2-v0" "$FRAMES" "${SEEDS[@]}"

echo "=== KC-S3R2: Infomap ==="
run_pairs "infomap-si2e" "infomap" "MiniGrid-KeyCorridorS3R2-v0" "$FRAMES" "${SEEDS[@]}"

echo ""
echo "=== CLUSTERING METHODS SUMMARY ==="
python3 - <<'PYEOF'
import csv, os, numpy as np
from collections import defaultdict

kmeans_ref = {
    'DoorKey-8x8':        [100.0, 100.0],   # s1,s2 done; s3 pending
    'KeyCorridorS3R2':    [25.5, 100.0, 10.95],
}

p = "/workspace/learn-si2e/results/clustering-methods/summary.csv"
data = defaultdict(lambda: defaultdict(list))
fps_data = defaultdict(lambda: defaultdict(list))
with open(p) as f:
    for row in csv.DictReader(f):
        try:
            env = row['env'].replace('MiniGrid-', '').replace('-v0', '')
            data[env][row['method']].append(float(row['success_rate_pct']))
            if row.get('fps'):
                fps_data[env][row['method']].append(float(row['fps']))
        except: pass

print("\n--- Accuracy (success rate %) ---")
for env in sorted(data):
    print(f"\n{env}:")
    ref = kmeans_ref.get(env, [])
    if ref:
        print(f"  {'kmeans (ref)':32s} N={len(ref):2d}  {np.mean(ref):.1f}% ± {np.std(ref):.1f}")
    for method, vals in sorted(data[env].items()):
        print(f"  {method:32s} N={len(vals):2d}  {np.mean(vals):.1f}% ± {np.std(vals):.1f}")

print("\n--- FPS (training speed) ---")
all_fps = defaultdict(list)
for env in fps_data:
    for m, vals in fps_data[env].items():
        all_fps[m].extend(vals)
for m, vals in sorted(all_fps.items()):
    if vals:
        print(f"  {m:32s} {np.mean(vals):.0f} FPS")
print(f"  {'kmeans (ref DK-8x8)':32s} ~1879 FPS")
PYEOF
