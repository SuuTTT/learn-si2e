#!/usr/bin/env bash
# batch_adaptive_beta.sh
# Adaptive-β scheduling: test if beta_adaptive reduces seed variance on hard tasks.
# Target: RedBlueDoors-6x6 std < 20% (vs current SI2E std=47%).
#
# Acceleration: --fast_se on all runs, seeds run 2 at a time (RAM-limited).
#
# Usage:
#   chmod +x batch_adaptive_beta.sh
#   nohup ./batch_adaptive_beta.sh > logs/adaptive_beta.log 2>&1 &

set -e

SEEDS=(1 2 3 4 5)
FRAMES=3000000
A2C_DIR="/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files"
RESULTS_BASE="/workspace/learn-si2e/results/adaptive-beta"

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
    local env_short="${env//MiniGrid-/}"
    env_short="${env_short//-v0/}"

    local model_name="abeta-${method}-${env_short}-s${seed}"
    local out_dir="$RESULTS_BASE/${method}-${env_short}-s${seed}"
    mkdir -p "$out_dir"

    local log_csv="${A2C_DIR}/storage/${model_name}/log.csv"
    local status_pt="${A2C_DIR}/storage/${model_name}/status.pt"
    if [[ -f "$log_csv" ]] && [[ -f "$status_pt" ]]; then
        local last_f
        last_f=$(tail -1 "$log_csv" | cut -d',' -f2 | tr -d ' ')
        if [[ "$last_f" =~ ^[0-9]+$ ]] && (( last_f >= 2900000 )); then
            echo "[SKIP-TRAIN] ${model_name} already at ${last_f} frames, has checkpoint"
            if grep -q "^${method},${env},${seed}," "$SUMMARY" 2>/dev/null; then
                echo "[SKIP-EVAL] ${model_name} already in summary"
                return
            fi
            echo "[EVAL-ONLY] ${model_name}: running eval on existing checkpoint..."
            local eval_out
            eval_out=$(cd "$A2C_DIR" && python3 -m scripts.eval_success \
                --env "$env" --model "$model_name" \
                --episodes 200 --argmax --seed 999 2>&1)
            echo "$eval_out"
            local sr; sr=$(echo "$eval_out" | grep "^SUCCESS_RATE=" | cut -d'=' -f2)
            local fps; fps=$(grep "FPS" "${out_dir}/train.log" 2>/dev/null | awk -F'FPS ' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
            echo "${method},${env},${seed},${sr},${fps}" >> "$SUMMARY"
            echo "[DONE] ${model_name}: SR=${sr}%  FPS=${fps}"
            return
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
    local final_frames
    final_frames=$(tail -1 "${out_dir}/log.csv" 2>/dev/null | cut -d',' -f2 | tr -d ' ' || echo "$FRAMES")
    echo "${method},${env},${seed},${sr},${final_frames}" >> "$SUMMARY"
    echo "[DONE] ${model_name}: SR=${sr}%"
}

# All methods use --fast_se for speed consistency
SI2E_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005 --fast_se"
SI2E_ADAPTIVE_FLAGS="--algo a2c --use_entropy_reward --use_value_condition --beta 0.005 --beta_adaptive --fast_se"
PPO_ADAPTIVE_FLAGS="--algo ppo --use_entropy_reward --use_value_condition --beta 0.005 --beta_adaptive --fast_se"

# Run seeds in pairs — each training run uses ~6 GB RAM, 15 GB available after PPO finishes
run_pairs() {
    local method="$1" env="$2" flags="$3"
    shift 3
    local pids=()
    for seed in "$@"; do
        run_and_eval "$method" "$env" "$seed" "$flags" &
        pids+=($!)
        if (( ${#pids[@]} >= 2 )); then
            wait "${pids[@]}"; pids=()
        fi
    done
    (( ${#pids[@]} > 0 )) && wait "${pids[@]}"
}

# ── RedBlueDoors: primary variance reduction test ──────────────────────────
echo "=== RedBlueDoors: SI2E fixed-β  (seeds 2 at a time) ==="
run_pairs "si2e-fixed" "MiniGrid-RedBlueDoors-6x6-v0" "$SI2E_FLAGS" "${SEEDS[@]}"

echo "=== RedBlueDoors: SI2E adaptive-β  (seeds 2 at a time) ==="
run_pairs "si2e-adaptive" "MiniGrid-RedBlueDoors-6x6-v0" "$SI2E_ADAPTIVE_FLAGS" "${SEEDS[@]}"

echo "=== RedBlueDoors: PPO-SI2E adaptive-β  (seeds 2 at a time) ==="
run_pairs "ppo-si2e-adaptive" "MiniGrid-RedBlueDoors-6x6-v0" "$PPO_ADAPTIVE_FLAGS" "${SEEDS[@]}"

# ── KC-S3R2: secondary variance reduction test ─────────────────────────────
echo "=== KC-S3R2: SI2E adaptive-β  (seeds 2 at a time) ==="
run_pairs "si2e-adaptive" "MiniGrid-KeyCorridorS3R2-v0" "$SI2E_ADAPTIVE_FLAGS" "${SEEDS[@]}"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== ADAPTIVE-β SUMMARY ==="
python3 - <<'PYEOF'
import csv, os
from collections import defaultdict
p = "/workspace/learn-si2e/results/adaptive-beta/summary.csv"
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
