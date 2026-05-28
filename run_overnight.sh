#!/usr/bin/env bash
# Run fast_si2e and adaptive_beta sequentially (alongside the running PPO).
# Memory budget: PPO ~6 GB + run_pairs ~12 GB = 18 GB (within 24 GB RAM+swap).
set -e
cd /workspace/learn-si2e
echo "[$(date)] Starting fast_si2e..."
chmod +x batch_fast_si2e.sh batch_adaptive_beta.sh
./batch_fast_si2e.sh > logs/fast_si2e.log 2>&1
echo "[$(date)] fast_si2e done. Starting adaptive_beta..."
./batch_adaptive_beta.sh > logs/adaptive_beta.log 2>&1
echo "[$(date)] All experiments done."
