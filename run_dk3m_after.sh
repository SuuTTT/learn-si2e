#!/usr/bin/env bash
# Wait for run_adaptive.sh (PID 705963 = batch_adaptive_beta) to finish, then run dk3m.
ADAPTIVE_PID=705963
echo "[$(date)] Waiting for adaptive_beta (PID $ADAPTIVE_PID) to finish..."
while kill -0 $ADAPTIVE_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] adaptive_beta done. Starting 3M DK-8x8 fast-si2e runs..."
cd /workspace/learn-si2e
./batch_fastsi2e_dk3m.sh > logs/fast_si2e_dk3m.log 2>&1
echo "[$(date)] DK-8x8 3M done."
