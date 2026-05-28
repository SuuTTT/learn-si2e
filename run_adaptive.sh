#!/usr/bin/env bash
# Wait for batch_fast_si2e to finish, then run adaptive_beta
FAST_PID=704058
echo "[$(date)] Waiting for fast_si2e (PID $FAST_PID) to finish..."
wait $FAST_PID 2>/dev/null || true
while kill -0 $FAST_PID 2>/dev/null; do sleep 30; done
echo "[$(date)] fast_si2e done. Starting 3M DK-8x8 fast-si2e runs..."
cd /workspace/learn-si2e
./batch_fastsi2e_dk3m.sh > logs/fast_si2e_dk3m.log 2>&1
echo "[$(date)] 3M DK-8x8 done. Starting adaptive_beta..."
./batch_adaptive_beta.sh > logs/adaptive_beta2.log 2>&1
echo "[$(date)] adaptive_beta done."
