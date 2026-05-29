#!/usr/bin/env bash
# run_phase2.sh
# Waits for batch_clustering_methods.sh (PID $1 or 3154976) to finish,
# then launches batch_phase2.sh.
#
# Usage:
#   nohup ./run_phase2.sh [CLUSTERING_PID] > logs/phase2_coordinator.log 2>&1 &

set -e

CLUSTERING_PID="${1:-3154976}"
cd /workspace/learn-si2e

echo "[$(date)] Phase-2 coordinator started. Waiting for clustering batch PID=${CLUSTERING_PID}..."
while kill -0 "$CLUSTERING_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] Clustering batch done. Launching phase2..."

chmod +x batch_phase2.sh
./batch_phase2.sh > logs/phase2.log 2>&1

echo "[$(date)] Phase 2 complete."
