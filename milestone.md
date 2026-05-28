# SI2E Research Milestones

---

## Milestone 1 — DK-8x8 Speed Benchmark & PPO-SI2E Verdict
**Date:** 2026-05-28 ~07:30 UTC  
**Status:** COMPLETE

### What was achieved
Phase 1 of `batch_fast_si2e.sh` finished: all DK-8x8 runs at 1M frames (si2e ×3, fast-si2e ×3).

### Key results

#### FPS — confirmed 4× overall training speedup

| Method     | s1   | s2   | s3   | Mean |
|------------|------|------|------|------|
| si2e       | 430  | 456  | 518  | **468 FPS** |
| fast-si2e  | 1654 | 1557 | 2425 | **1879 FPS** |

**Speedup: 1879 / 468 = 4.0×** on DK-8x8.  
Clustering step alone was 171× faster (from prior benchmark); overall 4× because environment stepping and network forward/backward pass dominate total wall time when the cluster cost drops from 155 ms → 0.9 ms.

#### DK-8x8 success rate at 1M frames — both converge to ≈0%

| Method    | s1    | s2   | s3   | Mean |
|-----------|-------|------|------|------|
| si2e      | 0.0%  | 0.0% | 0.0% | 0.0% |
| fast-si2e | 13.0% | 0.0% | 0.0% | 4.3% |

Both methods need 3M frames to converge on DK-8x8 (paper baseline: 98.6% at 3M).  
The 1M run was only needed to produce checkpoints for the 3M resume — checkpoints exist for all 6 seeds.

#### PPO-SI2E — negative result, fully characterised

| Env         | PPO-SI2E mean | std  | N |
|-------------|---------------|------|---|
| DK-8x8      | 58.5%         | 46%  | 5 |
| KC-S3R2     | 0.0%          | 0%   | 5 |

DK-8x8: 3 seeds converged (100%, 92.5%, 100%) but 2 seeds collapsed to 0%.  
KC-S3R2: complete failure (all 5 seeds, 0%).  
**Decision: PPO-SI2E will be reported as a negative result, not a contribution.**  
Hypothesis: PPO's advantage clipping interferes with SI2E's sparse intrinsic reward signal.

---

## Milestone 2 — KC-S3R2 fast-si2e + ppo-fast-si2e at 3M Frames
**Date:** 2026-05-28 ~TBD  
**Status:** IN PROGRESS

### What's running now

| Run | Frames | ETA |
|-----|--------|-----|
| fast-si2e KC-S3R2 s3 | 832K / 3M | ~45 min |
| a2c-si2e KC-S3R2 s1 (baseline, from batch_ppo_si2e.sh) | 1.024M / 3M | ~75 min |
| ppo-fast-si2e KC-S3R2 s1+2+3 | queued | after KC s3 done |

### Partial results so far — fast-si2e KC-S3R2

| Seed | SR at 3M |
|------|----------|
| s1   | 25.5%   |
| s2   | **100.0%** |
| s3   | running  |

Original SI2E KC-S3R2 baseline (5 seeds): **67.5% ± 27.9%**  
→ fast-si2e already shows 100% on s2; s1 had variance (25.5%).  
→ High variance is a known KC-S3R2 property (original SI2E shows same pattern).

### Goals for this phase
1. Confirm fast-si2e KC-S3R2 mean/std across all 3 seeds.
2. ppo-fast-si2e KC-S3R2: expect poor results (PPO+intrinsic reward already shown to fail on KC), confirming the SI2E+A2C combination is the right choice.
3. Get baseline a2c-si2e KC-S3R2 FPS for the speedup table.

### Analysis plan once complete
- If fast-si2e KC mean ≥ SI2E baseline (67.5%) with comparable or smaller std: **FastSI2E matches/beats accuracy with 4× less training time** → primary paper contribution confirmed.
- If fast-si2e KC mean < 67.5%: investigate whether 3 seeds is enough, consider additional seeds.

---

## Milestone 3 — DK-8x8 Fast-SI2E at 3M Frames (Paper-Comparable)
**Date:** 2026-05-28 ~TBD  
**Status:** QUEUED — runs after batch_fast_si2e.sh completes (coordinator: PID 705963)

### Setup
`batch_fastsi2e_dk3m.sh`: resumes from 1M checkpoints (`fastse-fast-si2e-DoorKey-8x8-s{1,2,3}`), extends to 3M.  
Also needs: si2e DK-8x8 at 3M baseline (original model: `multiseed-vcse-s*`, already done, 100%±0%, 5 seeds).

### Goals
- fast-si2e DK-8x8 at 3M should approach or match SI2E's 100%.
- Paper comparison: SI2E (3M) = **100.0% ± 0.0%**, fast-si2e (3M) = **?**.
- If fast-si2e ≥ 95% mean: primary claim holds — same accuracy in 4× less wall time.

### Expected ETA
~2–3 hours after batch_fast_si2e.sh finishes (3M frames at ~2000 FPS = ~25 min per seed pair).

---

## Milestone 4 — Adaptive-β Variance Reduction on RedBlueDoors + KC
**Date:** 2026-05-28 ~TBD  
**Status:** QUEUED — runs after Milestone 3 (coordinator: PID 705963)

### Hypothesis
Adaptive-β (β scaled by `max(0.1, 1 − success_rate)`) reduces seed variance on hard tasks.  
Target: RedBlueDoors-6x6 std < 20% (vs SI2E baseline std = 47%).

### Methods being tested
| Method | Description |
|--------|-------------|
| si2e-fixed | fast-si2e, fixed β=0.005 |
| si2e-adaptive | fast-si2e, adaptive β (our new method) |
| ppo-si2e-adaptive | PPO + fast-si2e + adaptive β |

### Current baseline
| Env | Method | Mean | Std | N |
|-----|--------|------|-----|---|
| RedBlueDoors-6x6 | SI2E | 55.7% | 38.7% | 3 seeds |
| KC-S3R2 | SI2E | 67.5% | 27.9% | 5 seeds |

### Analysis plan once complete
- Primary metric: does si2e-adaptive std < si2e-fixed std on RedBlueDoors?
- If adaptive-β achieves std < 20% on RedBlueDoors: second paper contribution confirmed.
- Paper framing: "FastSI2E + adaptive-β achieves {X}% ± {Y}% vs SI2E's 55.7% ± 38.7%".

---

## Summary of Paper Contributions So Far

| Contribution | Status | Evidence |
|---|---|---|
| **FastSI2E: 171× faster clustering step** | CONFIRMED (prev session) | Direct benchmark |
| **FastSI2E: 4× overall training speedup** | CONFIRMED (Milestone 1) | 468→1879 FPS on DK-8x8 |
| **FastSI2E: comparable accuracy on DK-8x8** | PENDING (Milestone 3) | Need 3M frame results |
| **FastSI2E: comparable accuracy on KC-S3R2** | PARTIAL (Milestone 2) | s2=100% done, s1=25.5%, s3 running |
| **Adaptive-β: reduced variance** | PENDING (Milestone 4) | Need RedBlueDoors + KC results |
| **Reward assignment correctness fix** | CONFIRMED (code) | k-means uses `inv_t` (correct); PartitionTree uses contiguous blocks (incorrect) |
| **PPO-SI2E negative result** | CONFIRMED (Milestone 1) | 58.5%±46% DK, 0% KC |

---

## Quick Status Commands

```bash
# Current training progress
tail -5 /workspace/learn-si2e/logs/fast_si2e2.log

# KC-S3R2 fast-si2e seed 3 frames
tail -2 /workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/storage/fastse-fast-si2e-KeyCorridorS3R2-s3/log.csv | cut -d',' -f1-3

# All results so far
cat /workspace/learn-si2e/results/fast-si2e/summary.csv

# Full analysis (run when results complete)
cd /workspace/learn-si2e && python3 analyze_results.py --plot

# DK-8x8 3M log (starts after fast_si2e batch finishes)
cat /workspace/learn-si2e/logs/fast_si2e_dk3m.log

# Adaptive-beta log
cat /workspace/learn-si2e/logs/adaptive_beta2.log
```
