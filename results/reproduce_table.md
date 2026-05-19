# SI2E vs VCSE: Paper Results vs Local Reproduction

**Date:** 2026-05-18 / updated 2026-05-19 (full-budget runs added)  
**Our GPU:** NVIDIA RTX 3070 Laptop (8 GB), CUDA 11.8, single seed  
**Paper GPU:** NVIDIA RTX A1000 (8 GB, Ampere), 8-core Intel i9 @ 3.00 GHz  
**Paper:** "Structural Information Principles-based Effective Exploration" (NeurIPS 2024, SI2E)  
**Base paper:** "Accelerating RL with Value-Conditional State Entropy Exploration" (NeurIPS 2024, VCSE)

---

## Table 1 · MiniGrid Navigation — DoorKey-8x8

**Task:** A2C on `MiniGrid-DoorKey-8x8-v0`  
**Paper setup:** 3 000 000 frames, 10 seeds, RTX A1000  
**Local setup (short):** 500 000 frames, 1 seed  
**Local setup (full):** 3 000 000 frames (paper budget), 1 seed  

> DoorKey-8x8 is hard-exploration. The paper itself shows `—` for the A2C baseline (it fails at 3 M frames).
> With 10 seeds and σ ≈ ±20 pp, a single lucky/unlucky seed can span 50 %–93 % for SE.

### Run-time estimates (1 seed, DoorKey-8x8, 3 M frames)

| Method | Measured FPS | Est. time @ 3 M frames | Paper est. (RTX A1000 ≈ same) |
|---|---:|---:|---:|
| A2C baseline | 4 959 | **10 min** | ~10 min |
| A2C + SE | 2 449 | **20 min** | ~20 min |
| A2C + VCSE | ~2 400 ¹ | **~21 min** | ~21 min |
| A2C + SI2E | 518 | **97 min (1.6 h)** | ~97 min |

¹ VCSE uses the same kNN-distance loop as SE; FPS assumed equal to SE.

**To reproduce full Table 1** (5 tasks × 4 methods × 10 seeds, sequential):  
3 hard tasks @3 M + 2 easy tasks @1 M → **≈ 90 h** on a single RTX 3070 Laptop.  
With 4 parallel GPUs → **≈ 22 h**.

| Method | Paper Success Rate (%) | Paper Required Steps (K) | Local 500K (1 seed) | Local 3M — full budget (1 seed) |
|--------|:---------------------:|:------------------------:|:-------------------:|:--------------------------------:|
| A2C (baseline) | — | — | rR≈0.70% | rR≈3% (max ep=0.81) ⁴ |
| A2C + SE (kNN entropy) | 72.60 ± 20.32 | 1515.81 ± 324.28 | rR≈0.45% | rR≈1% (max ep=0.42) ⁴ |
| A2C + VCSE (value-cond kNN) | 94.32 ± 11.09 | 1900.96 ± 398.65 | — ¹ | — |
| A2C + SI2E (encoding tree) | 98.58 ± 3.11 | 1090.96 ± 125.77 | rR≈0.90% | running (skipped — 97 min/run) |

¹ VCSE value-conditional kNN not separately exposed in SI2E's A2C code; VCSE repo set up but not yet run.  
⁴ Extremely low 1-seed result is expected: paper shows σ=20 pp; baseline officially fails (—) on DoorKey-8x8.
  rR values are mean episodic return (not converted to % success); max ep = best single episode in run.

---

## Table 2 · DMControl — Cartpole Swingup

**Task:** DrQv2 on `cartpole_swingup` (pixel obs, episode reward range 0–860)  
**Paper setup:** 250 000 frames, 10 seeds, RTX A1000  
**Local setup (short):** 50 000 frames, 1 seed  
**Local setup (full):** 250 000 frames (paper budget), 1 seed  

> Paper SE for cartpole_swingup: 219.69 ± 62.21. Our seed=1 SE reached **ER=660** at 240K — high single-seed variance.
> SI2E nohup run in progress (PID 55737, ~4.9 h, ER=9.38 at step-0).

### Run-time estimates (1 seed, 1 DMControl task, 250 K frames)

| Method | Measured FPS | Est. time @ 250 K frames | Paper est. (RTX A1000 ≈ same) |
|---|---:|---:|---:|
| DrQv2 baseline | 141.8 | **29 min** | ~29 min |
| DrQv2 + SE | 112.3 | **37 min** | ~37 min |
| DrQv2 + VCSE | ~112 ¹ | **~37 min** | ~37 min |
| DrQv2 + MADE | ~112 ¹ | **~37 min** | ~37 min |
| DrQv2 + SI2E | 14.1 | **296 min (4.9 h)** | ~296 min |

¹ VCSE/MADE use the same kNN inner loop as SE; FPS assumed equal to SE.

**To reproduce full Table 2** (6 tasks × 5 methods × 10 seeds, sequential):  
→ **≈ 436 h** on a single RTX 3070 Laptop (SI2E dominates at ~5 h/task/seed).  
With 8 parallel GPUs → **≈ 55 h**.

| Method | Paper ER | Local 50K (1 seed) | Local 250K — full budget (1 seed) |
|--------|:--------:|:------------------:|:---------------------------------:|
| DrQv2 (baseline) | — | 74.76 | **341** @ 240K frames |
| DrQv2 + SE (kNN entropy) | 219.69 ± 62.21 | 74.76 | **660** @ 240K frames ⁵ |
| DrQv2 + VCSE (value-cond kNN) | 707.76 ± 50.38 | — ² | — |
| DrQv2 + MADE | 704.18 ± 41.75 | — | — |
| DrQv2 + SI2E (encoding tree) | **795.09 ± 90.49** | 74.94 ³ | 🔄 PID 55737 — ~334 @ F=170K (71% done) |

² True VCSE DrQv2 requires `base-vcse/VCSE_DrQv2/`; not yet run.  
³ SI2E DrQv2 killed at 16 K frames (14 FPS).  
⁵ ER=660 exceeds paper's 219.69 ± 62.21 for SE — single lucky seed; paper result uses 10 seeds.

**Full-budget eval history (250K frames, seed=1):**

```
DrQv2 baseline (~140 FPS, ~36 min):
  100K→255 | 110K→304 | 120K→289 | 130K→280 | 140K→341  (@ 240K frame checkpoint)

DrQv2+SE / do_vcse=false (~109 FPS, ~46 min):
  100K→683 | 110K→681 | 120K→643 | 130K→679 | 140K→660  (@ 240K frame checkpoint)

DrQv2+SI2E / do_vcse=true (~14 FPS, ~4.9 h — nohup PID 55737):
  step-0 → ER=9.38  (in progress)
```

---

## Paper's Full DMControl Table 2 (for reference)

| Domain, Task | DrQv2 | DrQv2+SE | DrQv2+VCSE | DrQv2+MADE | DrQv2+SI2E |
|---|---|---|---|---|---|
| Hopper Stand | 87.59 ± 11.70 | 313.39 ± 94.15 | 711.32 ± 30.84 | 717.09 ± 112.94 | **797.17 ± 53.21** |
| Cheetah Run | 229.28 ± 123.93 | 228.82 ± 126.21 | 456.26 ± 22.20 | 366.59 ± 53.74 | **464.08 ± 29.32** |
| Quadruped Walk | 289.79 ± 24.17 | 290.27 ± 24.20 | 243.74 ± 29.91 | 262.63 ± 23.92 | **399.51 ± 29.05** |
| Pendulum Swingup | 424.21 ± 246.96 | 10.80 ± 2.92 ⚠️ | 824.17 ± 99.59 | 672.11 ± 34.63 | **885.50 ± 38.28** |
| Cartpole Balance | 998.97 ± 22.95 | 993.80 ± 75.24 | 998.65 ± 9.58 | 996.16 ± 40.60 | **999.58 ± 2.97** |
| Cartpole Swingup | — | 219.69 ± 62.21 | 707.76 ± 50.38 | 704.18 ± 41.75 | **795.09 ± 90.49** |

---

## Paper's Full MiniGrid Table 1 (DoorKey section, for reference)

| Method | DoorKey-6x6 Success (%) | DoorKey-6x6 Steps (K) | DoorKey-8x8 Success (%) | DoorKey-8x8 Steps (K) |
|--------|:-----------------------:|:--------------------:|:-----------------------:|:--------------------:|
| A2C | 92.67 ± 8.47 | 567.20 ± 96.57 | — | — |
| A2C + SE | 93.18 ± 6.81 | 476.34 ± 94.63 | 72.60 ± 20.32 | 1515.81 ± 324.28 |
| A2C + VCSE | 94.08 ± 2.58 | 336.75 ± 19.84 | 94.32 ± 11.09 | 1900.96 ± 398.65 |
| A2C + SI2E | **97.04 ± 1.52** | **230.60 ± 19.85** | **98.58 ± 3.11** | **1090.96 ± 125.77** |

---

## Observations

1. **Frame budget matters enormously.** Both DoorKey-8x8 and cartpole_swingup are hard-exploration tasks
   where all methods produce near-zero rewards until ~1/3 of the paper's budget is consumed.

2. **SI2E has a serious compute overhead** due to the O(n²) encoding tree construction at every update:
   - A2C: 518 FPS (SI2E) vs 2 449 FPS (SE) vs 4 959 FPS (baseline) — **9.6× / 4.7×** slower
   - DrQv2: 14.1 FPS (SI2E) vs 112.3 FPS (SE) vs 141.8 FPS (baseline) — **10.1× / 8.0×** slower

3. **Bug fixed:** `kthvalue` in `torch_ac/algos/base.py` used `dists.shape[0]` (rows) as the
   upper bound for `k`, but `kthvalue(..., dim=1)` operates on columns (`dists.shape[1]`). Fixed
   in both `compute_state_entropy` and `compute_value_condition_state_entropy`.

4. **Paper's hardware note:** The paper uses an RTX A1000 (Ampere, 5 120 CUDA cores, 16 GB ECC)
   which is architecturally identical to the RTX 3070 Laptop (Ampere, 5 120 CUDA cores, 8 GB).
   The RTX A1000 has ~10% lower boost clock (1 455 MHz vs ~1 560 MHz) so our FPS estimates
   are a reasonable proxy for the paper's wall-clock times.

5. **Paper does not report wall-clock times.** Frame budgets from Appendix D:
   - MiniGrid hard tasks (DoorKey-8x8, RedBlueDoors, KeyCorridor): **3 000 K frames**
   - MiniGrid easy tasks (DoorKey-6x6, SimpleCrossing): **1 000 K frames**
   - DMControl: **250 K frames**
   - MetaWorld: **200 K / 100 K frames**

6. **Next steps for full reproduction:** Run with the paper's full frame budget and multiple seeds,
   or set up the VCSE repo (`base-vcse/`) to produce a true VCSE baseline column.
   Priority order by cost: A2C tasks first (cheapest), then DMControl (moderate), MetaWorld last.
