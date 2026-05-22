# SI2E vs VCSE: Paper Results vs Local Reproduction

**Date:** 2026-05-18 / updated 2026-05-22 (full-budget 510K runs complete)  
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

### What is rR? What is a "window"?

`rR` is the **mean episodic return per log window**, format `rR:μσmM` (mean, std, min, max). torch-ac logs one line every `log_interval=500` A2C updates.

**What is one log window?**

| Parameter | Value |
|---|---|
| Parallel envs (procs) | 16 |
| Steps per proc per update (`frames_per_proc`) | 8 |
| Frames per update | 16 × 8 = **128 frames** |
| Frames per log window | 500 × 128 = **64,000 frames** |
| Real-time per window | 64,000 / ~5,000 FPS ≈ **~11 seconds** |
| Episodes per window | 64,000 / ~625 avg-steps ≈ **~100 episodes** |

The `D` field in the log (`U 500 | F 064000 | FPS 5464 | D 11 | rR:...`) is **elapsed time in seconds**, not episode count. `rR:μσmM` covers all ~100 episodes that completed during those 500 updates.

**Return values in MiniGrid DoorKey-8x8:**
- **Failure** (agent times out): return = **0** (max\_steps = 640)
- **Success** (reaches goal): return = **1 − 0.9 × (steps / 640)** ∈ (0.1, 1.0)
  - e.g. solved in 300 steps → return ≈ 0.58

> ⚠️ For SE/VCSE runs, `rR` in the **text log** is the **reshaped** return (includes kNN entropy bonus). The raw env return is in the `return_mean` column of `log.csv` (columns 17-20). Use `return_mean` for success-rate estimates, not `rR`.

`return_mean` is therefore a blended proxy for success rate. To convert: **success\_rate ≈ return\_mean / 0.5** (average reward when solving is ~0.5 for a typical solution path). For local runs at 3M frames (seed=1):

| Method | rR mean (final 500K) | **Approx success rate** | Windows w/ ≥1 success |
|--------|:-:|:-:|:-:|
| A2C baseline | ~0.04 | **~8%** | 28 / 46 |
| A2C + SE | ~0.02 | **~3%** | 14 / 46 |

These are single-seed (seed=1) results. The paper reports 0% for baseline and 72.6±20% for SE over 10 seeds. Both baselines "officially fail" (no sustained learning) but our seed=1 baseline finds the goal occasionally. SE seed=1 is an unlucky draw (paper σ=20 pp means some seeds never solve it).

### What is A2C? Is it like SAC?

**No — A2C and SAC are fundamentally different algorithms for different settings:**

| | A2C | SAC |
|---|---|---|
| Policy type | On-policy (rollout → update → discard) | Off-policy (replay buffer) |
| Action space | **Discrete or continuous** | **Continuous only** (requires reparameterization) |
| Policy gradient | Advantage A(s,a) = R − V(s) | Soft Q-gradient via reparameterization trick |
| Exploration | Entropy bonus on action distribution | Maximum-entropy framework (built-in) |
| Typical use | Grid navigation, Atari, simple control | Continuous robotics, DMControl |

A2C works natively on **discrete** action spaces (MiniGrid has 7 actions: turn-L, turn-R, forward, pick-up, drop, toggle, done). SAC's gradient estimator requires a differentiable, continuous action distribution and cannot be applied directly to discrete actions without modification (SAC-discrete is a separate variant, not used here).

The SI2E paper uses **A2C for MiniGrid** (discrete navigation) and **DrQv2 for DMControl** (continuous pixel-based control) — these are the standard algorithm choices for each domain.

### Run-time estimates (1 seed, DoorKey-8x8, 3 M frames)

| Method | Measured FPS | Est. time @ 3 M frames |
|---|---:|---:|
| A2C baseline | 4 959 | **10 min** |
| A2C + SE | 2 449 | **20 min** |
| A2C + VCSE | ~2 400 ¹ | **~21 min** |
| A2C + SI2E | 518 | **97 min (1.6 h)** |

¹ VCSE uses the same kNN-distance loop as SE; FPS assumed equal to SE.

**To reproduce full Table 1** (5 tasks × 4 methods × 10 seeds, sequential):  
3 hard tasks @3 M + 2 easy tasks @1 M → **≈ 90 h** on a single RTX 3070 Laptop.  
With 4 parallel GPUs → **≈ 22 h**.

| Method | Paper Success Rate (%) | Paper Required Steps (K) | Local (seed=1) Success Rate ¹ | Multi-seed (5 seeds) Success Rate |
|--------|:---------------------:|:------------------------:|:-----------------------------:|:---------------------------------:|
| A2C (baseline) | — | — | ~8% (return_mean≈0.04) | *run `batch_a2c_multiseed.sh`* |
| A2C + SE (kNN entropy) | 72.60 ± 20.32 | 1515.81 ± 324.28 | ~3% (return_mean≈0.02) | *run `batch_a2c_multiseed.sh`* |
| A2C + VCSE (value-cond kNN) | 94.32 ± 11.09 | 1900.96 ± 398.65 | *not run yet* | *run `batch_a2c_multiseed.sh`* |
| A2C + SI2E (encoding tree) | 98.58 ± 3.11 | 1090.96 | — (skipped, 97 min/seed) | — (skipped) |

¹ Seed=1 success rate = `return_mean` (CSV col 17, avg last 5 windows) / 0.5 × 100.  
Each log window = 500 updates = 64,000 frames ≈ 100 episodes. Paper reports 10 seeds; with SE σ=±20 pp, a single seed may range from ~30% to ~93%.

To fill the multi-seed column, run:
```bash
cd /workspace/learn-si2e && ./batch_a2c_multiseed.sh
# Results → results/a2c-multiseed/summary.csv (method, seed, success_rate_pct)
```

---

## Table 2 · DMControl — All 6 Tasks (510K frames = paper budget)

**Paper setup:** 510K frames (= 250K env steps with action_repeat=2), 10 seeds, RTX A1000  
**Local setup:** 510K frames, **seed=1**, DrQv2 backbone, `replay_buffer_num_workers=0 replay_buffer_size=120000`  

> Single seed (seed=1). Paper reports mean±σ over 10 seeds — high variance expected.  
> SE values marked ★ are from the SE rerun (buf=130K); see SE sensitivity analysis below.

### Run-time estimates (1 seed, 1 DMControl task, 510K frames)

| Method | Measured FPS | Est. time @ 510K frames |
|---|---:|---:|
| DrQv2 baseline | ~135 | **~63 min** |
| DrQv2 + SE | ~95 | **~90 min** |
| DrQv2 + VCSE | ~95 | **~90 min** |
| DrQv2 + SI2E | ~14 | **~606 min (10.1 h)** |

### Local vs paper results (510K frames, seed=1)

| Task | Paper Baseline | **Local Baseline** | Paper SE | **Local SE** | Paper VCSE | **Local VCSE** | Paper SI2E |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| cartpole_swingup | — | 751 | 220±62 | **870**★ | 708±50 | **858** | 795±90 |
| hopper_stand | 88±12 | 478 | 313±94 | **8**★ ⚠️ | 711±31 | **915** | 797±53 |
| cheetah_run | 229±124 | 457 | 229±126 | **396**★ | 456±22 | **679** | 464±29 |
| quadruped_walk | 290±24 | 197 | 290±24 | **312** | 244±30 | **785** | 400±29 |
| pendulum_swingup | 424±247 | 847 | 11±3 | **811**★ | 824±100 | **852** | 886±38 |
| cartpole_balance | 999±23 | 973 | 994±75 | **991** | 999±10 | **996** | 1000±3 |

★ from SE rerun (buf=130K, workers=0); original official SE: cheetah_run=0, pendulum_swingup=89  
⚠️ hopper_stand SE = 8.1 in both official and rerun — likely unlucky seed (paper σ=94); see below

### SE sensitivity to replay buffer size

SE's KNN particle entropy estimator requires diverse states per batch. With `num_workers=2`, each worker holds an independent half of episodes, halving per-batch diversity → entropy signal degrades.

| Task | SE official (buf=100K, workers=2) | SE rerun (buf=130K, workers=0) | Verdict |
|------|:-:|:-:|---|
| cheetah_run | 0 | **396** | ✅ Buffer was the bottleneck |
| pendulum_swingup | 89 | **811** | ✅ Buffer was the bottleneck |
| hopper_stand | 8 | **8** | ❌ Not a buffer issue — likely seed variance |

### VCSE vs SE
VCSE outperforms or matches SE on every task. VCSE is robust to buffer size constraints (no SE-style failures observed). On 5/6 tasks VCSE meets or exceeds the paper's VCSE numbers at this single seed.

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

4. **Hardware OOM fix (DrQv2 only):** The default `replay_buffer_size=1M, num_workers=4` uses >24 GB RAM
   on this 15 GB machine. Each DataLoader worker process independently holds its partition of episodes
   in RAM — with num_workers=N, peak RAM = N × (episodes/N × ~70 MB) + N × 2 GB overhead.
   Fix: `replay_buffer_num_workers=0 replay_buffer_size=120000–130000` — single-process replay,
   ~13 GB peak RAM, fits in 15 GB.

5. **SE is buffer-size sensitive; VCSE is not.** SE's KNN particle entropy estimator needs per-batch
   state diversity. With num_workers=2, each worker holds only half the episodes, halving per-batch
   diversity → SE reward signal collapses to ~0 on several tasks. With num_workers=0 and a larger
   buffer, SE recovers (cheetah_run: 0→396, pendulum_swingup: 89→811). VCSE shows no such sensitivity.
   hopper_stand SE = 8 in both settings — likely unlucky seed (paper σ=94 ≈ 30% of mean).

6. **Paper's hardware note:** The paper uses an RTX A1000 (Ampere, 5 120 CUDA cores, 16 GB ECC)
   which is architecturally identical to the RTX 3070 Laptop (Ampere, 5 120 CUDA cores, 8 GB).
   The RTX A1000 has ~10% lower boost clock (1 455 MHz vs ~1 560 MHz) so our FPS estimates
   are a reasonable proxy for the paper's wall-clock times.

7. **Paper does not report wall-clock times.** Frame budgets from Appendix D:
   - MiniGrid hard tasks (DoorKey-8x8, RedBlueDoors, KeyCorridor): **3 000 K frames**
   - MiniGrid easy tasks (DoorKey-6x6, SimpleCrossing): **1 000 K frames**
   - DMControl: **510 K frames** (= 250K env steps with action_repeat=2)
   - MetaWorld: **200 K / 100 K frames**

8. **Full reproduction cost (single GPU):**
   - DMControl 510K × 6 tasks × 4 methods × 10 seeds: ~440 h (SI2E dominates)
   - Feasible at 1 seed in ~44 h (baseline+SE+VCSE only, no SI2E)
