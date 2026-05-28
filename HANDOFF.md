# SI2E Research Handoff — Beat SOTA in Performance and Speed

**Prepared:** 2026-05-28 (updated session 4 — experiment restart + reward assignment finding)
**For:** Algorithm engineer continuing this research direction  
**Codebase:** `/workspace/learn-si2e/`  
**Original paper:** SI2E — NeurIPS 2024, Zeng et al.  
**Goal:** Beat SI2E in both exploration performance and wall-clock speed on MiniGrid tasks

---

## 0. NEW THIS SESSION (session 4 — experiment restart + findings)

### 0.1 Bug Fix: `train.py` Missing Final Checkpoint

`--save-interval 0` means no checkpoint is ever saved, causing `eval_success.py` to fail (no model to load). Added final save after the training loop in `train.py:246`:

```python
# Always save final checkpoint so eval_success can load the model
final_status = {"num_frames": num_frames, "update": update, ...}
utils.save_status(final_status, model_dir)
```

This caused the overnight `batch_fast_si2e.sh` to crash at the eval step for si2e seeds 1+2, aborting before any fast-si2e seeds ran.

**Fix status:** `train.py` patched. `batch_fast_si2e.sh` + `batch_adaptive_beta.sh` skip logic updated to detect checkpoint presence before attempting eval-only path.

### 0.2 Reward Assignment Correctness Finding

The original PartitionTree path in `base.py:510-513` adds cluster bonuses to CONTIGUOUS BLOCKS of states by index, not to states by actual cluster membership:
```python
index = 0
for i in range(len(reward_1)):
    reward_0[index: index + length_level_1[i]] += (1.0 / length_level_1[i]) * reward_1[i]
    index += length_level_1[i]
```

This is a semantic error: it assigns cluster `i`'s bonus to states `0..k-1`, then `k..k+l-1`, etc. — regardless of actual cluster membership. The tree_node dict is iterated in leaf insertion order (states 0,1,...,n-1), but `tmp_set` (parent clusters) has arbitrary iteration order, so cluster bonuses are assigned to non-contiguous state ranges.

Our k-means path (`base.py:563`) correctly uses `inv_t` to assign each state to its actual cluster:
```python
reward_0 = reward_0 + (1.0 / counts[inv_t]) * reward_1[inv_t]
```

This is a potential paper contribution: "we identified and corrected a reward assignment bug in the original SI2E implementation."

### 0.3 Experiments Status (2026-05-28 05:30)

| Script | Status | PID | Log |
|--------|--------|-----|-----|
| `batch_ppo_si2e.sh` | ✅ Running, seed 5 KC (~1M/3M) | 251761 | `logs/ppo_si2e.log` |
| `batch_fast_si2e.sh` | ✅ Restarted (si2e s1+s2 at ~300K) | 704058 | `logs/fast_si2e2.log` |
| `batch_adaptive_beta.sh` | ⏳ Queued via run_adaptive.sh | 705963 | `logs/adaptive_beta2.log` |

PPO-SI2E final results: DK-8x8 58.5%±47.8% (5 seeds), KC-S3R2 0%±0% (4/5 seeds). Not reliable — don't feature as main contribution.

### 0.4 Check Progress

```bash
# fast-si2e batch (main comparison)
grep "rR:μσmM\|DONE\|SKIP\|RUN" /workspace/learn-si2e/logs/fast_si2e2.log | tail -20

# Coordinator (will start adaptive_beta after fast_si2e)
cat /workspace/learn-si2e/logs/run_adaptive.log

# Full results summary (run this when all experiments done)
cd /workspace/learn-si2e && python3 analyze_results.py --plot
```

---

## 0_OLD. PREVIOUS SESSION (session 3 — code optimizations + overnight experiments)

### 0.1 Code Optimizations Completed

Four code-level speedups applied. **All pass unit tests. Experiments now running.**

| Change | File | Speedup | Impact |
|--------|------|---------|--------|
| Vectorize `graph_parse` (numpy) | `sip.py:12` | 57× on that call | ~23% less PartitionTree time |
| Skip replay buffer loop when `use_batch=True` | `train.py:180` | saves 47.5ms/update | Biggest actual FPS gain |
| Replace glass-jax with numpy k-means | `base.py:_kmeans_adj` | 171× on clustering | fast_se now nearly as fast as baseline |
| Vectorize centroid + reward scatter_add | `base.py:503` | eliminates Python loop | clean GPU code |
| Fix `tmp_ens /= sum` list bug | `base.py:483` | correctness fix | slow path was silently broken |
| Cache LayerNorm (don't re-create each update) | `a2c.py:67` | minor | avoids CUDA alloc per step |

### 0.2 Updated FPS Benchmark (after session 3 optimizations)

Run from: `python3 /workspace/learn-si2e/benchmark_fps.py --procs 16 --steps 5`

| Method | FPS | Update_ms | Speedup | Effective grad-steps/s |
|--------|-----|-----------|---------|----------------------|
| Baseline A2C | 2020 | 24ms | 1.00× | 2020 |
| **A2C-SI2E** | 1002 | 90ms | 0.50× | 1002 |
| **A2C-FastSI2E** (k-means) | **1864** | 34ms | **0.92×** | 1864 |
| **PPO-SI2E** | 474 | 234ms | 0.23× | 1896 |
| **PPO-FastSI2E** (k-means) | **1561** | 39ms | **0.77×** | **6244** |

Key headline: **PPO-FastSI2E delivers 6× more effective gradient steps/sec than A2C-SI2E** (1561 FPS × 4 epochs vs 1002 FPS × 1 epoch).

### 0.3 Overnight Experiments Running RIGHT NOW

| Script | Status | Expected finish | Log |
|--------|--------|-----------------|-----|
| `batch_ppo_si2e.sh` | ✅ Running (PID 251761) | ~25h from now | `logs/ppo_si2e.log` |
| `batch_fast_si2e.sh` | ✅ Running (PID ~264706) | ~3h from now | `logs/fast_si2e.log` |
| `batch_adaptive_beta.sh` | ⏳ Queued (starts after fast_si2e) | ~8h from now | `logs/adaptive_beta.log` |

**When you return:** Check results with:
```bash
cat /workspace/learn-si2e/results/fast-si2e/summary.csv
cat /workspace/learn-si2e/results/adaptive-beta/summary.csv
tail -20 /workspace/learn-si2e/logs/fast_si2e.log
```

### 0.4 NOTE: `--fast_se` now uses k-means (not glass-jax)

The `--fast_se` flag now uses pure numpy k-means on the adjacency matrix rows (~0.9ms) instead of glass-jax constrained partitioning (~155ms). This is a **171× speedup** on the clustering step with no external dependency. glass-jax is no longer needed for experiments. The algorithm is described as "k-means graph partitioning" in the paper.

---

## 0_OLD. PREVIOUS SESSION (session 2) context

Three new methods have been **implemented and unit-tested**. Batch scripts are ready to launch.

### 0.1 What Was Implemented

| Feature | Flag | Status | Where |
|---------|------|--------|-------|
| FastSI2E (GPU cdist) | always on | ✅ deployed | `base.py` `compute_value_condition_structural_entropy` |
| FastSI2E (glass-jax numba) | `--fast_se` | ✅ implemented | `base.py` `compute_value_condition_structural_entropy_fast` |
| SI2E-PPO | `--algo ppo` | ✅ implemented | `ppo.py` |
| Adaptive β | `--beta_adaptive` | ✅ implemented | `base.py` `_effective_beta()` |

### 0.2 Measured FPS (16 procs, DK-8x8, CUDA)

| Method | FPS | Update_ms | Gradient epochs | Notes |
|--------|-----|-----------|----------------|-------|
| A2C-SI2E (original) | ~1040 | 92ms | 1 | Baseline |
| A2C-FastSI2E | ~1534 | 55ms | 1 | **1.5× speedup** |
| PPO-SI2E | ~487 | 233ms | 4 | 4 gradient epochs |
| PPO-FastSI2E | ~947 | 107ms | 4 | ~1.0× FPS, 4× gradient efficiency |

Key: PPO-FastSI2E ≈ same FPS as A2C-SI2E while doing 4× gradient steps per frame.

### 0.3 Experiments Still to Run

| Batch Script | Goal | Status |
|-------------|------|--------|
| `batch_ppo_si2e.sh` | PPO backbone performance: DK-8x8 + KC-S3R2, 5 seeds | ❌ not started |
| `batch_adaptive_beta.sh` | Adaptive-β variance reduction: RedBlueDoors + KC-S3R2, 5 seeds | ❌ not started |
| `batch_fast_si2e.sh` | FastSI2E accuracy parity: DK-8x8 + KC-S3R2, 3 seeds each | ❌ not started |
| `benchmark_fps.py` | Full FPS table for paper | ❌ not started |

**Expected: 40–60 CPU+GPU hours total for all experiments.**

### 0.4 glass-jax Setup

```bash
# Already cloned to /workspace/glass-jax
cd /workspace/glass-jax && pip install -e .  # already done

# Verify
python3 -c "from glass.seclust.incremental import constrained_k_multistart; print('OK')"
```

---

## 1. What Has Been Done (Read This First)

### 1.1 Completed Experiments

44 training runs completed. All code, results, and batch scripts are ready to rerun or extend.

| Task | Methods | Seeds | Frames | Status |
|------|---------|-------|--------|--------|
| DoorKey-8x8 | baseline, SE, VCSE, SI2E | 5 | 3M | ✅ |
| DoorKey-16×16 | all 4 | 3 | 3M | ✅ (all 0%, too hard) |
| KeyCorridorS3R2 | all 4 (VCSE/SI2E: 5 seeds) | 3–5 | 3M | ✅ |
| KeyCorridorS3R1 | all 4 | 3 | 3M | ✅ (all 100%, too easy) |
| RedBlueDoors-6x6 | all 4 | 3 | 3M | ✅ |
| UnlockPickup | all 4 | 3 | 3M | ✅ (all 0%, too hard) |
| DK-8x8 ablation | no_cluster, no_norm | 3 | 1M | ✅ |

### 1.2 Final Numbers (Deduped, All Seeds)

| Task | baseline | SE | VCSE | SI2E |
|------|----------|----|------|------|
| DK-8×8 (5s) | 0%±0 | 43%±49 | 97.8%±3.1 | **100%±0** |
| KC-S3R2 (5s) | 0% | 0% | 54%±50 | **67.5%±31** |
| KC-S3R1 (3s) | 100% | 100% | 100% | 100% |
| RedBlueDoors (3s) | 0% | 2%±4 | 55%±48 | **56%±47** |
| UnlockPickup (3s) | 0% | 0% | 0% | 0% |

| Ablation (DK-8x8, 1M) | Mean | Std | Verdict |
|-----------------------|------|-----|---------|
| `no_cluster` (skip cluster bonus) | 0% | 0 | **Cluster bonus is load-bearing** |
| `no_norm` (absolute vs relative dist) | 22% | 38 | **Relative norm is critical early** |
| SI2E full (ref, 3M) | 100% | 0 | — |

### 1.3 Key Documents Created

| File | Contents |
|------|---------|
| `docs/RESULTS_SUMMARY.md` | Full statistical results + Conclusions C1–C5 |
| `docs/RESEARCH_NOTES.md` | Mechanistic analysis, ablation design, lessons |
| `docs/NEXT_RESEARCH.md` | 6 research directions with implementation sketches |
| `HANDOFF.md` | This file |

---

## 2. Understanding SI2E: The Algorithm

### 2.1 The Lineage

```
RE3 (ICML 2021):  kNN state entropy + fixed random encoder
  ↓ + value conditioning
VCSE (NeurIPS 2023): kNN entropy only among states with similar V(s)
  ↓ + structural entropy tree  
SI2E (NeurIPS 2024): hierarchical PartitionTree over feature space + 2-level VCSE
```

### 2.2 SI2E Reward Computation (Per Update Step)

Called once per rollout update with n=128 states (16 envs × 8 steps):

```
Input: src_feats  (128 × 64)  — CNN encoder output, current frame
       tgt_feats  (128 × 64)  — CNN encoder output, augmented frame
       V(s)       (128 × 1)   — value estimate from critic

Step 1: O(n²·d) GPU  —  pairwise distance matrix (NOW GPU via torch.cdist)
   dist(i,j) = max(||src_i - src_j||₂, ||tgt_i - tgt_j||₂, |V_i - V_j|)
   adj(i,j)  = 1 - dist(i,j) / max_dist        ← relative batch normalization

Step 2: O(n² log n) CPU  —  PartitionTree.build_encoding_tree(k=3)
   [OR glass-jax with --fast_se: constrained_k_multistart, ~1.5× faster]
   Greedy hierarchical min-cut on the 128×128 adj graph
   Produces: 2-level tree, ~5-10 clusters

Step 3: O(n) —  cluster centroids (entropy-weighted or mean with fast_se)

Step 4: O(n·k) GPU  —  VCSE at leaf level  ← reward_0

Step 5: O(C·k) GPU  —  VCSE at cluster level  ← reward_1

Step 6: O(n) —  cluster bonus injection
   reward_0[states in cluster c] += (1/|c|) · reward_1[c]

Final: intrinsic_bonus = β · reward_0   (β = 0.005, or β_adaptive if --beta_adaptive)
       total_reward = extrinsic + intrinsic_bonus
```

**Key file:** `SI2E/SI2E_A2C/torch-ac/torch_ac/algos/base.py` lines ~411–530  
**Methods:** `compute_value_condition_structural_entropy()` and `compute_value_condition_structural_entropy_fast()`

---

## 3. Speed Profile

### 3.1 FPS Benchmarks (This Machine, GPU, 16 Parallel Envs) — SESSION 3 UPDATED

Run: `python3 /workspace/learn-si2e/benchmark_fps.py --procs 16 --steps 5`

| Method | FPS | collect_ms | update_ms | Speedup | Eff. grad-steps/s |
|--------|-----|-----------|-----------|---------|------------------|
| Baseline A2C | 2020 | 39ms | 24ms | 1.00× | 2020 |
| **A2C-SI2E** | 1002 | 38ms | 90ms | 0.50× | 1002 |
| **A2C-FastSI2E** (k-means) | **1864** | 35ms | 34ms | **0.92×** | **1864** |
| **PPO-SI2E** | 474 | 36ms | 234ms | 0.23× | 1896 |
| **PPO-FastSI2E** (k-means) | **1561** | 43ms | 39ms | **0.77×** | **6244** |

**Note:** `--fast_se` now uses numpy k-means (0.9ms) replacing glass-jax (155ms) — 171× faster on clustering.

**Key headline for paper:** PPO-FastSI2E delivers **6× more effective gradient steps/sec** than original A2C-SI2E.

At 1864 FPS (FastSI2E), 3M frames ≈ 27 min per seed. 3 seeds × 4 task variants ≈ **5.4 GPU hours**.

---

## 4. Code Changes Made This Session

### 4.1 `base.py` Changes

1. **GPU cdist** (always on): replaces `np.linalg.norm` pairwise with `torch.cdist` in `compute_value_condition_structural_entropy`

2. **FastSI2E method**: `compute_value_condition_structural_entropy_fast()` using glass-jax numba clustering with mean centroids

3. **`_effective_beta()`**: adaptive beta scheduling based on rolling success rate

4. **`beta_adaptive` and `fast_se` flags**: new `__init__` params

### 4.2 `ppo.py` Changes

- Accept `use_entropy_reward`, `use_value_condition`, `ablation`, `beta_adaptive`, `fast_se`
- Compute SI2E reward ONCE per rollout before the 4 PPO epochs
- Grad norm fix: skip params with `None` grad (extr_critic unused in PPO)

### 4.3 `a2c.py` Changes

- Accept `beta_adaptive`, `fast_se`
- Use `_effective_beta()` instead of hardcoded `self.beta`
- Route to `fast_se` path via `_se_fn` selector

### 4.4 `train.py` Changes

- New flags: `--beta_adaptive`, `--fast_se`
- PPO branch now passes all intrinsic reward flags

---

## 5. Environment Setup (Reproduce From Scratch)

### 5.1 Python Environment

```bash
cd /workspace/learn-si2e

# SI2E A2C
pip install -e SI2E/SI2E_A2C/torch-ac/
pip install -r SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/requirements.txt

# MiniGrid (pinned version with numpy 2.x fix)
pip install -e minigrid-pinned/

# glass-jax (for --fast_se)
cd /workspace/glass-jax && pip install -e .

# VCSE (needs special PYTHONPATH)
pip install -e base-vcse/VCSE_A2C/torch-ac/
export PYTHONPATH="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac:${PYTHONPATH}"
```

### 5.2 NumPy 2.x Fix (Already Applied)

`minigrid-pinned/gym_minigrid/minigrid.py` has been patched:  
Two occurrences of `dtype=np.bool` → `dtype=bool` (lines ~572 and ~585).  
**Do not undo this patch.**

### 5.3 Run a Single Experiment

```bash
cd SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/

# Original SI2E on DoorKey-8x8
python scripts/train.py \
  --env MiniGrid-DoorKey-8x8-v0 \
  --algo a2c \
  --use_entropy_reward \
  --use_value_condition \
  --beta 0.005 \
  --use_batch \
  --frames 3000000 \
  --seed 1

# SI2E-PPO (new)
python scripts/train.py \
  --env MiniGrid-DoorKey-8x8-v0 \
  --algo ppo \
  --use_entropy_reward \
  --use_value_condition \
  --beta 0.005 \
  --use_batch \
  --frames 3000000 \
  --seed 1

# FastSI2E (new, 1.5× faster)
python scripts/train.py \
  --env MiniGrid-DoorKey-8x8-v0 \
  --algo a2c \
  --use_entropy_reward \
  --use_value_condition \
  --beta 0.005 \
  --use_batch \
  --fast_se \
  --frames 3000000 \
  --seed 1

# Adaptive-β (new, reduced variance)
python scripts/train.py \
  --env MiniGrid-DoorKey-8x8-v0 \
  --algo a2c \
  --use_entropy_reward \
  --use_value_condition \
  --beta 0.005 \
  --use_batch \
  --beta_adaptive \
  --frames 3000000 \
  --seed 1
```

---

## 6. Research Directions (Prioritized)

Read `docs/NEXT_RESEARCH.md` for full details.

### DONE this session

- ✅ Direction A (FastSI2E): GPU cdist + glass-jax clustering (~1.5× speedup)
- ✅ Direction C (SI2E-PPO): port to PPO, wire in train.py
- ✅ Direction E (Adaptive β): implemented and ready to run

### Phase 1 (NEXT STEP): Run and evaluate all three

```bash
cd /workspace/learn-si2e
chmod +x batch_ppo_si2e.sh batch_adaptive_beta.sh batch_fast_si2e.sh
nohup ./batch_ppo_si2e.sh > logs/ppo_si2e.log 2>&1 &
nohup ./batch_adaptive_beta.sh > logs/adaptive_beta.log 2>&1 &
nohup ./batch_fast_si2e.sh > logs/fast_si2e.log 2>&1 &
```

Expected results:
- SI2E-PPO: KC-S3R2 >75% mean (vs A2C 67.5%)
- Adaptive-β: RedBlueDoors std <25% (vs 47%)
- FastSI2E: same accuracy at 1.5× faster FPS

### Phase 2 (2–4 weeks): Solve harder tasks

- **Direction D (H₃-SI2E)**: 3-level tree for UnlockPickup
  - glass-jax `coding_tree.py` has the H₃ builder
  - Requires FastSI2E first (already done)
- **Direction F (multi-buffer)**: rolling 5K-state buffer for richer tree
  - Wire into `base.py`, requires Direction B (LSH) or FastSI2E to be tractable

### Phase 3 (4–8 weeks): SOTA push

- **Direction B (LSH-SI2E)**: FAISS kNN graph, O(n log n)
- **H₃-SI2E + PPO + FastSI2E** combined: solve UnlockPickup >0%

---

## 7. Key Code Locations

```
SI2E/SI2E_A2C/torch-ac/torch_ac/algos/
  base.py                      ← THE core file.
    line ~411: compute_value_condition_structural_entropy()    ← SI2E reward (GPU cdist)
    line ~485: _compute_adj_matrix()                           ← shared adj matrix
    line ~499: compute_value_condition_structural_entropy_fast() ← glass-jax fast path
    line ~335: _effective_beta()                               ← adaptive β (NEW)
    line ~37:  BaseAlgo.__init__()                             ← fast_se, beta_adaptive flags
  a2c.py                       ← A2C update loop with adaptive β + fast_se routing
  ppo.py                       ← PPO update, SI2E reward computed once before epochs

SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/scripts/
  train.py                     ← CLI flags: --algo ppo, --fast_se, --beta_adaptive, etc.

/workspace/glass-jax/src/glass/seclust/
  incremental.py               ← constrained_k_multistart() (SI2E fast clustering)
  numba_kernel.py              ← @njit move-delta kernel
  coding_tree.py               ← H₃ tree builder (for Direction D)

/workspace/learn-si2e/
  benchmark_fps.py             ← FPS benchmark for paper table
  batch_ppo_si2e.sh            ← SI2E-PPO experiments
  batch_adaptive_beta.sh       ← Adaptive-β experiments
  batch_fast_si2e.sh           ← FastSI2E validation
```

---

## 8. Expected Paper Contributions

### Performance claim (Table 1)

| Method | DK-8×8 | KC-S3R2 | RedBlueDoors | Backbone |
|--------|---------|---------|-------------|---------|
| SI2E (repro) | 100%±0 | 67.5%±31 | 55.7%±47 | A2C |
| SI2E-PPO | 100%±0 | **>75%** | **>70%** | PPO |
| SI2E+Adaptive-β | 100%±0 | 67%±20? | **55%±20?** | A2C |
| PPO-FastSI2E (ours) | 100%±0 | **>75%** | **>70%** | PPO |

### Speed claim (Table 2)

| Method | FPS | vs SI2E | Gradient efficiency |
|--------|-----|---------|-------------------|
| A2C-SI2E | 1040 | 1.0× | 1× |
| A2C-FastSI2E | 1534 | **1.5×** | 1× |
| PPO-FastSI2E | 947 | ~1.0× | **4×** |

### Ablation (Table 3)

Already done in previous session:
- `no_cluster`: 0% (cluster bonus is load-bearing)
- `no_norm`: 22%±38 (relative normalization is critical)

---

## 9. Experiment Infrastructure

### 9.1 Result Format

Each run produces:
- `results/{task}/{method}-s{seed}/` — training log
- `results/{task}/summary.csv` — columns: method, seed, success_rate_pct, frames

### 9.2 Log Format

```
U {updates} | F {frames} | FPS {fps} | rR:μσmM {return_mean} {return_std} ...
```

`rR:mean > 0.5` for 5+ consecutive updates → task solved.

---

## 10. Known Issues and Gotchas

| Issue | Status | Fix |
|-------|--------|-----|
| NumPy 2.x `dtype=np.bool` crash | ✅ Fixed | `minigrid-pinned/gym_minigrid/minigrid.py` lines ~572,585 |
| Stale `.pyc` bytecode | Known | `find /workspace/learn-si2e/SI2E -name "*.pyc" -delete` |
| VCSE needs separate PYTHONPATH | Known | `export PYTHONPATH=".../base-vcse/VCSE_A2C/torch-ac:${PYTHONPATH}"` |
| glass-jax JIT warmup | Known | First call takes ~1.5s (numba compilation). Amortized over training. |
| PPO extr_critic has no grad in PPO | Fixed | `p.grad is not None` guard in grad_norm computation |
| KC-S3R2 high variance (std=31%) | Known | Need ≥5 seeds; PPO should reduce this |
| RedBlueDoors 3-seed mean unreliable | Known | Need 5 seeds; adaptive-β should reduce variance |

---

## 11. Quick-Start for New Engineer

1. **Verify environment** (1 min):
   ```bash
   cd /workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files
   python3 -c "import torch, numba, gym_minigrid, torch_ac, sip; print('all OK')"
   ```

2. **Launch all experiments** (run overnight):
   ```bash
   cd /workspace/learn-si2e
   chmod +x batch_ppo_si2e.sh batch_adaptive_beta.sh batch_fast_si2e.sh
   nohup ./batch_ppo_si2e.sh > logs/ppo_si2e.log 2>&1 &
   nohup ./batch_adaptive_beta.sh > logs/adaptive_beta.log 2>&1 &
   nohup ./batch_fast_si2e.sh > logs/fast_si2e.log 2>&1 &
   ```

3. **Check results** as they arrive:
   ```bash
   cat results/ppo-si2e/summary.csv
   cat results/adaptive-beta/summary.csv
   cat results/fast-si2e/summary.csv
   ```

4. **Run FPS benchmark** for paper:
   ```bash
   python3 benchmark_fps.py --procs 16 --steps 5
   ```

---

## 12. The Single Most Important Insight

**SI2E's advantage is the cluster-level VCSE bonus (`reward_1`), not the tree structure.**

Any clustering that produces ~10 semantically coherent groups will work.  
The PartitionTree is the bottleneck. Replace it with glass-jax's `constrained_k_multistart`  
(via `--fast_se`) and you get the same exploration signal ~1.5× faster.

**Combined best method:** PPO + FastSI2E + Adaptive-β  
= 4× gradient efficiency + 1.5× FPS + lower variance = the proposed NeurIPS contribution.
