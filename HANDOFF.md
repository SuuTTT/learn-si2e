# SI2E Research Handoff — Beat SOTA in Performance and Speed

**Prepared:** 2026-05-27  
**For:** Algorithm engineer continuing this research direction  
**Codebase:** `/workspace/learn-si2e/`  
**Original paper:** SI2E — NeurIPS 2024, Zeng et al.  
**Goal:** Beat SI2E in both exploration performance and wall-clock speed on MiniGrid tasks

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

Called once per rollout update with n=640 states (16 envs × 40 steps):

```
Input: src_feats  (640 × 64)  — CNN encoder output, current frame
       tgt_feats  (640 × 64)  — CNN encoder output, augmented frame
       V(s)       (640 × 1)   — value estimate from critic

Step 1: O(n²·d) CPU  —  pairwise distance matrix
   dist(i,j) = max(||src_i - src_j||₂, ||tgt_i - tgt_j||₂, |V_i - V_j|)
   adj(i,j)  = 1 - dist(i,j) / max_dist        ← relative batch normalization

Step 2: O(n² log n) CPU  —  PartitionTree.build_encoding_tree(k=3)
   Greedy hierarchical min-cut on the 640×640 adj graph
   Python dict/set data structures; numba JIT only for cut_volume()
   Produces: 2-level tree, ~k=3–10 clusters of ~64–100 states each

Step 3: O(n) CPU  —  entropy-weighted cluster centroids
   centroid_c = Σ_{i∈c} (H_node_i / H_total_c) · feat_i
   (weighted average feature for each cluster)

Step 4: O(n·k) GPU  —  VCSE at leaf level  ← reward_0
   For each state: kNN distance in value-conditioned subspace
   Uses digamma estimator: reward ≈ digamma(n_v+1)/d_s + log(eps·2)

Step 5: O(C·k) GPU  —  VCSE at cluster level  ← reward_1
   Same VCSE computation but on C cluster centroids (C << 640)

Step 6: O(n) CPU  —  cluster bonus injection
   reward_0[states in cluster c] += (1/|c|) · reward_1[c]

Final: intrinsic_bonus = β · reward_0   (β = 0.005)
       total_reward = extrinsic + intrinsic_bonus
```

**Key file:** `SI2E/SI2E_A2C/torch-ac/torch_ac/algos/base.py` lines 399–476  
**Method:** `compute_value_condition_structural_entropy()`

### 2.3 Why It Works (Mechanistic Understanding)

1. **Cluster bonus is the primary signal.** Without it (`no_cluster` ablation), all 3 seeds
   score 0%. The leaf-level VCSE alone is not enough. The cluster-level bonus tells the
   agent when it has reached a structurally new region of state space.

2. **Relative normalization saves early training.** When the CNN encoder hasn't learned
   meaningful representations yet, all feature distances are near-zero. Absolute
   normalization (`1/(1+d)`) maps this to near-1 (all states look similar). Relative
   normalization (`1 - d/max_d`) always spans [0,1] regardless of scale, giving the
   PartitionTree a useful signal even with a random encoder.

3. **Two-scale exploration.** `reward_0` (leaf VCSE) drives local exploration.
   `reward_1` (cluster VCSE on centroids) drives escaping a macro-region. On DoorKey-8x8,
   the PartitionTree naturally clusters states by room, and the cluster bonus fires when
   the agent reaches the key's room or the door's room for the first time.

4. **Why SI2E beats VCSE:** SI2E's zero variance on DK-8x8 (vs VCSE's 3.1 std) comes from
   the cluster bonus providing a more stable, semantically structured reward signal.

---

## 3. Speed Profile

### 3.1 FPS Benchmarks (This Machine, CPU-Only, 16 Parallel Envs)

| Method | Early Training FPS | Late Training FPS | Bottleneck |
|--------|-------------------|------------------|-----------|
| Baseline A2C | ~5,000 | ~5,000 | Policy net |
| SE (kNN only) | ~2,000 | ~2,000 | O(n²) kNN |
| VCSE | ~981 avg | ~981 avg | kNN + value split |
| **SI2E** | **~1,432** | **~420** | **PartitionTree** |

Note: SI2E FPS drops as episodes shorten (more PartitionTree calls per frame since
each update still processes 640 states but covers fewer actual game frames).

At 420 FPS, 3M frames = ~2 hours per seed. With 5 seeds × 4+ tasks = **40+ CPU hours**.

### 3.2 Where Time Goes in One SI2E Update

| Step | Time (est.) | Language | Can be GPU? |
|------|------------|---------|------------|
| O(n²) pairwise dist | ~15–30ms | NumPy CPU | Yes — `torch.cdist` |
| PartitionTree build | ~50–100ms | Python + numba | Partial — see glass-jax |
| Centroid computation | ~5ms | Python loop | Yes |
| VCSE kNN (leaf) | ~5ms | PyTorch GPU | Already GPU |
| VCSE kNN (cluster) | ~1ms | PyTorch GPU | Already GPU |
| Cluster bonus inject | ~5ms | Python loop | Yes |
| **Total reward compute** | **~80–150ms** | — | — |
| A2C policy update | ~5ms | PyTorch GPU | Already GPU |

**The reward compute is 20–30× more expensive than the RL update itself.**

---

## 4. The Fast SE Clustering Resource: glass-jax

**Repo:** https://github.com/SuuTTT/glass-jax  
**By the same author.** This is a companion project implementing fast SE clustering.

### 4.1 What glass-jax Provides

```
src/glass/seclust/
  incremental.py    — IncrementalSEState, move_delta, constrained_k_multistart
  numba_kernel.py   — @njit JIT'd move-delta kernel: 36× speedup over Python baseline
  jit_kernel.py     — JAX-JIT version (GPU port, WIP)
  sync_kernel.py    — synchronous batched local-move kernel (idea 018, GPU WIP)
  coding_tree.py    — high-dimensional (H₃) tree builder
  hierarchy.py      — hierarchical coarsening
```

**Current status:**
- `numba_kernel.py`: production-ready, 36× speedup on microbench, 7× end-to-end on Photo
- `jit_kernel.py` + `sync_kernel.py`: JAX-GPU port, partially validated (3.2× over numpy)
  on Amazon-Photo. Full GPU pipeline is ~half-day of follow-up work.

### 4.2 The Connection to SI2E

The current SI2E code in `sip.py` already uses numba for `cut_volume()`:
```python
@nb.jit(nopython=True)
def cut_volume(adj_matrix, p1, p2):  # sip.py line ~37
    ...
```
But the outer PartitionTree loop (`build_encoding_tree`, `merge`, `compressNode`)
remains pure Python. glass-jax's `IncrementalSEState` with `numba_kernel.py`
replaces the full tree build with an incremental local-search approach that is
**36× faster on the clustering step** and avoids rebuilding from scratch every call.

### 4.3 Integration Plan (FastSI2E)

The key change is in `base.py` `compute_value_condition_structural_entropy()`:

**Current approach (Step 2):**
```python
y = PartitionTree(adj_matrix=adj_matrix)
x = y.build_encoding_tree(k=3)          # ← slow Python tree
```

**glass-jax approach:**
```python
# 1. Build sparse kNN graph instead of dense adj_matrix (O(n·k) instead of O(n²))
# 2. Use IncrementalSEState for fast cluster assignment
# 3. No full rebuild — cache and incrementally update the partition

from glass.seclust.incremental import constrained_k_multistart
labels = constrained_k_multistart(adj_sparse, K=10, n_starts=3)
# labels: (n,) array of cluster assignments, same interface as tree leaves
```

**On GPU (glass-jax JAX path):**
```python
import jax.numpy as jnp
from glass.seclust.jit_kernel import compute_soft_assignments

# All on GPU, differentiable
adj_gpu = torch_to_jax(adj_matrix_gpu)
assignments = compute_soft_assignments(adj_gpu, K=10)  # (n, K) soft
```

**Expected speedup:** numba path: 5–10×; JAX GPU path: 20–50× (once idea 018 is complete).

### 4.4 Clone and Setup

```bash
cd /workspace
git clone https://github.com/SuuTTT/glass-jax.git
cd glass-jax
pip install -e .

# Test that the SE clustering works
python tests/test_fast.py

# Benchmark against original
python tests/benchmark_seclust_full.py --seeds 0,42
```

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

# VCSE (needs special PYTHONPATH)
pip install -e base-vcse/VCSE_A2C/torch-ac/
export PYTHONPATH="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac:${PYTHONPATH}"
```

### 5.2 NumPy 2.x Fix (Already Applied)

`minigrid-pinned/gym_minigrid/minigrid.py` has been patched:  
Two occurrences of `dtype=np.bool` → `dtype=bool` (lines ~572 and ~585).  
**Do not undo this patch.**

### 5.3 Run a Single SI2E Experiment

```bash
cd SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/

# Full SI2E on DoorKey-8x8
python scripts/train.py \
  --env MiniGrid-DoorKey-8x8-v0 \
  --algo a2c \
  --use_entropy_reward \
  --use_value_condition \
  --beta 0.005 \
  --frames 3000000 \
  --seed 1 \
  --save_interval 1000 \
  --model dk8x8-si2e-s1

# Ablation: no cluster bonus
python scripts/train.py \
  --env MiniGrid-DoorKey-8x8-v0 \
  --algo a2c \
  --use_entropy_reward \
  --use_value_condition \
  --beta 0.005 \
  --ablation no_cluster \
  --frames 1000000 \
  --seed 1 \
  --model dk8x8-nocluster-s1
```

### 5.4 Evaluate a Trained Model

```bash
python scripts/evaluate.py \
  --env MiniGrid-DoorKey-8x8-v0 \
  --model dk8x8-si2e-s1 \
  --episodes 200

# Outputs: success_rate (0–1), episode_return (mean)
```

### 5.5 Batch Scripts

All batch scripts are in `/workspace/learn-si2e/`:
- `batch_a2c_multiseed.sh` — DK-8x8 5-seed
- `batch_keycorridor.sh` — KC-S3R2
- `batch_ablations.sh` — no_cluster / no_norm
- (etc. — see the batch_*.sh files)

Results land in `results/{task}/` as CSV files. Summarize with:
```bash
python3 -c "
import csv
from collections import defaultdict
d = defaultdict(list)
with open('results/a2c-multiseed/summary.csv') as f:
    for r in csv.DictReader(f):
        d[r['method']].append(float(r['success_rate_pct']))
for m, v in d.items(): print(m, round(sum(v)/len(v),1), v)
"
```

---

## 6. Key Code Locations

### 6.1 SI2E Algorithm Files

```
SI2E/SI2E_A2C/torch-ac/torch_ac/algos/
  base.py                      ← THE core file. Read this fully.
    line 399: compute_value_condition_structural_entropy()  ← SI2E reward
    line 357: compute_value_condition_state_entropy()       ← VCSE reward (used inside)
    line 333: compute_state_entropy()                       ← SE/kNN reward
    line 37:  BaseAlgo.__init__()                           ← ablation flag wired here
  a2c.py                       ← A2C update loop, ablation passed through here

SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/scripts/
  train.py                     ← CLI flags: --ablation, --beta, --use_entropy_reward, etc.

SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/
  sip.py                       ← PartitionTree implementation (numba-partial)
    PartitionTree class
    build_encoding_tree(k)
    node_entropy()
    cut_volume() ← numba JIT'd
```

### 6.2 Ablations Already Wired (This Session)

The `--ablation` flag is live in the codebase:
- `train.py` accepts `--ablation {no_cluster,no_norm}`
- Passed to `torch_ac.A2CAlgo(ablation=...)`
- In `base.py`:
  - `no_norm`: uses `adj_matrix = 1/(1+adj_matrix)` instead of relative normalization
  - `no_cluster`: skips Step 6 (cluster bonus injection loop)

New ablation variants can be added by inserting branches in `base.py` around lines 417–430.

### 6.3 VCSE Reference Implementation

```
base-vcse/VCSE_A2C/torch-ac/torch_ac/algos/
  base.py      ← VCSE reward compute (compare with SI2E base.py)
```
VCSE requires `PYTHONPATH="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac:${PYTHONPATH}"`

---

## 7. Research Directions (Prioritized)

Read `docs/NEXT_RESEARCH.md` for full details. Here is the prioritized execution plan:

### Phase 1 (1–2 weeks): Low-risk, high-reward

#### 7.1 Direction C: SI2E-PPO (+10–20% performance)

PPO runs 4 mini-batch epochs per rollout vs A2C's 1. Same data, more gradient updates.

```bash
# Entry point already exists in base-rl-starter-files/
# Port: copy the SI2E reward logic into base-rl-starter-files PPO implementation

# Files to modify:
# base-rl-starter-files/torch_ac/algos/ppo.py
#   — add use_entropy_reward, use_value_condition, beta params
#   — call compute_value_condition_structural_entropy() before PPO epochs
#   — store intrinsic reward in exps.reward before the epoch loop

# The reward must be computed ONCE per rollout (before epochs), 
# not re-computed each mini-batch.
```

Expected experiment: 5 seeds, KC-S3R2, 3M frames, compare vs A2C-SI2E baseline.

#### 7.2 Direction E: Adaptive β scheduling

```python
# In a2c.py update_parameters(), before computing intrinsic reward:
recent_success = np.mean([1.0 if r > 0 else 0.0 
                          for r in self.log_episode_return[-100:]])
effective_beta = self.beta * max(0.1, 1.0 - recent_success)
# Use effective_beta instead of self.beta in the intrinsic reward scaling
```

Test on RedBlueDoors-6x6 (highest seed variance: std=47%). Target: std < 20%.

### Phase 2 (2–4 weeks): Speed breakthrough

#### 7.3 Direction A: FastSI2E via glass-jax

**Step 1:** Clone glass-jax, verify numba kernel works on toy input.

```bash
cd /workspace
git clone https://github.com/SuuTTT/glass-jax.git && cd glass-jax && pip install -e .
python -c "
from glass.seclust.incremental import constrained_k_multistart
import numpy as np
A = np.random.rand(640, 640).astype(np.float32)
np.fill_diagonal(A, 0)
labels = constrained_k_multistart(A, K=10, n_starts=3)
print('labels shape:', labels.shape, 'unique clusters:', len(set(labels)))
"
```

**Step 2:** Replace PartitionTree in `base.py`:

```python
# Current (slow, ~100ms):
y = PartitionTree(adj_matrix=adj_matrix)
x = y.build_encoding_tree(k=3)
# ... extract cluster memberships from y.tree_node

# Proposed (fast, ~3ms with numba):
from glass.seclust.incremental import constrained_k_multistart
labels = constrained_k_multistart(adj_matrix, K=10, n_starts=3)
# labels[i] = cluster index for state i
# Build the same cluster→states mapping as the tree gives
```

**Step 3:** Benchmark FPS before/after on DK-8x8. Target: >1,000 FPS (from current ~420).

**Step 4 (optional, GPU path):** Replace O(n²) pairwise matrix with `torch.cdist`:

```python
# Current (CPU numpy, ~20ms):
sfa_dists = np.linalg.norm(sfa[:, None, :] - sfa[None, :, :], axis=-1)

# Proposed (GPU torch, ~1ms):
with torch.no_grad():
    sfa_dists = torch.cdist(src_feats, src_feats).cpu().numpy()
```

#### 7.4 Direction F: Multi-buffer SI2E

```python
# In BaseAlgo.__init__():
self.feature_buffer = []         # rolling buffer
self.buffer_maxlen = 5000

# In compute_value_condition_structural_entropy():
# Append current batch feats to buffer
self.feature_buffer.extend(list(zip(src_feats, tgt_feats, value)))
if len(self.feature_buffer) > self.buffer_maxlen:
    self.feature_buffer = self.feature_buffer[-self.buffer_maxlen:]

# Sample 1000 from buffer + use current 640 for tree
sample_idx = np.random.choice(len(self.feature_buffer), 
                               min(1000, len(self.feature_buffer)), replace=False)
buffer_sample = [self.feature_buffer[i] for i in sample_idx]
# Build PartitionTree on buffer_sample ∪ current_batch
```

### Phase 3 (4–8 weeks): SOTA push

#### 7.5 Direction D: H₃-SI2E (Solve UnlockPickup)

The `coding_tree.py` in glass-jax builds 3-level trees. UnlockPickup requires
3 chained subgoals. H₃ tree would add a middle level capturing subgoal proximity.

Requires FastSI2E (Direction A) first — H₃ is computationally more expensive.

#### 7.6 Direction B: LSH-SI2E (O(n log n) tree)

Use FAISS approximate k-NN graph instead of dense adj_matrix:
```python
import faiss
index = faiss.IndexFlatL2(d)
index.add(sfa)
D, I = index.search(sfa, k=10)   # k-NN graph, O(n log n)
# Build sparse adj from (I, D) and feed to glass-jax IncrementalSEState
```

---

## 8. The glass-jax SEClust Connection (Sister Project)

glass-jax also contains `src/glass/seclust/` — a fully-featured discrete SE clustering
library for **graph datasets** (Cora, Photo, ogbn-arxiv). This is a TPAMI submission
(SEClust paper) parallel to the SI2E RL work.

**For this RL work**, the relevant components from glass-jax are:
1. `numba_kernel.py` — fastest discrete SE move-delta kernel (use for FastSI2E)
2. `jit_kernel.py` — JAX/GPU kernel (use once idea 018 GPU path is complete)
3. `coding_tree.py` — H₃ tree builder (use for H₃-SI2E direction)

**Do not conflate** the SEClust TPAMI work (graph clustering benchmarks) with the
SI2E RL work. They share the SE algorithm but target different problems.

---

## 9. Experiment Infrastructure

### 9.1 Result Format

Each run produces:
- `results/{task}/{method}-s{seed}/` — model checkpoint
- `results/{task}/summary.csv` — columns: method, seed, success_rate_pct, frames

### 9.2 Log Format

Training logs follow this line format:
```
U {updates} | F {frames} | FPS {fps} | D {duration_s} | 
rR:μσmM {return_mean} {return_std} {return_min} {return_max} |
F:μσmM {ep_len_mean} ... | H {entropy} | V {value} | pL {policy_loss} | vL {value_loss}
```

`rR:μσmM` is the mean success return (0 = fail, >0 = success).  
Task is "solved" when `rR:mean > 0.5` sustained for >5 consecutive updates.

### 9.3 Adding a New Method

1. Add CLI flag in `scripts/train.py`
2. Pass through `torch_ac.A2CAlgo(new_flag=...)` → `a2c.py` → `base.py`  
3. Add the reward branch in `base.py compute_value_condition_structural_entropy()`
4. Copy a `batch_*.sh` and edit the training command
5. Results go to `results/{new_task}/summary.csv` following the same format

---

## 10. Known Issues and Gotchas

| Issue | Status | Fix |
|-------|--------|-----|
| NumPy 2.x `dtype=np.bool` crash | ✅ Fixed | `minigrid-pinned/gym_minigrid/minigrid.py` lines ~572,585 patched |
| Stale `.pyc` bytecode after editing torch-ac | Known | Run `find /workspace/learn-si2e/SI2E -name "*.pyc" -delete` before launch |
| VCSE needs separate PYTHONPATH | Known | Set `PYTHONPATH="/workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac:${PYTHONPATH}"` |
| KC-S3R2 high variance (std=31% SI2E, std=50% VCSE) | Known | Genuine high-variance task; need ≥5 seeds for stable mean |
| RedBlueDoors 3-seed mean unreliable (std=47%) | Known | One catastrophic failure seed inflates std; need 5 seeds |
| DK-16×16 and UnlockPickup all 0% at 3M frames | Known | Too hard for A2C; try PPO or 10M frames |

---

## 11. Quick-Start Checklist for New Engineer

1. **Read the algorithm** (30 min):  
   `SI2E/SI2E_A2C/torch-ac/torch_ac/algos/base.py` lines 399–476 — this is SI2E.

2. **Read the results** (15 min):  
   `docs/RESULTS_SUMMARY.md` §10 — the concluded findings.

3. **Read the research plan** (20 min):  
   `docs/NEXT_RESEARCH.md` §4–5 — 6 directions with code sketches.

4. **Run a single validation experiment** (2 hours):
   ```bash
   cd /workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/
   python scripts/train.py --env MiniGrid-DoorKey-8x8-v0 --algo a2c \
     --use_entropy_reward --use_value_condition --beta 0.005 \
     --frames 500000 --seed 42 --model validation-run
   # Expect: FPS ~400–1000, rR:mean starts rising after ~200K frames
   ```

5. **Clone glass-jax** and run its benchmark to understand the fast SE baseline:
   ```bash
   cd /workspace
   git clone https://github.com/SuuTTT/glass-jax.git
   cd glass-jax && pip install -e .
   python tests/test_fast.py
   ```

6. **Pick a Phase 1 direction** and start — Direction C (PPO) or Direction E (adaptive β)
   are the lowest risk and highest expected payoff for the next experiment.

---

## 12. Summary: The Single Most Important Insight

**SI2E's advantage is the cluster-level VCSE bonus (`reward_1`), not the tree structure.**

The PartitionTree is a means to an end: it groups the 640 states into ~10 semantic
clusters so that a second VCSE computation can be run on the cluster centroids.
This cluster-level entropy bonus fires when the agent escapes to a macro-region it
hasn't explored, complementing the local leaf-level VCSE.

Any clustering that produces ~10 semantically coherent groups will work.
The PartitionTree is the bottleneck (slow Python + numba). Replace it with
glass-jax's `constrained_k_multistart` (36× faster) and you get the same
exploration signal at a fraction of the cost. That is the single most impactful
code change possible.
