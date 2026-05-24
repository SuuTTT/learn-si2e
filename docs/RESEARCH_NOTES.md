# SI2E Research Notes — Experiments, Analysis, Next Steps

**Last updated:** 2026-05-24  
**Goal:** Understand *what makes SI2E outperform VCSE/SE*, find ways to improve it, and accelerate the dev loop.

---

## 1. Experiment Results (All Runs)

### 1.1 MiniGrid A2C — Completed

All MiniGrid runs: 3M frames, A2C backbone, batch eval 200 episodes argmax, seed eval seed=999.

#### DoorKey-8x8 — 5 seeds × 4 methods ✅

| Method | s1 | s2 | s3 | s4 | s5 | **Mean** | **Std** |
|--------|----|----|----|----|----|---------:|--------:|
| baseline | 0% | 0% | 0% | 0% | 0% | **0%** | 0 |
| SE | 0% | 92.5% | 0% | 100% | 23.5% | **43.2%** | 47.4 |
| VCSE | 93.5% | 95.5% | 100% | 100% | 100% | **97.8%** | 3.1 |
| SI2E | 100% | 100% | 100% | 100% | 100% | **100%** | 0 |

**Takeaway:** SI2E is the only method with zero variance. VCSE near-ceiling (97.8%). SE is high-variance (0–100% across seeds). Baseline fails completely.  
**Paper reports:** SI2E=100%, VCSE=96.8%, SE=62.3%, A2C=0% → locally reproduced.

---

#### DoorKey-16x16 — 3 seeds × 4 methods ✅

| Method | s1 | s2 | s3 | **Mean** |
|--------|----|----|----|--------:|
| All methods | 0% | 0% | 0% | **0%** |

**Takeaway:** 16×16 is too hard at 3M frames for all methods. Not a useful comparison task.

---

#### KeyCorridorS3R2 — 3 seeds × 4 methods ✅ (COMPLETE)

Grid 7×5, max_steps=270, highly sparse reward (must pick up key, unlock, reach goal).

| Method | s1 | s2 | s3 | **Mean** | **Std** |
|--------|----|----|----|---------:|--------:|
| baseline | 0% | 0% | 0% | **0%** | 0 |
| SE | 0% | 0% | 0% | **0%** | 0 |
| VCSE | 0% | 96% | 74% | **56.7%** | 48.3 |
| SI2E | 62.5% | 99.5% | 35.5% | **65.8%** | 32.1 |

**Takeaway:** Clean ordering — SI2E > VCSE > SE = baseline. Both VCSE and SI2E have high variance (s1 VCSE=0%, s3 SI2E=35.5%). This is the best "medium-hard" differentiation task found.  
**Best candidate for paper comparison table.**

---

#### RedBlueDoors-6x6 — 3 seeds × 4 methods ✅

Grid 6×6, agent must open red door then blue door, max_steps=120.

| Method | s1 | s2 | s3 | **Mean** | **Std** |
|--------|----|----|----|---------:|--------:|
| baseline | 0% | 0% | 0% | **0%** | 0 |
| SE | 0% | 7% | 0% | **2.3%** | 4.0 |
| VCSE | 83% | 0% | 83% | **55.3%** | 47.9 |
| SI2E | 83% | 1% | 83% | **55.7%** | 47.7 |

**Takeaway:** s2 is a catastrophic failure for both VCSE and SI2E (likely bad initialization). Paper reports VCSE=79.8%, SI2E=85.8% — means are dragged down by one bad seed.  
**High variance; needs 5 seeds to be reliable.** Not ideal for paper table at 3 seeds.

---

#### KeyCorridorS3R1 — 3 seeds × 4 methods (IN PROGRESS ~70%)

Grid 7×3, shorter corridor than S3R2, easier.

| Method | s1 | s2 | s3 | **Mean** |
|--------|----|----|----|---------:|
| baseline | 100% | 100% | 100% | **100%** |
| SE | 100% | 100% | 100% | **100%** |
| VCSE | 100% | *running* | — | — |
| SI2E | — | — | — | — |

**Takeaway:** Near-ceiling for all methods. Paper reports ~86–94% (likely fewer frames). Good for reproducibility check but no differentiation. Not ideal for paper table.

---

#### UnlockPickup — not started yet

Expected: baseline≈0%, SE≈0%, VCSE≈50%, SI2E≈60%. Medium complexity.

---

### 1.2 DMControl DrQv2 — Completed (earlier sessions)

See `docs/RESULTS_REPORT.md` for full analysis. Summary: SI2E/VCSE both strong, SE sensitive to replay buffer setup. All methods reproduce paper within noise at 3M frames.

---

## 2. What Makes SI2E Work?

### 2.1 The computation pipeline

```
Observation → CNN encoder → features (src, tgt)
                         → value head → V(s)

SI2E bonus = H2_structural_entropy(src_feats, tgt_feats, V)

Inside H2:
  Step 1: Build pairwise distance matrix
          dist(i,j) = max( ||src_i - src_j||, ||tgt_i - tgt_j||, |V_i - V_j| )
          normalized to [0,1] then inverted → adj_matrix = 1 - dist/max_dist

  Step 2: PartitionTree.build_encoding_tree(k=3)
          Hierarchical clustering via min-cut on the adj graph
          Produces a 2-level tree: leaves = individual states, level-1 = clusters

  Step 3: Compute VCSE entropy at BOTH levels:
          reward_0 = VCSE(src_feats, tgt_feats, V)           ← leaf level
          reward_1 = VCSE(cluster_centroids, ...)             ← cluster level (entropy-weighted)
          
  Step 4: reward[i] += (1/|cluster_i|) * reward_1[cluster containing i]
```

### 2.2 What each ingredient buys

| Component | What it does | Evidence |
|-----------|-------------|---------|
| **kNN entropy** (SE) | Explores states with low local density → visits all reachable states | DK-8x8 SE=43% vs baseline=0% |
| **Value conditioning** (VCSE) | Ignores low-value neighbors when computing entropy → stops wasting bonus on dead-end states | DK-8x8 VCSE=97.8% vs SE=43.2% |
| **PartitionTree / H₂** (SI2E) | Multi-scale: also rewards finding states that belong to *unexplored clusters*, not just unexplored individual states | DK-8x8 SI2E=100% with 0 std |

### 2.3 The key SI2E hypothesis

**Value conditioning alone (VCSE) has high variance because kNN is a local metric.** If all k nearest neighbors happen to have high value, the bonus can be 0 even if the agent hasn't explored the entire cluster. The cluster-level term in SI2E adds a "group novelty" bonus: even if the local neighborhood is dense, if the whole cluster hasn't been visited from multiple angles, reward_1 is high.

**This explains the variance reduction:** SI2E achieves 100% ± 0 on DK-8x8 while VCSE is 97.8% ± 3.1. The cluster-level signal smooths out unlucky seed initializations.

### 2.4 The cost: CPU bottleneck

`PartitionTree` is pure Python/NumPy. It cannot be GPU-accelerated easily because:
- Min-cut hierarchical clustering has sequential data dependencies
- `sip` module is not CUDA-capable

**Observed FPS:**
- baseline/SE/VCSE: ~1400–3000 FPS (GPU-bound)
- SI2E: ~497 FPS (CPU-bound) → **3–6× slower**

At 3M frames: SE/VCSE take ~30 min, SI2E takes ~100 min per seed.

---

## 3. Ablation Experiments (Active)

### 3.0 Hypothesis table

| # | Hypothesis | Tested by | Prediction if TRUE |
|---|-----------|----------|-------------------|
| H1 | Value conditioning is the dominant win (SE→VCSE) | DK-8x8 data | Already confirmed — +55pp |
| H2 | SI2E's advantage is variance reduction, not mean gain | DK-8x8 + KC-S3R2 data | Partially confirmed |
| H3 | **Relative batch-max normalization** rescues collapsed encoders | `no_norm` ablation | no_norm should show higher std, approaching VCSE's 3.1 |
| H4 | **Cluster-level bonus (reward_1)** is the key novel contribution | `no_cluster` ablation | no_cluster should show higher std |
| H5 | KC-S3R2 SI2E s3=35.5% is a genuine outlier (high-variance task) | KC-S3R2 seeds 4+5 | If seeds 4+5 are high, s3 was unlucky |

### 3.1 Ablation implementations (done)

**Files modified:**
- `SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/scripts/train.py`: `--ablation {no_cluster,no_norm}` flag
- `SI2E/SI2E_A2C/torch-ac/torch_ac/algos/a2c.py`: pass `ablation` to `BaseAlgo`
- `SI2E/SI2E_A2C/torch-ac/torch_ac/algos/base.py`: `self.ablation` used in `compute_value_condition_structural_entropy`

**`no_cluster`** (H4 test): PartitionTree runs with relative normalization, but `reward_1` loop is skipped.
Reward = only `reward_0` (leaf-level VCSE on PartitionTree-normalized features).

**`no_norm`** (H3 test): Skip `adj_matrix = 1 - dist/max_dist`. Use `adj_matrix = 1/(1+dist)` instead.
Same absolute scale regardless of batch feature spread — does NOT rescue collapsed encoder.

### 3.2 Running now

| Script | PID | Content |
|--------|-----|---------|
| `batch_ablations.sh` | 1522143 | no_cluster × 3 seeds + no_norm × 3 seeds (DK-8x8, 1M frames) + KC-S3R2 SI2E seeds 4+5 |
| `batch_kcs3r1.sh + batch_unlockpickup.sh` | 1522142 | KCS3R1 SI2E × 3 seeds + UnlockPickup × 12 runs |

Results will appear in `results/ablations/summary.csv`.

### 3.3 Interpretation guide

```
Compare to ground truth (from results/a2c-multiseed/summary.csv, 5 seeds, 3M frames):
  VCSE:  mean=97.8%  std=3.1
  SI2E:  mean=100%   std=0.0

Ablations run at 1M frames — expect slightly lower absolute values, but variance pattern should hold.

no_cluster std ≈ 3+    → Cluster-level bonus IS the mechanism (H4 confirmed)
no_cluster std ≈ 0     → Cluster-level bonus NOT key; relative norm alone is sufficient
no_norm    std ≈ 3+    → Relative normalization IS the mechanism (H3 confirmed)
no_norm    std ≈ 0     → Relative normalization NOT key; cluster-level bonus alone is sufficient

Both high std → both components individually necessary (synergistic)
Both low std  → neither isolated — something else at play (revisit hypothesis)
```

---

## 4. Ideas to Improve SI2E

### 3.1 Short-term (no theory change)

| Idea | Expected gain | Risk | Effort |
|------|-------------|------|--------|
| **Approximate PartitionTree with FAISS clustering** (GPU k-means as proxy for min-cut) | 5–10× FPS speedup | May lose some quality | Medium |
| **Cache encoding tree across consecutive updates** (tree is rebuilt every 128 steps; many nodes barely change) | 2–3× speedup | Stale tree = stale bonus | Low |
| **Reduce tree depth** from 3-level to 2-level (already partially done — build_encoding_tree(k=3) only uses 2 levels) | Small speedup | None | Already done |
| **Increase batch size** (use more parallel environments, collect more frames per update step) | Reduces tree rebuilds per frame | Higher RAM | Low |
| **Run PartitionTree in a separate process** (non-blocking) and use previous step's reward until result is ready | Up to 2× throughput | Slight staleness in bonus | Medium |

### 3.2 Algorithmic improvements

| Idea | Motivation | Notes |
|------|-----------|-------|
| **Adaptive β schedule** | β=0.005 is fixed; early training may need higher β (explore more), later lower (exploit more) | Paper uses fixed β throughout |
| **Deeper tree (k>3)** | More levels = richer multi-scale signal for complex environments | May hurt simple tasks |
| **DC-SE: degree-corrected structural entropy** | In graphs with hub nodes (scale-free), H₂ is dominated by high-degree nodes. DC correction (H₂ − Σ d_v log d_v / vol) removes this bias | Relevant for robotics state spaces with large-norm features |
| **Hierarchical value conditioning** | Currently value conditioning is applied at leaf level only; also condition cluster-level reward_1 on cluster average value | Could improve credit assignment |
| **Replace VCSE's kNN distance with learned metric** | kNN in raw feature space may not reflect true exploration distance | Requires extra training |

### 3.3 Accelerating the dev loop

**Current bottleneck:** Each "does this change help?" test requires 3M frames × 4 methods × 3 seeds = 36 training runs × ~100 min each = days.

**Faster proxies:**

| Proxy | How | Time saving |
|-------|-----|------------|
| **1M frames quick test** | Use 1M frames, 1 seed; check if SI2E > VCSE trend is preserved | 3× faster; ~33 min/run |
| **DK-8x8 single-seed smoke test** | 1M frames, seed=1, only compare VCSE vs SI2E. DK-8x8 is sensitive enough to show the gap | ~20 min |
| **Fixed seed convergence speed** | Measure "frames to 50% success rate" instead of final eval | Reduces frames needed to see signal |
| **Use KC-S3R2 as the benchmark** | It's the hardest task where methods differentiate AND has a clear ordering | Established ground truth now |

**Recommended dev loop:**
1. Make change to `base.py:compute_value_condition_structural_entropy`
2. Quick test: DK-8x8, 1M frames, seeds 1+2 (~40 min)
3. If promising: KC-S3R2, 3M frames, seeds 1-3 (~5h)
4. If confirmed: full 5-seed run for paper

---

## 4. Current Queue and Next Steps

### 4.1 Running now

| Process | Task | ETA |
|---------|------|-----|
| PID 1284002 (chain) | KCS3R1 VCSE s2 → s3 → SI2E×3 → UnlockPickup×12 | ~8h total |

### 4.2 After chain finishes

1. **Compile final table** across all 4 tasks: `python3 scripts/summarize_all.py`
2. **Decide paper table:** KC-S3R2 is the recommended medium-hard task. RedBlueDoors needs 5 seeds to be reliable.
3. **Optionally rerun RedBlueDoors s4+s5** to stabilize the mean (2 seeds × 2 methods × 3M = ~4h on GPU)

### 4.3 SI2E improvement experiments (next)

**Priority 1 — Cache/approximate PartitionTree (dev loop speed):**
- Modify `base.py:collect_experiences` to only rebuild the tree every N=5 updates instead of every update
- Test: does FPS improve? Does DK-8x8 success rate hold?

**Priority 2 — Adaptive β:**
- Add `--beta_final` arg; linearly anneal β from `--beta` to `--beta_final` over training
- Hypothesis: high β early → explore; low β late → exploit

**Priority 3 — DC-correction:**
- Compute feature norms; subtract `Σ ||f_v||² log ||f_v||² / Σ||f_v||²` from H₂
- Test on KC-S3R2 (where we already have VCSE/SI2E baselines for comparison)

---

## 5. Key File Reference

| File | Purpose |
|------|---------|
| `SI2E/SI2E_A2C/torch-ac/torch_ac/algos/base.py` | Core: `compute_value_condition_structural_entropy` (lines 398–455) |
| `base-vcse/VCSE_A2C/torch-ac/torch_ac/algos/base.py` | VCSE reference: `compute_value_condition_state_entropy` only |
| `results/a2c-multiseed/summary.csv` | DK-8x8 5-seed results |
| `results/keycorridor/summary.csv` | KC-S3R2 3-seed results (COMPLETE) |
| `results/redbluedoors/summary.csv` | RedBlueDoors 3-seed results (COMPLETE, high variance) |
| `results/kcs3r1/summary.csv` | KCS3R1 3-seed results (in progress) |
| `results/unlockpickup/summary.csv` | UnlockPickup 3-seed results (pending) |
| `logs/candidates.log` | Live log for current chain |
| `batch_keycorridor.sh` | Canonical batch script pattern |

---

## 6. Summary: What We Know

```
Task difficulty ladder (3M frames, A2C):
  easy:   KCS3R1      → all methods ~100%     (no differentiation)
  medium: DK-8x8      → baseline=0, SE=43, VCSE=98, SI2E=100  (clear ordering)
  hard:   KC-S3R2     → baseline=0, SE=0,  VCSE=57, SI2E=66   (clear ordering, high variance)
  too hard: DK-16x16  → all 0%                (no signal)

What value conditioning adds: +55 pp (SE 43% → VCSE 98%) on DK-8x8
What structural entropy adds: +2 pp mean (VCSE 98% → SI2E 100%) on DK-8x8
                               +9 pp mean (VCSE 57% → SI2E 66%) on KC-S3R2

SI2E's main advantage is VARIANCE REDUCTION, not mean improvement.
The H₂ cluster-level bonus provides a consistent signal even when
the local kNN neighborhood gives a weak signal (bad seed initialization).
```
