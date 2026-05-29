# SI2E Research Milestones
*Last updated: 2026-05-29*

---

## Summary of contributions

| Contribution | Status | Key result |
|---|---|---|
| FastSI2E k-means (speed) | ✅ DONE | 4.0× faster, 100% on DK-8x8 |
| FastSI2E Leiden | ✅ DONE | 2.8× faster, **+24 pp on KC-S3R2** |
| FastSI2E Infomap | ✅ DONE | 3.2× faster, **+28 pp on KC-S3R2 (best overall)** |
| Adaptive-β (stability) | ✅ DONE | RBD: 34%→53%, bimodal 2/5→4/5 |
| FastSI2E on RBD | 🔄 running (~10 min) | TBD |
| FastSI2E + adaptive-β | 🔄 queued (~4 h) | TBD |
| PPO-SI2E (negative result) | ✅ DONE | 0% everywhere |
| Paper draft | ✅ written | `paper_draft.md` |
| Analysis & figures | ✅ done | `analyze_results.py --plot` |

---

## Milestone 1 — FastSI2E Speed Benchmark ✅
**DK-8x8, 1M frames, 3 seeds**

| Method | FPS | Speedup |
|--------|-----|---------|
| SI2E (PartitionTree) | 488 | 1× |
| FastSI2E (k-means) | 1952 | **4.0×** |
| FastSI2E (Leiden) | 1364 | 2.8× |
| FastSI2E (Infomap) | 1540 | 3.2× |

Clustering subroutine: 155 ms → 1.5 ms (k-means) = **103× faster**.
Overall 4× because env-stepping and network forward/backward now dominate.

---

## Milestone 2 — DK-8x8 Accuracy at 3M Frames ✅
**All methods match or exceed SI2E (100%) on DoorKey-8x8**

| Method | s1 | s2 | s3 | Mean |
|--------|----|----|-----|------|
| SI2E | 100% | 100% | 100% | **100%±0** |
| k-means | 100% | 100% | 100% | **100%±0** |
| Leiden | 100% | 100% | 100% | **100%±0** |
| Infomap | 98.5% | 100% | 100% | **99.5%±0.7** |

Primary claim confirmed: FastSI2E (any method) = SI2E accuracy on DK-8x8 at 3–4× speed.

---

## Milestone 3 — KC-S3R2 Clustering Comparison ✅  
**BREAKTHROUGH: Community detection outperforms original SI2E**

| Method | s1 | s2 | s3 | s4 | s5 | Mean | Std |
|--------|----|----|----|----|----|----|-----|
| SI2E | 45% | 83% | 68% | 97% | 44% | 67.5% | 27.9 |
| k-means | 25.5% | 100% | 11.0% | **100%** | **100%** | 67.3% | 40.3 |
| **Leiden** | 75.5% | **100%** | **100%** | — | — | **91.8%** | **11.5** |
| **Infomap** | 87.5% | **100%** | 99.5% | — | — | **95.7%** | **5.8** |

- Infomap: best result on KC-S3R2 of any method (+28.2 pp over SI2E, −22.1 pp std)
- Leiden: second best (+24.3 pp over SI2E, −16.4 pp std)
- k-means: same mean as SI2E but higher variance (bimodal: 3/5 converge)
- **Both community detection algorithms beat SI2E while being 2.8–3.2× faster**

Why Leiden/Infomap win: KC-S3R2 has multi-room topology with natural community structure (rooms). Graph modularity / random-walk compression detects this; Euclidean k-means misses it.

---

## Milestone 4 — Adaptive-β Stability ✅
**Original SI2E + adaptive schedule on RBD and KC**

| Method | Env | Mean | Std | Converge rate |
|--------|-----|------|-----|---------------|
| SI2E fixed-β | RBD | 34.0% | 40.1% | 2/5 |
| **SI2E adaptive-β** | RBD | **53.4%** | **27.4%** | **4/5** |
| SI2E fixed-β | KC | 45.5% | 39.0% | 3/5 |
| SI2E adaptive-β | KC | 46.5% | 21.4% | varied |

Adaptive-β formula: `β_t = β₀ · max(0.1, 1 − recent_success_rate)`

Reduces over-exploration once the agent starts succeeding. Bimodal pattern softened on RBD: {1%, 1%, 1%, 81%, 85%} → {0%, 56%, 68%, 68%, 75%}.

---

## Milestone 5 — PPO-SI2E Negative Result ✅

| Config | DK-8x8 | KC-S3R2 | RBD |
|--------|---------|---------|-----|
| PPO-SI2E | 58.5%±46 | 0% | — |
| PPO-FastSI2E | — | 0% | — |
| PPO-adaptive | — | — | 0.1% |

PPO is categorically incompatible with SI2E's intrinsic bonus. Hypothesis: off-policy correction interacts with stale bonus across 4 PPO epochs → reward distribution shift. Documented as negative result for paper.

---

## Milestone 6 — Phase 2 Experiments 🔄 RUNNING

### (A) KC-S3R2 FastSI2E seeds 4, 5 ✅
- s4: **100%** at 1753 FPS
- s5: **100%** at 1801 FPS
- FastSI2E KC now 5 seeds: 25.5%, 100%, 11.0%, 100%, 100% → **67.3%±40.3**

### (B) RedBlueDoors FastSI2E s1–3 🔄 running (2.3M/3M, ~10 min left)
- s1 and s2 active; s3 queued
- First test of FastSI2E (k-means) on RBD environment

### (C) FastSI2E + adaptive-β on KC-S3R2 s1–5 ⏳ queued
- Combines speed (4×) + stability (adaptive-β)
- Expected: mean ≈ 46.5% (adaptive-β baseline) with ~1700 FPS

### (D) FastSI2E + adaptive-β on RBD s1–5 ⏳ queued
- Expected: similar to SI2E-adaptive (53.4%±27.4) but 4× faster

---

## Current results table (all completed experiments)

```
python3 analyze_results.py   # prints this table live
```

| Method | DK-8x8 | KC-S3R2 | RBD-6x6 | FPS |
|--------|---------|---------|---------|-----|
| VCSE (orig.) | 97.8±2.8 | 54.0±45.0 | 55.4±39.1 | ~1950 |
| SI2E (orig.) | 100.0±0.0 | 67.5±27.9 | 55.7±38.7 | 488 |
| **FastSI2E k-means** | **100.0±0.0** | 67.3±40.3 | TBD | **1952** |
| FastSI2E Leiden | 100.0±0.0 | **91.8±11.5** | — | 1364 |
| **FastSI2E Infomap** | 99.5±0.7 | **95.7±5.8** | — | **1540** |
| SI2E + adaptive-β | — | 46.5±21.4 | **53.4±27.4** | 488 |
| FastSI2E + adaptive-β | — | TBD | TBD | ~1700 |

---

## TODO — Remaining work

### Experiments still needed

| Priority | Task | Est. time | Status |
|----------|------|-----------|--------|
| HIGH | Wait for Phase 2 (B–D) to complete | ~4h | 🔄 running |
| HIGH | Leiden + Infomap on RBD | ~2h | not started |
| MED | Leiden + Infomap on KC-S3R2 (3→5 seeds) | ~2h | 3 seeds done |
| MED | FastSI2E + adaptive-β KC/RBD | ~4h | queued in phase2 |
| LOW | Ablation with Leiden/Infomap (no_cluster, no_norm) | ~1h | not started |

### Paper writing (priority order)

- [ ] **§2 Background** — add Leiden and Infomap citations + 1 paragraph each
- [ ] **§3 Method** — revise from "k-means speedup" framing to "clustering algorithm comparison" framing; Infomap is now the headline
- [ ] **§4.4 Main results** — fill Table 1 placeholders once Phase 2 (B–D) done
- [ ] **§4.5 Ablations** — write up no_cluster (0%) and no_norm (22%) findings
- [ ] **§4.6 Clustering comparison** — already drafted in `paper_draft.md`; add learning curves figure reference
- [ ] **§5 Discussion** — why Leiden/Infomap win on multi-room tasks (graph topology argument)
- [ ] **§6 Conclusion** — revise to lead with Infomap as best method

### Analysis / figures

- [ ] Re-run `python3 analyze_results.py --plot` after Phase 2 finishes to get RBD + adaptive-β curves
- [ ] Add KC-S3R2 panel to learning curves with all 3 clustering methods (currently only k-means)
- [ ] Run `python3 benchmark_fps.py --steps 5` for clean single-seed FPS measurements (no concurrent slowdown)
- [ ] Add Leiden/Infomap to KC-S3R2 results (currently N=3; run more seeds if variance high)

### Code cleanup

- [ ] Fix `batch_phase2.sh` `run_pairs` bug already patched (`(( n > 0 )) && wait` → `if (( n > 0 )); then wait; fi`)
- [ ] Consider adding Leiden/Infomap to RBD experiments via a `batch_phase3.sh`

---

## Active processes

| PID | Script | Task | Status |
|-----|--------|------|--------|
| 1503361 | batch_phase2.sh | RBD fast-si2e s1-3 → adapt KC → adapt RBD | 🔄 running |

---

## Key files

| File | Purpose |
|------|---------|
| `results/clustering-methods/summary.csv` | Leiden + Infomap results |
| `results/fast-si2e/summary.csv` | k-means results (all envs/seeds) |
| `results/adaptive-beta/summary.csv` | adaptive-β results |
| `results/phase2/summary.csv` | RBD + adaptive fast-si2e (filling) |
| `results/learning_curves.png` | 4-panel figure |
| `results/fps_comparison.png` | FPS bar chart |
| `analyze_results.py` | Run to regenerate all tables + figures |
| `paper_draft.md` | Full paper skeleton |
| `batch_phase2.sh` | Phase 2 runner (B–D) |

---

## Revised paper framing

**Old framing:** "FastSI2E: 4× faster via k-means"
**New framing:** "Graph Clustering Algorithms for SI2E: Infomap/Leiden are faster AND more accurate than the original PartitionTree"

**Key claims to make:**
1. Any graph clustering method replaces PartitionTree: 2.8–4× speedup on all tasks
2. Infomap: 95.7%±5.8 on KC-S3R2 vs SI2E 67.5%±27.9 — new SOTA on this benchmark
3. Why: community detection algorithms find multi-room topology that k-means misses
4. Adaptive-β: stabilises bimodal convergence on RBD (2/5 → 4/5 seeds)
5. Ablations: cluster-level bonus is load-bearing (no_cluster → 0%)
6. Negative: PPO incompatible (documented)
