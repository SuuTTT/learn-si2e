# SI2E Research Milestones
*Last updated: 2026-05-29*

---

## Executive Summary

We set out to beat the original SI2E (NeurIPS 2024) on both **accuracy** and **speed**.

| Axis | Status | Best result |
|---|---|---|
| Speed | ✅ | k-means: **4.0× faster** (488 → 1952 FPS) |
| Accuracy (KC-S3R2) | ✅ | Infomap: **95.3%±4.6** (N=5) vs SI2E 67.5%±27.9 (+28 pp) |
| Accuracy (DK-8x8) | ✅ | All methods match SI2E 100% |
| Accuracy (RBD-6x6) | ⚠️ | Infomap 54.4%±38.5 ≈ SI2E 55.7%; Leiden 27.5%±38.9 underperforms |
| Paper | ✅ | `paper/main.pdf` — 5 pages, fully compiled |

**Core story:** Replace the PartitionTree with graph community detection → simultaneously faster AND more accurate. Leiden/Infomap detect the multi-room topology that k-means' Euclidean distance misses.

---

## Completed Experiments

### M1 — Speed Benchmark ✅
**DK-8x8, 1M frames, 3 seeds**

| Method | FPS | Speedup | Clustering cost |
|--------|-----|---------|-----------------|
| SI2E (PartitionTree) | 488 | 1× | ~155 ms |
| FastSI2E k-means | 1952 | **4.0×** | ~1.5 ms |
| FastSI2E Leiden | 1364 | 2.8× | ~27 ms |
| FastSI2E Infomap | 1540 | 3.2× | ~24 ms |

---

### M2 — DK-8x8 Accuracy ✅
**3M frames, 3–5 seeds — all fast methods match SI2E**

| Method | Mean | Std |
|--------|------|-----|
| SI2E | 100.0% | 0.0 |
| FastSI2E k-means | 100.0% | 0.0 |
| FastSI2E Leiden | 100.0% | 0.0 |
| FastSI2E Infomap | 99.5% | 0.7 |

---

### M3 — KC-S3R2 Clustering Comparison ✅
**BREAKTHROUGH: Community detection beats original SI2E**

| Method | N | Mean | Std | vs SI2E |
|--------|---|------|-----|---------|
| SI2E | 5 | 67.5% | 27.9 | — |
| FastSI2E k-means | 5 | 67.3% | 40.3 | −0.2 pp |
| **FastSI2E Leiden** | **5** | **95.1%** | **9.8** | **+27.6 pp** |
| **FastSI2E Infomap** | **5** | **95.3%** | **4.6** | **+27.8 pp** |

Why: KC-S3R2 has multi-room topology. Leiden/Infomap find room-level communities directly via modularity/random-walk compression. k-means' Euclidean distance misses this structure.

---

### M4 — Adaptive-β Stability ✅

| Method | Env | Mean | Std | Converge rate |
|--------|-----|------|-----|---------------|
| SI2E fixed-β | RBD | 34.0% | 40.1 | 2/5 |
| **SI2E adaptive-β** | RBD | **53.4%** | **27.4** | **4/5** |
| SI2E fixed-β | KC | 67.5% | 27.9 | — |
| SI2E adaptive-β | KC | 46.5% | 21.4 | — |

Formula: `β_t = β₀ · max(0.1, 1 − recent_success_rate)`

Note: adaptive-β *hurts* SI2E on KC (−21 pp). It helps on RBD where bimodal failure is the dominant problem.

---

### M5 — PPO-SI2E Negative Result ✅

| Config | DK-8x8 | KC-S3R2 | RBD |
|--------|---------|---------|-----|
| PPO-SI2E | 58.5%±46 | 0% | — |
| PPO-FastSI2E | — | 0% | — |
| PPO-adaptive | — | — | 0.1% |

Hypothesis: PPO's 4-epoch update creates reward-distribution shift with the once-per-rollout intrinsic bonus. Documented as negative result.

---

### M6 — Phase 2 Experiments ✅ DONE

#### (A) KC-S3R2 FastSI2E extra seeds ✅
- s4: 100%, s5: 100% → full 5-seed KC result: 67.3%±40.3

#### (B) RBD FastSI2E k-means s1–3 ✅
- s1: 68.5%, s2: 82.6%, s3: 0.0% → **50.4%±36.1**
- Comparable to SI2E (55.7%), at 4× speed
- Still bimodal (2/3 converge)

#### (C) FastSI2E + adaptive-β on KC s1–5 ✅ — NEGATIVE
- 7.5%, 66.5%, 37.0%, 0.0%, 32.0% → **28.6%±23.6**
- **Worse than FastSI2E alone (67.3%)** and SI2E+adaptive (46.5%)

#### (D) FastSI2E + adaptive-β on RBD s1–5 ✅ — NEGATIVE
- 73.5%, 0.0%, 0.5%, 76.0%, 0.0% → **30.0%±36.5**
- **Worse than SI2E+adaptive (53.4%±27.4)**
- Hypothesis: noisy k-means clusters make the success-rate signal unreliable for β scheduling

---

### M7 — Phase 3: RBD Clustering + 5-Seed KC ✅ DONE

#### (A) KC-S3R2 Leiden s4,s5 ✅
- s4: 100%, s5: 100% → N=5 full: **95.1%±9.8** (+28 pp vs SI2E)

#### (B) KC-S3R2 Infomap s4,s5 ✅
- s4: 93.0%, s5: 96.5% → N=5 full: **95.3%±4.6** (+28 pp vs SI2E, all 5 seeds converge)

#### (C) RBD-6x6 Leiden s1–3 ✅
- s1: 82.5%, s2: 0%, s3: 0% → **27.5%±38.9**
- Bimodal: 1/3 seeds converge; Leiden underperforms on RBD

#### (D) RBD-6x6 Infomap s1–3 ✅
- s1: 80%, s2: 0%, s3: 83.3% → **54.4%±38.5**
- Matches SI2E (55.7%); bimodal pattern persists (2/3 converge)

#### (E) RBD-6x6 Infomap+adaptive-β s1–3 ✅ — NEGATIVE
- s1: 75.7%, s2: 0%, s3: 0% → **25.2%±35.7**
- Below Infomap alone (54.4%) and k-means+adaptive (30.0%)
- Confirms bimodal RBD convergence is environment-level, not cluster-method-level

---

## Full Results Table

| Method | DK-8x8 | KC-S3R2 | RBD-6x6 | FPS |
|--------|---------|---------|---------|-----|
| VCSE (orig.) | 97.8±2.8 | 54.0±45.0 | 55.4±39.1 | ~1950 |
| SI2E (orig.) | 100.0±0.0 | 67.5±27.9 | 55.7±38.7 | 488 |
| FastSI2E k-means | **100.0±0.0** | 67.3±40.3 | 50.4±36.1 | **1952** |
| FastSI2E Leiden | **100.0±0.0** | 95.1±9.8 | 27.5±38.9 | 1364 |
| **FastSI2E Infomap** | 99.5±0.7 | **95.3±4.6** | 54.4±38.5 | 1540 |
| SI2E + adaptive-β | — | 46.5±21.4 | **53.4±27.4** | 488 |
| FastSI2E + adaptive-β | — | 28.6±23.6 ⚠️ | 30.0±36.5 ⚠️ | ~1900 |

---

## Paper Status

| File | Status |
|------|--------|
| `paper/main.tex` | ✅ written, all results filled |
| `paper/main.pdf` | ✅ compiled, 5 pages |
| `paper/references.bib` | ✅ 13 citations |
| `paper/learning_curves.png` | ✅ 4-panel figure |
| `paper/fps_comparison.png` | ✅ bar chart |

**Paper sections status:**
- Abstract ✅
- §1 Introduction ✅
- §2 Background ✅ (VCSE, SI2E, PartitionTree)
- §3 FastSI2E ✅ (k-means, Leiden, Infomap, adaptive-β algorithm box)
- §4 Experiments ✅ (Table 1 filled, ablations, clustering comparison, FPS, learning curves, PPO negative)
- §5 Discussion ✅ (k-means proxy argument, Leiden vs Infomap, adaptive-β+k-means incompatibility, bimodal, PPO)
- §6 Conclusion ✅
- References ✅

---

## TODO — Ranked by Impact on Paper

### HIGH PRIORITY — Missing data for complete paper

| # | Task | Why important | Est. time | Status |
|---|------|---------------|-----------|--------|
| 1 | **Leiden + Infomap on RBD** (s1–3) | Paper Table 1 shows "---" for RBD Leiden/Infomap; need to know if community detection helps on RBD too | ~2h | ✅ DONE |
| 2 | **Extend KC-S3R2 Leiden + Infomap to 5 seeds** | Currently N=3; variance is large (11.5, 5.8); N=5 would be more credible for paper | ~2h | ✅ DONE |
| 3 | **Adaptive-β + Infomap/Leiden on RBD** | If community detection fixes adaptive-β incompatibility, it's a strong finding | ~3h | ✅ DONE (negative) |

### MEDIUM PRIORITY — Figures and analysis

| # | Task | Why | Est. time | Status |
|---|------|-----|-----------|--------|
| 4 | Re-run `python3 analyze_results.py --plot` after HIGH tasks done | Learning curves panel for KC needs all 3 clustering methods (currently k-means only) | 5 min | ✅ DONE |
| 5 | Run `python3 benchmark_fps.py --steps 5` | Get clean single-seed FPS (current FPS from training logs, noisy due to concurrent seeds) | 15 min | ✅ DONE |
| 6 | Add KC-S3R2 Leiden/Infomap to learning curves figure | Currently only k-means on KC panel | 30 min | ✅ DONE |

### LOW PRIORITY — Nice to have

| # | Task | Why | Est. time | Status |
|---|------|-----|-----------|--------|
| 7 | Ablation with Leiden/Infomap (`no_cluster`, `no_norm`) | Currently ablations are k-means only; verify cluster bonus is load-bearing for all methods | ~1h | not started |
| 8 | Fix `analyze_results.py` numpy RuntimeWarning (empty slice) | Cosmetic | 10 min | not started |

### PAPER WRITING — After experiment gaps filled

| # | Section | Task |
|---|---------|------|
| W1 | §4.4 Table 1 | Fill RBD column for Leiden/Infomap once task #1 done | ✅ DONE |
| W2 | §4.6 Clustering comparison | Update Table 3 with 5-seed KC data (task #2) | ✅ DONE |
| W3 | §4.3 Key findings | Add finding about whether adaptive-β works with Leiden/Infomap (task #3) | ✅ DONE |
| W4 | §5 Discussion | Strengthen the "why community detection wins on multi-room tasks" argument with RBD data | ✅ DONE |
| W5 | §6 Conclusion | Final pass once all results are in | ✅ DONE |

---

## Recommended Next Step

**Run batch_phase3.sh** to close the biggest paper gaps in one go:

```bash
# Phase 3: Leiden + Infomap on RBD; extend KC to 5 seeds
# ~4h total, can run overnight
nohup ./batch_phase3.sh > logs/phase3.log 2>&1 &
```

This would require creating `batch_phase3.sh` covering:
- Leiden on RBD s1–3
- Infomap on RBD s1–3
- Leiden on KC-S3R2 s4–5 (extend from 3 to 5 seeds)
- Infomap on KC-S3R2 s4–5
- (optional) Infomap+adaptive-β on RBD s1–3

After that, the paper would have a complete Table 1 with no "---" entries for the main methods.

---

## Key Files

| File | Purpose |
|------|---------|
| `results/clustering-methods/summary.csv` | Leiden + Infomap results (DK + KC, N=3) |
| `results/fast-si2e/summary.csv` | k-means results (all envs/seeds) |
| `results/adaptive-beta/summary.csv` | adaptive-β results |
| `results/phase2/summary.csv` | RBD fast-si2e, fast+adaptive KC/RBD |
| `paper/main.tex` | Full paper |
| `paper/main.pdf` | Compiled PDF (5 pages) |
| `analyze_results.py` | Run to regenerate all tables + figures |
| `batch_phase2.sh` | Phase 2 runner (complete) |

---

## Negative Results (all documented in paper)

| Experiment | Result | Hypothesis |
|---|---|---|
| PPO + SI2E (any variant) | 0% everywhere | 4-epoch update causes reward-distribution shift with stale bonus |
| FastSI2E k-means + adaptive-β | Worse than either alone on KC and RBD | Noisy k-means clusters make success-rate signal unreliable for β scheduling |
| k-means on KC-S3R2 (bimodal) | 3/5 seeds converge, 2/5 fail near 0% | Euclidean clustering misses multi-room community structure |
