# Main Result Tables: RE3 → VCSE → SI2E

Three papers in the state-entropy exploration lineage. Each builds on the previous.

| Paper | Venue | Algorithm | Key idea |
|---|---|---|---|
| **RE3** (Seo et al.) | ICML 2021 | A2C+RE3, RAD+RE3, Dreamer+RE3 | kNN entropy with fixed random encoder |
| **VCSE** (Kim et al.) | NeurIPS 2023 | A2C+VCSE, DrQv2+VCSE | Value-conditional kNN entropy |
| **SI2E** (Zeng et al.) | NeurIPS 2024 | A2C+SI2E, DrQv2+SI2E | Structural entropy via encoding tree + value conditioning |

> **Note on result format:** RE3 and VCSE report results primarily as **learning curves** (figures),
> not summary tables. Numeric values below are read from the paper text. SI2E provides full
> tables with mean ± std.

---

## RE3 — "State Entropy Maximization with Random Encoders for Efficient Exploration"
**Seo, Chen, Shin, Lee, Abbeel, Seo — ICML 2021**  
**Backbone:** RAD (model-free) or Dreamer (model-based) for DMControl; A2C for MiniGrid  
**Runs:** 5 seeds  
**DMControl steps:** 500K  

### Table 1 (from paper text) — MiniGrid Navigation (A2C backbone)

Results reported as learning curves (Figure 11 in paper); numbers extracted from text.

| Task | A2C | A2C+ICM | A2C+RND | **A2C+RE3** |
|---|---|---|---|---|
| Empty-16x16 | solves | — | — | solves faster |
| DoorKey-6x6 | partial | — | — | improves |
| **DoorKey-8x8** @2.4M steps | **fails** (≈0) | 0.20 return | fails | **0.49 return** |
| Unlock | — | — | — | improves |

> RE3 is the first to solve DoorKey-8x8 with A2C. Return 0.49 means ~49% of max reward
> (agent solves in ~half the allowed steps on successful episodes).

### Table 2 (from paper text) — DMControl (RAD + Dreamer backbones)

Results reported as learning curves (Figure 4, 14 in paper).

| Task | RAD | DrQ | Dreamer | **RAD+RE3** | **Dreamer+RE3** |
|---|---|---|---|---|---|
| Cheetah Run Sparse | fails | fails | — | **601.6** | — |
| Hopper Stand | low | — | low | improves | improves |
| Reacher Hard | — | — | — | improves | improves |
| Cartpole Balance | high | high | high | no degradation | no degradation |
| Cartpole Swingup | — | — | — | improves | improves |

> RE3 key finding: improves sparse-reward tasks without hurting dense-reward tasks.
> Results at 500K steps (model-free) or 1M steps (model-based).

---

## VCSE — "Accelerating RL with Value-Conditional State Entropy Exploration"
**Kim, Shin, Abbeel, Seo — NeurIPS 2023**  
**Backbone:** A2C for MiniGrid; DrQv2 for DMControl & MetaWorld  
**Runs:** 16 seeds  
**DMControl steps:** 50K (much shorter than RE3's 500K or SI2E's 250K)  

### Table 1 — MiniGrid Navigation (A2C backbone, results from learning curves Fig. 2)

Key numbers quoted in paper text:

| Task | Steps budget | A2C | A2C+SE | **A2C+VCSE** |
|---|---|---|---|---|
| LavaGapS7 | 100K | low | 13.9% | **88.8%** |
| Empty-16x16 | 250K | solves | improves | **best** |
| DoorKey-6x6 | 300K | partial | partial | **best** |
| DoorKey-8x8 | 3M | fails | partial | **best** |
| Unlock | 1M | fails | partial | **best** |
| SimpleCrossingS9N1 | 1M | partial | partial | **best** |

> VCSE key finding: SE fails on LavaGapS7 (13.9%) because the entropy bonus pushes the agent
> to explore low-value states before the crossing point. VCSE (value-conditioned) reaches 88.8%.

### Table 2 — DMControl (DrQv2 backbone, results from learning curves Fig. 5)

VCSE evaluates on **different tasks** from SI2E — includes sparse-reward variants.  
Results at 50K environment steps.

| Task | Reward type | DrQv2 | DrQv2+SE | **DrQv2+VCSE** |
|---|---|---|---|---|
| Hopper Stand | sparse | low | improves | **best** |
| Walker Walk Sparse | sparse | low | low | **best** |
| Walker Walk | dense | high | **degrades** | best (≈DrQv2) |
| Cheetah Run Sparse | sparse | low | low | **best** |
| Cartpole Swingup Sparse | sparse | low | low | **best** |
| Pendulum Swingup | dense | high | slight drop | **≈DrQv2** |

> VCSE key finding: SE **degrades** Walker Walk (dense reward). VCSE fixes this.
> Note: VCSE paper uses 50K steps vs SI2E's 250K; direct number comparison not valid.

---

## SI2E — "Effective Exploration Based on Structural Information Principles"
**Zeng, Peng, Li — NeurIPS 2024**  
**Backbone:** A2C for MiniGrid/MetaWorld; DrQv2 for DMControl  
**Runs:** 10 seeds  
**MiniGrid steps:** 3000K (hard) / 1000K (easy)  
**MetaWorld steps:** 200K (hard) / 100K (easy)  
**DMControl steps:** 250K  

### Table 1 — MiniGrid Navigation + MetaWorld Manipulation

Bold = best, underline = second best.

#### MiniGrid Navigation (A2C backbone)

| Task | A2C | A2C+SE | A2C+VCSE | **A2C+SI2E** | Avg. Abs.↑ |
|---|---|---|---|---|---|
| **RedBlueDoors-6x6** Success (%) | — | — | 79.82 ± 7.26 | **85.80 ± 1.48** | +5.98 |
| RedBlueDoors-6x6 Steps (K) | — | — | 1161.90 ± 241.59 | **461.90 ± 61.53** | −700↓ |
| **SimpleCrossingS9N1** Success (%) | 88.18 ± 3.46 | 88.59 ± 4.62 | 91.30 ± 1.92 | **93.64 ± 1.63** | +2.34 |
| SimpleCrossingS9N1 Steps (K) | 570.08 ± 15.87 | 394.39 ± 66.14 | 204.02 ± 25.60 | **139.17 ± 27.03** | −64.85↓ |
| **KeyCorridorS3R1** Success (%) | 86.57 ± 2.26 | 87.20 ± 4.94 | 86.01 ± 0.91 | **94.20 ± 0.42** | +7.00 |
| KeyCorridorS3R1 Steps (K) | 658.74 ± 21.03 | 463.86 ± 38.27 | 190.20 ± 6.11 | **129.06 ± 6.11** | −61.14↓ |
| **DoorKey-6x6** Success (%) | 92.67 ± 8.47 | 93.18 ± 6.81 | 94.08 ± 2.58 | **97.04 ± 1.52** | +2.96 |
| DoorKey-6x6 Steps (K) | 567.20 ± 96.57 | 476.34 ± 94.63 | 336.75 ± 19.84 | **230.60 ± 19.85** | −106.15↓ |
| **DoorKey-8x8** Success (%) | — | 72.60 ± 20.32 | 94.32 ± 11.09 | **98.58 ± 3.11** | +4.26 |
| DoorKey-8x8 Steps (K) | — | 1515.81 ± 324.28 | 1900.96 ± 398.65 | **1090.96 ± 125.77** | −424.85↓ |

#### MetaWorld Manipulation (DrQv2 backbone)

| Task | DrQv2 | DrQv2+SE | DrQv2+VCSE | **DrQv2+SI2E** |
|---|---|---|---|---|
| Button Press Success (%) | 92.48 ± 11.96 | 91.34 ± 18.37 | 93.12 ± 3.43 | **97.13 ± 3.35** |
| Button Press Steps (K) | 669.78 ± 154.74 | 634.37 ± 240.51 | 405.22 ± 52.22 | **309.14 ± 53.71** |
| Door Open Success (%) | — | — | 80.90 ± 10.19 | **95.77 ± 1.05** |
| Door Open Steps (K) | — | — | — | **87.5 ± 2.5** |
| Unlock Success (%) | — | 25.31 ± 7.40 | 82.74 ± 7.46 | **95.96 ± 3.00** |
| Unlock Steps (K) | — | — | 175.0 ± 5.0 | **82.5 ± 2.5** |
| Drawer Open Success (%) | 94.55 ± 4.64 | 93.05 ± 7.67 | 89.80 ± 3.29 | **99.60 ± 0.57** |
| Drawer Open Steps (K) | 105.0 ± 5.0 | 95.0 ± 5.0 | 77.5 ± 2.5 | **62.5 ± 7.5** |
| Faucet Close Success (%) | 53.33 ± 1.92 | 92.36 ± 3.66 | 94.21 ± 1.74 | **99.37 ± 1.18** |
| Faucet Open Success (%) | — | — | 87.23 ± 5.29 | **97.06 ± 1.39** |
| Window Open Success (%) | 88.18 ± 1.50 | 93.14 ± 2.03 | 93.17 ± 1.45 | **99.46 ± 0.35** |

### Table 2 — DMControl Continuous Control (DrQv2 backbone, 250K steps, 10 seeds)

| Domain, Task | DrQv2 | DrQv2+SE | DrQv2+VCSE | DrQv2+MADE | **DrQv2+SI2E** | Abs.↑ (%) |
|---|---|---|---|---|---|---|
| Hopper Stand | 87.59 ± 11.70 | 313.39 ± 94.15 | 711.32 ± 30.84 | 717.09 ± 112.94 | **797.17 ± 53.21** | +80.08 (11.17%) |
| Cheetah Run | 229.28 ± 123.93 | 228.82 ± 126.21 | 456.26 ± 22.20 | 366.59 ± 53.74 | **464.08 ± 29.32** | +7.82 (1.71%) |
| Quadruped Walk | 289.79 ± 24.17 | 290.27 ± 24.20 | 243.74 ± 29.91 | 262.63 ± 23.92 | **399.51 ± 29.05** | +109.24 (37.63%) |
| Pendulum Swingup | 424.21 ± 246.96 | 10.80 ± 2.92 ⚠️ | 824.17 ± 99.59 | 672.11 ± 34.63 | **885.50 ± 38.28** | +61.33 (7.44%) |
| Cartpole Balance | 998.97 ± 22.95 | 993.80 ± 75.24 | 998.65 ± 9.58 | 996.16 ± 40.60 | **999.58 ± 2.97** | +0.93 (0.09%) |
| Cartpole Swingup | — | 219.69 ± 62.21 | 707.76 ± 50.38 | 704.18 ± 41.75 | **795.09 ± 90.49** | +87.33 (12.34%) |

> ⚠️ SE catastrophically fails on Pendulum Swingup (10.80 vs baseline 424.21) — the entropy bonus
> pushes the agent away from the narrow high-value swinging region. VCSE and SI2E both fix this.
> Cartpole Balance is near-saturated for all methods (baseline already gets ~999).

---

## Cross-paper comparison on shared tasks

### DoorKey-8x8 (A2C, MiniGrid)

| Method | Paper | Success @ steps |
|---|---|---|
| A2C baseline | RE3 | fails (0) @ 2.4M |
| A2C + SE (RE3) | RE3 | 0.49 return @ 2.4M |
| A2C + SE | VCSE / SI2E | 72.60 ± 20.32% @ 3M |
| A2C + VCSE | VCSE / SI2E | 94.32 ± 11.09% @ 3M |
| **A2C + SI2E** | SI2E | **98.58 ± 3.11%** @ 3M |

> RE3 uses episode return (0–1 scale), VCSE/SI2E use success rate (%). Not directly comparable
> but the trend is clear: VCSE > SE > baseline; SI2E > VCSE.

### DMControl — key sparse tasks

| Method | Cheetah Run Sparse (RE3) | Hopper Stand (SI2E) |
|---|---|---|
| Baseline | fails | 87.59 |
| +SE | — | 313.39 |
| +RE3 (RAD backbone) | **601.6** | — |
| +VCSE | — | 711.32 |
| +SI2E | — | **797.17** |

> RE3 and SI2E evaluate on different tasks and different backbones — not directly comparable.

---

## Key algorithmic differences

| Property | RE3 | VCSE | SI2E |
|---|---|---|---|
| Entropy estimator | kNN (random encoder) | kNN (value-conditioned) | Structural entropy via encoding tree |
| Value conditioning | ✗ | ✓ | ✓ |
| Encoder | Fixed random | Fixed random | Learned (DB bottleneck) |
| Complexity per update | O(n log n) kNN | O(n log n) kNN | O(n²) encoding tree ← bottleneck |
| Compute overhead vs baseline | ~2× slower | ~2× slower | ~10× slower |
| Hyperparameter β | sensitive | less sensitive | robust |
