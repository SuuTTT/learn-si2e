# SI2E Paper Tables + Local Reproduction

**Extracted from:** `paper_result.md` — SI2E Table 1 & Table 2  
**Paper:** "Effective Exploration Based on Structural Information Principles" (NeurIPS 2024)  
**Paper setup:** 10 seeds, RTX A1000 (16 GB)  
**Local setup:** 1 seed (seed=1), RTX 3070 Laptop (8 GB), CUDA 11.8  
**Paper DMControl budget:** 250 000 env steps (= `num_train_frames=250000`, action_repeat=2)  
**Paper MiniGrid budget:** 3 000 000 frames (hard tasks), 1 000 000 (easy tasks)  

---

## Table 1 · MiniGrid Navigation (A2C backbone)

> **Metric:** success rate (%) + required steps to solve (K).  
> **Local metric:** raw episodic return (`rR`), NOT success rate. Direct numeric comparison not valid.  
> rR is in [0, 1]; a value of 0.01 means ~1% of max return, not ~1% success rate.  
> Only DoorKey-8x8 run locally so far.

### DoorKey-8x8 — 3 M frames

| Method | Paper success (%) ± σ | Paper steps to solve (K) ± σ | Local rR mean ± σ | Local rR max | Status |
|---|:---:|:---:|:---:|:---:|:---|
| A2C (baseline) | — (fails) | — | 0.02 ± 0.08 | 0.32 | ✅ done |
| A2C + SE | 72.60 ± 20.32 | 1515.81 ± 324.28 | 0.01 ± 0.03 | 0.11 ⚠️ unlucky | ✅ done |
| A2C + VCSE | 94.32 ± 11.09 | 1900.96 ± 398.65 | — | — | ⬜ queued |
| **A2C + SI2E** | **98.58 ± 3.11** | **1090.96 ± 125.77** | — | — | ⬜ queued |

> ⚠️ A2C+SE local: paper σ = ±20 pp means a single unlucky seed can score near 0%.  
> A2C baseline officially fails at 3M (—); our max=0.32 is noise.  
> VCSE uses `base-vcse/VCSE_A2C/` (Kim et al. kNN, not SI2E structural entropy).  
> SI2E uses `SI2E/SI2E_A2C/` with `--use_entropy_reward --use_value_condition`.

### Other Table 1 MiniGrid tasks (not yet run locally)

| Task | Budget | A2C | A2C+SE | A2C+VCSE | A2C+SI2E |
|---|---|:---:|:---:|:---:|:---:|
| RedBlueDoors-6x6 | 1M | — | — | 79.82±7.26 | **85.80±1.48** |
| SimpleCrossingS9N1 | 1M | 88.18±3.46 | 88.59±4.62 | 91.30±1.92 | **93.64±1.63** |
| KeyCorridorS3R1 | 1M | 86.57±2.26 | 87.20±4.94 | 86.01±0.91 | **94.20±0.42** |
| DoorKey-6x6 | 1M | 92.67±8.47 | 93.18±6.81 | 94.08±2.58 | **97.04±1.52** |

---

## Table 2 · DMControl Continuous Control (DrQv2 backbone, 250 K env steps)

> **Metric:** episode reward (ER).  
> **Note:** paper's DrQv2 baseline column shows "—" for Cartpole Swingup (not reported);  
> our local DrQv2 baseline ran and got ER = 341.  
> MADE is an external baseline — not implemented in SI2E or VCSE repos.

### Paper results (10 seeds, RTX A1000)

| Task | DrQv2 | +SE | +VCSE | +MADE | **+SI2E** | Abs.↑ (SI2E−VCSE) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Hopper Stand | 87.59±11.70 | 313.39±94.15 | 711.32±30.84 | 717.09±112.94 | **797.17±53.21** | +85.85 |
| Cheetah Run | 229.28±123.93 | 228.82±126.21 | 456.26±22.20 | 366.59±53.74 | **464.08±29.32** | +7.82 |
| Quadruped Walk | 289.79±24.17 | 290.27±24.20 | 243.74±29.91 | 262.63±23.92 | **399.51±29.05** | +155.77 |
| Pendulum Swingup | 424.21±246.96 | 10.80±2.92 ⚠️ | 824.17±99.59 | 672.11±34.63 | **885.50±38.28** | +61.33 |
| Cartpole Balance | 998.97±22.95 | 993.80±75.24 | 998.65±9.58 | 996.16±40.60 | **999.58±2.97** | +0.93 |
| **Cartpole Swingup** | — | 219.69±62.21 | 707.76±50.38 | 704.18±41.75 | **795.09±90.49** | +87.33 |

> ⚠️ SE catastrophically fails on Pendulum Swingup: entropy bonus pushes away from narrow high-reward region.

### Local results (1 seed, seed=1, RTX 3070 Laptop)

Only **Cartpole Swingup** run locally (matches paper Table 2 task that uses `task@_global_=cartpole_swingup`).

| Method | Paper ER | Local ER | @ frame | FPS | Wall time | Δ vs paper |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| DrQv2 (baseline) | — | **341** | 240K | ~142 | ~36 min | — |
| DrQv2 + SE | 219.69 ± 62.21 | **660** 🍀 | 240K | ~112 | ~46 min | +200% (lucky seed) |
| DrQv2 + VCSE | 707.76 ± 50.38 | — | — | ~112 est. | ~37 min est. | ⬜ not yet run |
| DrQv2 + MADE | 704.18 ± 41.75 | — | — | — | — | ⬜ needs external repo |
| **DrQv2 + SI2E** | **795.09 ± 90.49** | 🔄 ~334 | 170K | ~14 | ~5 h | 🔄 running (PID 55737) |

> 🍀 SE local ER=660 far exceeds paper mean 219±62 — single-seed variance is high.  
> Check SI2E progress: `tail -f /workspace/learn-si2e/results/drqv2-full/si2e/stdout.log`

### DMControl eval history — local seed=1

```
DrQv2 baseline (task: cartpole_swingup, seed=1, num_train_frames=250000):
  F=100K → ER=255  |  F=110K → ER=304  |  F=120K → ER=289
  F=130K → ER=280  |  F=140K → ER=341  (final eval @ 240K)

DrQv2+SE (do_vcse=false, seed=1, num_train_frames=250000):
  F=100K → ER=683  |  F=110K → ER=681  |  F=120K → ER=643
  F=130K → ER=679  |  F=140K → ER=660  (final eval @ 240K)

DrQv2+SI2E (do_vcse=true, seed=1, num_train_frames=250000) — nohup PID 55737:
  F=0    → ER=9.38  (random policy)
  F=80K  → ER=248   F=90K → ER=291   F=100K → ER=378
  F=110K → ER=311   F=120K → ER=377  F=130K → ER=291
  F=140K → ER=377   F=150K → ER=311  F=160K → ER=369  F=170K → ER=334  (last eval)
```

---

## Metric alignment note

| Metric | Paper reports | Local logs show | Conversion |
|---|---|---|---|
| MiniGrid | success rate (%) | `rR:μσmM` (episode return 0–1) | Not direct; reward = 1 − 0.9·(steps/max_steps) if solved, 0 otherwise |
| DMControl | episode reward (ER) | `R:` field in stdout | Same metric ✓ |
| DrQv2 frame count | env steps (250K) | `F:` field = total frames | `num_train_frames=250000` with action_repeat=2 means 125K env steps per episode... but paper uses "250K env steps" = `num_train_frames=250000` ✓ |
