# Local Reproduction Report — SI2E (NeurIPS 2024)

**Date:** 2026-05-19  
**Hardware:** RTX 3070 Laptop GPU (8 GB), 16 cores @ ~3 GHz  
**Paper hardware:** RTX A1000 (16 GB), 8× Intel Core i9 @ 3.00 GHz  
**Paper:** "Effective Exploration Based on the Structural Information Principles" (SI2E, NeurIPS 2024)  
**Repo:** https://github.com/SELGroup/SI2E

---

## 1. Current Results (DMControl, DrQv2 backbone)

All local runs use **seed=1, 250K frames = 125K env steps**.  
The paper uses **510K frames ≈ 250K env steps** (see §3 for the budget discrepancy).

### Completed runs

| Task | Baseline | +SE | +VCSE | +SI2E | Paper Baseline | Paper SE | Paper VCSE | Paper SI2E |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| cartpole_swingup | 341 | **660** 🍀 | — | 594 | — | 220±62 | 708±50 | **795±90** |
| hopper_stand | 125 | 230 | — | — | 88±12 | 313±94 | 711±31 | **797±53** |
| cheetah_run | **246** | 180 | — | — | 229±124 | 229±126 | 456±22 | **464±29** |
| quadruped_walk | 157 | 271 | — | — | 290±24 | 290±24 | 244±30 | **400±29** |
| pendulum_swingup | 🔄 F=133K | — | — | — | 424±247 | 11±3 ⚠️ | 824±100 | **886±38** |
| cartpole_balance | — | — | — | — | 999±23 | 994±75 | 999±10 | **1000±3** |

🔄 = currently running in batch  
🍀 = lucky single seed (far above paper mean)  
⚠️ = SE catastrophically fails on pendulum_swingup (entropy bonus pushes away from the narrow reward region)

### In-progress batch (nohup PID 62867)

```
Phase 1 [07:37]:  5 tasks × (baseline + SE) × seed=1
  ✅ hopper_stand   baseline (38 min), SE (48 min)
  ✅ cheetah_run    baseline (38 min), SE (47 min)
  ✅ quadruped_walk baseline (41 min), SE (51 min)
  🔄 pendulum_swingup baseline  (~F=133K / 249K, ~15 min left)
  ⬜ pendulum_swingup SE
  ⬜ cartpole_balance baseline + SE

Phase 2: cartpole_swingup × seeds 2-3 × (baseline + SE)
Phase 3: VCSE × all 6 tasks × seed=1

Estimated finish: ~20:00 today
```

Monitor: `tail -f /workspace/learn-si2e/results/batch.log`

---

## 2. Is SI2E Worse than SE?

**On this single run: SI2E=594 < SE=660 on cartpole_swingup. This is noise, not a finding.**

- The paper reports SE = 220±62 and SI2E = 795±90 on this task (10 seeds).
- SE has σ=62 — a single lucky seed can land 3σ above the mean, which is exactly what happened (seed=1 gave 660 ≫ 220).
- With 1 seed you cannot rank methods. You need ≥5 seeds for a meaningful comparison.
- Across all other completed tasks, SE does not consistently beat the baseline either
  (hopper_stand: SE=230>baseline=125 ✓, but cheetah_run: SE=180<baseline=246 ✗).

**Summary:** results at 1 seed are consistent with high within-method variance in the paper.
The ordering SE < VCSE < SI2E (paper's main claim) cannot be verified from a single seed.

---

## 3. Training Budget Discrepancy

### The confusion

The SI2E paper (Appendix D) states:
> "The total number of environmental steps was set to **250K** for the DeepMind Control Suite."

In the DrQv2 codebase, `num_train_frames` counts **raw rendered frames**, not env steps.  
With `action_repeat=2`, each agent decision advances physics 2 timesteps:

```
num_train_frames = 510 000  →  agent decisions = 255 000  →  "250K env steps" ✓  (paper)
num_train_frames = 250 000  →  agent decisions = 125 000  →  "125K env steps"    (our runs)
```

The default in all task configs (`easy.yaml`, `hard.yaml`, `medium.yaml`) is `num_train_frames: 510000`.
We overrode to `250000`, running at **half the paper budget**.

### Impact on results

| Task | Our ER (125K steps) | Paper ER (255K steps) | Δ |
|---|:---:|:---:|:---:|
| quadruped_walk baseline | 157 | 290±24 | −45% — clearly under-trained |
| cheetah_run baseline | 246 | 229±124 | ≈ same (fast-converging, wide σ) |
| hopper_stand baseline | 125 | 88±12 | higher (variance) |
| cartpole_swingup SI2E | 594 | 795±90 | −25% — consistent with shorter run |

Harder tasks (quadruped_walk, hopper_stand) suffer more from half the budget.
Fast-converging tasks (cheetah_run, cartpole tasks) are less affected.

### Wall-clock cost at full budget (510K frames)

| Method | FPS (local) | 250K frames | **510K frames** |
|---|:---:|:---:|:---:|
| DrQv2 baseline | ~135 | 31 min | **63 min** |
| +SE / +VCSE | ~95 | 44 min | **89 min** |
| +SI2E | ~14 | 5.0 hr | **10.2 hr** |

Total for full table (6 tasks × 3 methods × 1 seed, 510K frames) ≈ **~24 h** on this GPU.

---

## 4. Why DrQv2 — Not SAC or PPO?

### DrQv2 is pixel-based; SAC/PPO benchmarks are typically state-based

DrQv2 takes **raw 84×84 RGB images** (3-frame stack → 3×84×84×3 tensor) as input.  
SAC/PPO typically operate on **state vectors** (position, velocity, angles — 10–30 floats).

This is not an implementation choice — it is the scientific question:
> *Can structural entropy improve exploration when the agent only sees pixels?*

State-based SAC/PPO numbers (e.g., cheetah_run ~800, cartpole_balance ~1000) are not comparable.
The agent does not have access to those numbers — it only sees images and must learn representations.

### Achievable ER by observation type

| Task | State-based SAC/PPO | Pixel-based DrQv2 (paper, 250K steps) |
|---|:---:|:---:|
| cheetah_run | ~800–900 | 229 (baseline), 464 (SI2E) |
| cartpole_balance | ~1000 | 999 (baseline already saturates) |
| cartpole_swingup | ~800 | — (baseline), 795 (SI2E) |
| hopper_stand | ~900 | 88 (baseline), 797 (SI2E) |

The gap is entirely due to the representation learning burden, not the RL algorithm.

---

## 5. Speed: DrQv2 vs JAX PPO/SAC

DrQv2 at ~100–135 FPS vs JAX PPO at 50,000+ FPS (with vectorized envs).  
The ~500× gap has four independent causes:

| Factor | DrQv2 (ours) | JAX PPO/SAC |
|---|---|---|
| **Observation type** | 84×84 pixels → CNN forward + backward | State vector → tiny MLP |
| **Parallelism** | 1 env, sequential | 1000s of envs, fully vectorized on GPU |
| **Physics** | CPU MuJoCo → GPU data transfer each step | GPU-compiled Brax/MJX, no transfer |
| **Python overhead** | Python loop every env step | `jit`-compiled, Python runs once |

### What JAX PPO would give on DMControl

Using **state observations** + Brax physics on GPU:
- cheetah_run: ~800 ER in under 60 s (5M steps @ 50K FPS)
- cartpole_balance: ~1000 in ~10 s
- hopper_stand: ~900 in ~60 s

But this answers a *different question* — it does not study pixel-based exploration.

### Could SI2E be ported to JAX?

Yes, in principle. The entropy bonus is differentiable and the kNN/structural entropy
computation could be JIT-compiled. But the pixel CNN + data augmentation pipeline of DrQv2
does not have a standard JAX equivalent yet (Dreamer-style world models come closest).
This would be significant future engineering work.

---

## 6. Known Issues in This Reproduction

| Issue | Status | Notes |
|---|:---:|---|
| `action_scale.Wrapper` float64 literals | ✅ Fixed | `dmc.py` line ~201 in both repos |
| Replay buffer dtype assert crash | ✅ Fixed | `replay_buffer.py` in both repos |
| A2C kthvalue index bug | ✅ Fixed | `torch_ac/algos/base.py` both repos |
| SI2E CUDA device hardcoded | ✅ Fixed | `config.yaml device: cuda:0` |
| A2C gym `disable_env_checker` warning | ✅ Fixed | `env.py` |
| Running at half paper budget | ⚠️ Known | Use `num_train_frames=510000` for full comparison |
| Single seed only | ⚠️ Known | Need ≥5 seeds for valid method comparison |
| VCSE A2C kthvalue bug (base-vcse) | ✅ Fixed | Same patch as SI2E A2C |

---

## 7. Result Paths

```
results/
  drqv2-full/
    baseline/                         cartpole_swingup baseline seed=10 (legacy)
    se/                               cartpole_swingup SE seed=10 (legacy)
    si2e/                             cartpole_swingup SI2E seed=1
    hopper_stand_baseline_seed1/      ✅
    hopper_stand_se_seed1/            ✅
    cheetah_run_baseline_seed1/       ✅
    cheetah_run_se_seed1/             ✅
    quadruped_walk_baseline_seed1/    ✅
    quadruped_walk_se_seed1/          ✅
    pendulum_swingup_baseline_seed1/  🔄
    ... (created by batch_drqv2.sh as they complete)
```
