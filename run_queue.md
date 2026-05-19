# Experiment Run Queue

**Project:** SI2E vs VCSE reproduction  
**Hardware:** RTX 3070 Laptop (8 GB), 16 CPUs, 15 GB RAM  
**Ordered by:** estimated wall time (fastest first)  
**Scope:** SI2E paper Table 1 (DoorKey-8x8) + Table 2 (Cartpole Swingup), 1 seed each  

---

## Quick summary

| # | Task | Algo | Budget | Est. time | Result path | Status | Final result |
|---|---|---|---|---|---|---|---|
| 1 | DoorKey-8x8 | A2C baseline | 3M frames | ~9 min | `results/a2c-full/baseline/` | ✅ done | rR:μ=0.02, max=0.32 |
| 2 | DoorKey-8x8 | A2C + SE | 3M frames | ~20 min | `results/a2c-full/se/` | ✅ done | rR:μ=0.01, max=0.11 ⚠️ |
| 3 | DoorKey-8x8 | A2C + VCSE | 3M frames | ~21 min | `results/a2c-full/vcse/` | ⬜ queued | — |
| 4 | Cartpole Swingup | DrQv2 baseline | 250K frames | ~29 min | `results/drqv2-full/baseline/` | ✅ done | ER=341 @ 240K |
| 5 | Cartpole Swingup | DrQv2 + SE | 250K frames | ~37 min | `results/drqv2-full/se/` | ✅ done | ER=660 @ 240K 🍀 |
| 6 | Cartpole Swingup | DrQv2 + VCSE | 250K frames | ~37 min | `results/drqv2-full/vcse/` | ⬜ queued | — |
| 7 | DoorKey-8x8 | A2C + SI2E | 3M frames | ~97 min | `results/a2c-full/si2e/` | ⬜ queued | — |
| 8 | Cartpole Swingup | DrQv2 + SI2E | 250K frames | ~296 min | `results/drqv2-full/si2e/` | 🔄 running (PID 55737) | 🔄 ~334 @ 170K |

> MADE: external baseline not present in SI2E or VCSE repos — skipped until repo is sourced.

---

## Run details (ordered by estimated train time)

### ✅ 1 · A2C baseline — DoorKey-8x8

| Field | Value |
|---|---|
| **Repo** | `SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/` |
| **Budget** | 3 000 000 frames |
| **FPS (measured)** | 5 592 |
| **Est. wall time** | ~9 min |
| **Result path** | `results/a2c-full/baseline/train.log` |
| **Status** | ✅ completed |
| **Final rR** | μ=0.02, σ=0.08, max=0.32 @ F=2 944 000 |

```bash
cd /workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model baseline \
  --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/baseline
```

---

### ✅ 2 · A2C + SE — DoorKey-8x8

| Field | Value |
|---|---|
| **Repo** | `SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/` |
| **Budget** | 3 000 000 frames |
| **FPS (measured)** | 2 493 |
| **Est. wall time** | ~20 min |
| **Result path** | `results/a2c-full/se/train.log` |
| **Status** | ✅ completed |
| **Final rR** | μ=0.01, σ=0.03, max=0.11 @ F=2 944 000 ⚠️ unlucky seed |

```bash
cd /workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model se \
  --use_entropy_reward \
  --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/se
```

> SI2E repo SE uses **structural entropy** (not kNN). For paper's SE comparison baseline (kNN),
> use the VCSE_A2C repo with `--use_entropy_reward` only.

---

### ⬜ 3 · A2C + VCSE — DoorKey-8x8

| Field | Value |
|---|---|
| **Repo** | `base-vcse/VCSE_A2C/rl-starter-files/rl-starter-files/` |
| **Budget** | 3 000 000 frames |
| **FPS (estimated)** | ~2 400 (same loop as SE) |
| **Est. wall time** | ~21 min |
| **Result path** | `results/a2c-full/vcse/` (to be created) |
| **Status** | ⬜ queued |

```bash
cd /workspace/learn-si2e/base-vcse/VCSE_A2C/rl-starter-files/rl-starter-files
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model vcse \
  --use_entropy_reward --use_value_condition \
  --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/vcse
```

> Uses Kim et al. **kNN entropy with value conditioning** (true VCSE from NeurIPS 2023).

---

### ✅ 4 · DrQv2 baseline — Cartpole Swingup

| Field | Value |
|---|---|
| **Repo** | `SI2E/SI2E_DrQv2/` |
| **Budget** | 250 000 frames (`num_train_frames=250000`) |
| **FPS (measured)** | ~142 |
| **Est. wall time** | ~29 min |
| **Result path** | `results/drqv2-full/baseline/eval.csv` |
| **Status** | ✅ completed |
| **Final ER** | 341 @ frame 240K |

```bash
cd /workspace/learn-si2e/SI2E/SI2E_DrQv2
MUJOCO_GL=egl python3 train.py \
  agent._target_=drqv2.DrQV2Agent \
  task@_global_=cartpole_swingup \
  seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/baseline
```

---

### ✅ 5 · DrQv2 + SE — Cartpole Swingup

| Field | Value |
|---|---|
| **Repo** | `SI2E/SI2E_DrQv2/` |
| **Budget** | 250 000 frames |
| **FPS (measured)** | ~112 |
| **Est. wall time** | ~37 min |
| **Result path** | `results/drqv2-full/se/eval.csv` |
| **Status** | ✅ completed |
| **Final ER** | 660 @ frame 240K 🍀 (paper: 219±62) |

```bash
cd /workspace/learn-si2e/SI2E/SI2E_DrQv2
MUJOCO_GL=egl python3 train.py \
  agent._target_=si2e.SI2EAgent agent.do_vcse=false \
  task@_global_=cartpole_swingup \
  seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/se
```

> SI2E repo SE = **structural entropy** without value conditioning.

---

### ⬜ 6 · DrQv2 + VCSE — Cartpole Swingup

| Field | Value |
|---|---|
| **Repo** | `base-vcse/VCSE_DrQv2/` |
| **Budget** | 250 000 frames |
| **FPS (estimated)** | ~112 (same kNN loop as SE) |
| **Est. wall time** | ~37 min |
| **Result path** | `results/drqv2-full/vcse/` (to be created) |
| **Status** | ⬜ queued |

```bash
cd /workspace/learn-si2e/base-vcse/VCSE_DrQv2
MUJOCO_GL=egl python3 train.py \
  agent.do_vcse=true \
  task@_global_=cartpole_swingup \
  seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/vcse
```

> Uses Kim et al. **kNN entropy with value conditioning** (VCSE DrQv2 repo).  
> Verify `num_train_frames` is a supported arg: `grep num_train_frames base-vcse/VCSE_DrQv2/cfgs/config.yaml`

---

### ⬜ 7 · A2C + SI2E — DoorKey-8x8

| Field | Value |
|---|---|
| **Repo** | `SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/` |
| **Budget** | 3 000 000 frames |
| **FPS (measured at 500K)** | 518 |
| **Est. wall time** | ~97 min (1.6 h) |
| **Result path** | `results/a2c-full/si2e/` (to be created) |
| **Status** | ⬜ queued |

```bash
cd /workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files
PYTHONPATH="$(pwd):$PYTHONPATH" nohup python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model si2e \
  --use_entropy_reward --use_value_condition \
  --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/si2e \
  > /workspace/learn-si2e/results/a2c-full/si2e/stdout.log 2>&1 &
echo "PID: $!"
```

---

### 🔄 8 · DrQv2 + SI2E — Cartpole Swingup

| Field | Value |
|---|---|
| **Repo** | `SI2E/SI2E_DrQv2/` |
| **Budget** | 250 000 frames |
| **FPS (measured)** | ~14 |
| **Est. wall time** | ~296 min (4.9 h) |
| **Result path** | `results/drqv2-full/si2e/eval.csv` |
| **Status** | 🔄 running — PID 55737 |
| **Progress** | ~179K / 250K frames, ER ≈ 334 @ 170K |

```bash
# Check progress:
tail -5 /workspace/learn-si2e/results/drqv2-full/si2e/eval.csv
tail -3 /workspace/learn-si2e/results/drqv2-full/si2e/stdout.log

# Full launch command (for reference):
# cd /workspace/learn-si2e/SI2E/SI2E_DrQv2
# MUJOCO_GL=egl nohup python3 train.py \
#   agent._target_=si2e.SI2EAgent agent.do_vcse=true \
#   task@_global_=cartpole_swingup \
#   seed=1 num_train_frames=250000 device=cuda:0 \
#   hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/si2e \
#   > /workspace/learn-si2e/results/drqv2-full/si2e/stdout.log 2>&1 &
```

---

## Time budget to complete all remaining runs

| Remaining run | Est. time |
|---|---|
| A2C + VCSE (DoorKey-8x8) | ~21 min |
| DrQv2 + VCSE (Cartpole Swingup) | ~37 min |
| A2C + SI2E (DoorKey-8x8) | ~97 min |
| DrQv2 + SI2E (Cartpole Swingup) | ~296 min 🔄 already running |
| **Total remaining** | **~451 min (~7.5 h)** |

A2C runs are CPU-bound (can run in parallel with GPU runs).  
Optimal order: launch A2C+VCSE and A2C+SI2E simultaneously on CPU while DrQv2+VCSE runs on GPU.  
Parallel strategy: run #3 + #6 together → then #7 (or overlap #7 with GPU runs).
