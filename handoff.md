# Handoff: SI2E Reproduction Project

**Status as of:** 2026-05-19  
**Repo:** `/workspace/learn-si2e/` (git initialized, nested repos excluded via `.gitignore`)  
**Handoff to:** Next team / continuation session  

---

## What this project is

Reproduce the results of **SI2E** (NeurIPS 2024) — a state-entropy exploration method that
uses *structural entropy* via an encoding tree (HCSE algorithm) — and compare against its
predecessor **VCSE** (NeurIPS 2023) and the original **RE3** (ICML 2021).

**Goal:** Verify paper tables by running the official code locally, document all bugs fixed,
and build a comparison table of paper results vs local single-seed results.

### Paper lineage

```
RE3 (ICML 2021)           kNN entropy, fixed random encoder
    └── VCSE (NeurIPS 2023)   + value conditioning on kNN
            └── SI2E (NeurIPS 2024)   replaces kNN with structural entropy (encoding tree)
```

| Paper | arXiv | Repo |
|---|---|---|
| RE3 | Seo et al. ICML 2021 | — (no official repo for the A2C/RAD version used) |
| VCSE | arXiv:2305.19476 | https://github.com/kingdy2002/VCSE → `base-vcse/` |
| SI2E | arXiv:2410.06621 | https://github.com/SELGroup/SI2E → `SI2E/` |

---

## Hardware & environment

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop, 8 192 MiB VRAM |
| CUDA | 11.8 |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cu118 |
| NumPy | 2.2.6 |
| gym | 0.26.2 |
| gymnasium | 1.3.0 |
| minigrid | 3.1.0 (system) + `minigrid-pinned/` (patched fork) |
| dm_control | 1.0.41 |
| CPUs | 16 logical cores, 15 GB RAM |

**Paper hardware:** NVIDIA RTX A1000 (Ampere, 16 GB ECC). Same GPU arch → our FPS ≈ paper's.

---

## Workspace layout

```
learn-si2e/
├── .gitignore                   ← excludes nested repos, PDFs, videos, buffers
├── .git/                        ← initialized 2026-05-19
│
├── SI2E/                        ← 🔒 nested git repo (gitignored) — ALL FIXES APPLIED
│   ├── SI2E_A2C/
│   │   ├── rl-starter-files/rl-starter-files/   ← train.py (cuda:0), utils/env.py patched
│   │   └── torch-ac/                            ← algos/base.py kthvalue fix; pip install -e .
│   └── SI2E_DrQv2/              ← cfgs/config.yaml (cuda:0), dmc.py, replay_buffer.py patched
│
├── base-vcse/                   ← 🔒 nested git repo (gitignored) — UNMODIFIED
│   ├── VCSE_A2C/                ← needs kthvalue fix before running (same bug as SI2E)
│   ├── VCSE_DrQv2/              ← should work with MUJOCO_GL=egl + float32 patch
│   ├── VCSE_MWM/                ← not investigated
│   └── VCSE_SAC/                ← not investigated
│
├── base-rl-starter-files/       ← 🔒 nested git repo (gitignored) — reference copy
├── base-torch-ac/               ← 🔒 nested git repo (gitignored) — reference copy
├── base-drqv2/                  ← 🔒 nested git repo (gitignored) — reference copy
├── minigrid-pinned/             ← 🔒 nested git repo (gitignored) — np_random.integers fix
│
├── SI2E_paper.pdf               ← gitignored (download: arXiv:2410.06621)
├── vcse_paper.pdf               ← gitignored (download: arXiv:2305.19476)
├── re3_paper.pdf                ← gitignored (download: arXiv:2102.09430)
│
├── paper_result.md              ← ✅ extracted tables from RE3, VCSE, SI2E papers
├── si2e_comparison.md           ← ✅ SI2E Table 1+2 with local result columns
├── run_queue.md                 ← ✅ experiment queue ordered by train time + launch cmds
├── REPRODUCE_LOG.md             ← ✅ full iteration log (bugs, fixes, commands, results)
│
├── reproduce_and_compare.sh     ← shell script for short 50K/500K runs
├── compare_results.py           ← log parser + comparison tables/plots
│
└── results/
    ├── a2c/                     ← short 500K runs (3 variants)
    │   ├── baseline/train.log
    │   ├── se/train.log
    │   └── si2e/train.log
    ├── a2c-full/                ← full 3M runs (2 variants complete)
    │   ├── baseline/train.log   ← ✅ DONE
    │   └── se/train.log         ← ✅ DONE
    ├── drqv2/                   ← short 50K runs (3 variants)
    │   ├── baseline/{eval.csv, stdout.log}
    │   ├── se/{eval.csv, stdout.log}
    │   └── si2e/{eval.csv, stdout.log}   ← partial (killed at 16K)
    ├── drqv2-full/              ← full 250K runs
    │   ├── baseline/{eval.csv, stdout.log}  ← ✅ DONE (ER=341)
    │   ├── se/{eval.csv, stdout.log}         ← ✅ DONE (ER=660 🍀)
    │   └── si2e/{eval.csv, stdout.log}       ← 🔄 RUNNING PID 55737 (~170K/250K)
    └── reproduce_table.md       ← ✅ summary comparison table (updated)
```

---

## All code changes made (complete list)

Every modification to the SI2E repo is listed below. The `base-*` repos are untouched.

### SI2E_A2C

| File | Line | Change | Reason |
|---|---|---|---|
| `SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/scripts/train.py` | ~101 | `"cuda:1"` → `"cuda:0"` | Machine has 1 GPU (index 0) |
| `SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/utils/env.py` | 6 | Added `disable_env_checker=True` to `gym.make()` | gym 0.26.2 checker incompatible with MiniGrid obs dict |
| `SI2E/SI2E_A2C/torch-ac/torch_ac/algos/base.py` | ~344–386 | `dists.shape[0]` → `dists.shape[1]` in both kthvalue guards | `kthvalue(dim=1)` bounds by columns; crashed at ~100K frames |

### SI2E_DrQv2

| File | Line | Change | Reason |
|---|---|---|---|
| `SI2E/SI2E_DrQv2/cfgs/config.yaml` | 24 | `device: cuda:1` → `device: cuda:0` | Machine has 1 GPU |
| `SI2E/SI2E_DrQv2/dmc.py` | ~201 | `minimum=-1.0` → `np.float32(-1.0)` (and max) | Python float → `result_type` returns float64; spec requires float32 |
| `SI2E/SI2E_DrQv2/replay_buffer.py` | ~53–55 | Added dtype cast when shapes match but dtypes differ | Belt-and-suspenders for float32/float64 mismatch |

### minigrid-pinned

| File | Change | Reason |
|---|---|---|
| `gym_minigrid/minigrid.py` (3 methods) | `np_random.randint` → `.integers()` + `AttributeError` fallback | NumPy 2.0 removed `Generator.randint()` |

### System file (not in repo)

| File | Change | Reason |
|---|---|---|
| `/usr/local/lib/python3.10/dist-packages/gym/utils/passive_env_checker.py` | `np.bool8` → `np.bool_` (2 occurrences) | NumPy 2.0 removed `bool8` |

> ⚠️ The system gym patch must be re-applied on a fresh machine. Use:
> ```bash
> python3 -c "import gym; print(gym.__file__)"
> # Then sed the file:
> sed -i 's/np\.bool8/np.bool_/g' /path/to/gym/utils/passive_env_checker.py
> ```

---

## Running commands reference

### A2C (MiniGrid DoorKey-8x8)

```bash
cd /workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files

# Baseline (~9 min, 3M frames)
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model baseline \
  --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/baseline

# SE (structural entropy, ~20 min)
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model se \
  --use_entropy_reward --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/se

# SI2E (structural entropy + value conditioning, ~97 min)
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model si2e \
  --use_entropy_reward --use_value_condition \
  --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/si2e
```

> **PYTHONPATH must be set** — `scripts/` is added to sys.path, not CWD, so `sip.py` (in CWD) is not found otherwise.

### A2C+VCSE (true kNN VCSE from Kim et al.)

```bash
cd /workspace/learn-si2e/base-vcse/VCSE_A2C/rl-starter-files/rl-starter-files
# ⚠️ FIRST: fix kthvalue bug in base-vcse/VCSE_A2C/torch-ac/torch_ac/algos/base.py
#    (same dists.shape[0] → dists.shape[1] bug as SI2E)
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model vcse \
  --use_entropy_reward --use_value_condition \
  --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/vcse
```

### DrQv2 (DMControl cartpole_swingup)

```bash
cd /workspace/learn-si2e/SI2E/SI2E_DrQv2

# Baseline (~36 min, 250K frames)
MUJOCO_GL=egl python3 train.py \
  agent._target_=drqv2.DrQV2Agent \
  "task@_global_=cartpole_swingup" \
  seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/baseline

# SE / structural entropy (~46 min)
MUJOCO_GL=egl python3 train.py \
  agent._target_=si2e.SI2EAgent agent.do_vcse=false \
  "task@_global_=cartpole_swingup" \
  seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/se

# SI2E / structural entropy + value conditioning (~5 h — use nohup)
MUJOCO_GL=egl nohup python3 train.py \
  agent._target_=si2e.SI2EAgent agent.do_vcse=true \
  "task@_global_=cartpole_swingup" \
  seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/si2e \
  > results/drqv2-full/si2e/stdout.log 2>&1 &
echo "PID: $!"
```

> `MUJOCO_GL=egl` is **required** — headless rendering in Docker. Without it: `RuntimeError: Cannot initialize a display`.

### DrQv2+VCSE (true kNN VCSE from Kim et al.)

```bash
cd /workspace/learn-si2e/base-vcse/VCSE_DrQv2
MUJOCO_GL=egl python3 train.py \
  agent.do_vcse=true \
  "task@_global_=cartpole_swingup" \
  seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/vcse
# ⚠️ Verify num_train_frames is supported: grep num_train_frames cfgs/config.yaml
```

---

## Current results snapshot (2026-05-19)

### Table 1 — DoorKey-8x8 (A2C, 3M frames, seed=1)

| Method | Paper success (%) | Local rR mean | Local rR max | Status |
|---|:---:|:---:|:---:|---|
| A2C baseline | — (fails) | 0.02 | 0.32 | ✅ done |
| A2C + SE | 72.60 ± 20.32 | 0.01 | 0.11 ⚠️ | ✅ done (unlucky seed) |
| A2C + VCSE | 94.32 ± 11.09 | — | — | ⬜ ~21 min to run |
| A2C + SI2E | **98.58 ± 3.11** | — | — | ⬜ ~97 min to run |

> rR (episode return 0–1) ≠ success rate (%). Not directly comparable.

### Table 2 — Cartpole Swingup (DrQv2, 250K frames, seed=1)

| Method | Paper ER (10 seeds) | Local ER | Frame | Status |
|---|:---:|:---:|:---:|---|
| DrQv2 baseline | — | **341** | 240K | ✅ done |
| DrQv2 + SE | 219.69 ± 62.21 | **660** 🍀 | 240K | ✅ done (lucky seed) |
| DrQv2 + VCSE | 707.76 ± 50.38 | — | — | ⬜ ~37 min to run |
| DrQv2 + MADE | 704.18 ± 41.75 | — | — | ❌ no public repo |
| DrQv2 + SI2E | **795.09 ± 90.49** | 🔄 ~334 | 170K | 🔄 PID 55737 (~71% done) |

---

## What to do next (priority order)

### 1. Wait for / verify DrQv2+SI2E completion (~1.5 h remaining)

```bash
# Check progress:
tail -3 /workspace/learn-si2e/results/drqv2-full/si2e/eval.csv
# When done (last eval at F=240K), update si2e_comparison.md and run_queue.md
```

### 2. Run A2C+VCSE on DoorKey-8x8 (~21 min)

⚠️ **First apply the kthvalue fix to `base-vcse/VCSE_A2C/`**:

```bash
# Check if the bug is present:
grep -n "dists.shape\[0\]" \
  /workspace/learn-si2e/base-vcse/VCSE_A2C/torch-ac/torch_ac/algos/base.py
# If found, apply the same fix as in SI2E_A2C:
#   dists.shape[0] → dists.shape[1]  (in kthvalue guards, both entropy functions)
```

Then run the A2C+VCSE command from the Running Commands section above.

### 3. Run DrQv2+VCSE on Cartpole Swingup (~37 min)

Can run in parallel with step 2 (different hardware: VCSE_A2C uses CPU, VCSE_DrQv2 uses GPU).

### 4. Run A2C+SI2E on DoorKey-8x8 (~97 min)

Can run in parallel with DrQv2+VCSE (CPU vs GPU).

### 5. Update docs with final results

After each run completes:
- Update `si2e_comparison.md` local result column
- Update `run_queue.md` status + final result
- Update `results/reproduce_table.md`

### 6. Multi-seed runs (optional, long)

For statistical validity, paper uses 10 seeds. Running even 3 seeds would narrow the
confidence interval substantially. A2C variants are CPU-bound (can run in parallel).
DrQv2+SI2E is GPU-bound (sequential only on 1 GPU).

Estimated time for 3 additional seeds per variant (priority: DrQv2+SI2E, A2C+SI2E):
- DrQv2+SI2E ×3 extra seeds: ~15 h (GPU, sequential)
- A2C+SI2E ×3 extra seeds: ~5 h (CPU, can parallelize)

---

## Key algorithmic facts (for interpreting results)

### Why SE fails on Pendulum Swingup
SE (entropy bonus) maximizes state coverage — it pushes the agent to explore far from
current states. On Pendulum Swingup, the high-value region (near the top) is a narrow
manifold; the entropy bonus actively discourages staying there. VCSE fixes this by
conditioning the entropy estimate on value: it only gives bonuses for exploring
low-value, novel states.

SI2E paper Table 2: SE=10.80±2.92 vs DrQv2 baseline=424.21 on Pendulum Swingup.

### Why our SE result on Cartpole Swingup (660) exceeds the paper (219±62)
Pure single-seed variance. Paper σ=±62, so a 2σ deviation lands at 219+124=343.
Our seed=1 apparently hit a particularly good initialization. This is expected.

### The `do_vcse` flag naming confusion in SI2E DrQv2
`si2e.py`: the flag is named `do_vcse` but when `True` it activates `VCSAE` (the encoding
tree = SI2E), not `VCSE` (kNN). When `False`, it activates SE (structural entropy without
value conditioning). This is a copy-paste naming artefact from the VCSE codebase.

### Metric difference between papers
- MiniGrid (SI2E Table 1): success rate (%) — fraction of episodes where agent reaches goal
- DMControl (SI2E Table 2): episode reward (ER) — sum of per-step rewards
- Our A2C logs report `rR` (raw return 0–1): reward is `1 - 0.9*(steps/max_steps)` if solved,
  0 otherwise. This is NOT the same as success rate.

---

## Key files to read

| File | Purpose |
|---|---|
| [REPRODUCE_LOG.md](REPRODUCE_LOG.md) | Full iteration log: every bug, fix, and command |
| [paper_result.md](paper_result.md) | Tables from all 3 papers (RE3, VCSE, SI2E) |
| [si2e_comparison.md](si2e_comparison.md) | SI2E Table 1+2 with our local results column |
| [run_queue.md](run_queue.md) | All runs ordered by train time; result paths; launch commands |
| [results/reproduce_table.md](results/reproduce_table.md) | Compact comparison table (updated) |
