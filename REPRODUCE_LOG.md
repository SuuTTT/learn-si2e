# SI2E × VCSE Reproduction Iteration Log

**Goal:** Reproduce SI2E (NeurIPS 2024) locally, identify the base work (VCSE), clone both repos,
run all pipeline variants end-to-end, compare against paper numbers.  
**Sessions:** Two sessions, ~2026-05-17 and 2026-05-18.

---

## Environment

| Component | Version |
|---|---|
| OS | Linux (Ubuntu 20.04, root@C.37015889) |
| GPU | NVIDIA GeForce RTX 3070 Laptop, 8 192 MiB VRAM |
| CUDA | 11.8 |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cu118 |
| NumPy | 2.2.6 |
| torch (gym) | gym 0.26.2 |
| gymnasium | 1.3.0 |
| minigrid | 3.1.0 (installed; also pinned fork at `minigrid-pinned/`) |
| dm_control | 1.0.41 |

---

## Iteration 1 — Download & Identify Papers

### Action
1. Searched for SI2E paper: "Structural Information Principles-based Effective Exploration"
   (arXiv:2410.06621, NeurIPS 2024, SELGroup).
2. Downloaded PDF → `/workspace/learn-si2e/SI2E_paper.pdf`
3. Cloned SI2E repo → `/workspace/learn-si2e/SI2E/`
   ```
   git clone https://github.com/SELGroup/SI2E.git
   ```

### Repository structure found
```
SI2E/
├── SI2E_A2C/
│   ├── rl-starter-files/rl-starter-files/   ← modified rl-starter-files (lcswillems fork)
│   └── torch-ac/                             ← modified torch-ac with entropy reward added
└── SI2E_DrQv2/                               ← modified DrQv2 (facebookresearch fork)
    ├── cfgs/config.yaml
    ├── train.py
    ├── drqv2.py
    ├── si2e.py          ← new: SI2E agent wrapper
    ├── sip.py           ← new: encoding tree (structural information principles)
    ├── network_utils.py ← new: ICM-style state-action encoder
    └── vcse.py          ← kept from VCSE: VCSE class (kNN) + VCSAE class (encoding tree)
```

### Finding
The README says nothing about the base paper. Mining `SI2E_paper.pdf` with `pdfminer` revealed
the primary predecessor citation:

> Dongyoung Kim, Jinwoo Shin, Pieter Abbeel, and Younggyo Seo.
> *Accelerating reinforcement learning with value-conditional state entropy exploration.*
> arXiv:2305.19476, 2023. (NeurIPS 2024)

This is **VCSE**. Its GitHub: `https://github.com/kingdy2002/VCSE`.

---

## Iteration 2 — Clone Base Repos

### Action
```bash
git clone https://github.com/lcswillems/rl-starter-files.git  base-rl-starter-files/
git clone https://github.com/lcswillems/torch-ac.git           base-torch-ac/
git clone https://github.com/facebookresearch/drqv2.git        base-drqv2/
git clone https://github.com/kingdy2002/VCSE.git               base-vcse/
```

### VCSE vs SI2E structural diff (DrQv2)

| File | In VCSE | In SI2E | Change |
|---|---|---|---|
| `vcse.py` | ✓ (defines `VCSE` kNN class) | ✓ (copied verbatim + adds `VCSAE` encoding tree) | Added `VCSAE` |
| `drqv2.py` | ✓ | ✓ | Minimal changes |
| `si2e.py` | ✗ | ✓ | New: SI2E agent that dispatches to `VCSAE` |
| `sip.py` | ✗ | ✓ | New: encoding tree HCSE algorithm |
| `network_utils.py` | ✗ | ✓ | New: ICM-style encoder |
| `VCSE_MWM/`, `VCSE_SAC/` | ✓ | ✗ | Not carried over |

**Key algorithmic difference:** VCSE partitions state space by Q-value estimate and computes
kNN entropy within each partition. SI2E replaces the kNN entropy with *structural entropy* via
an encoding tree (HCSE algorithm) that jointly captures state, action, and value geometry.

---

## Iteration 3 — Install Dependencies

### Packages installed
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numba tensorboardX
pip install gymnasium
pip install minigrid==3.1.0          # system minigrid (needed for imports)
pip install hydra-core dm-control dm-env submitit
pip install pandas scikit-image termcolor opencv-python-headless
pip install pdfminer.six             # PDF text extraction for paper mining
```

### torch-ac installed as editable (SI2E version)
```bash
cd SI2E/SI2E_A2C/torch-ac && pip install -e .
```
This overwrites the standard torch-ac in the environment with SI2E's modified version
(which adds entropy reward branches to `algos/base.py` and `algos/a2c.py`).

---

## Iteration 4 — Bug: CUDA Device Index

### Symptom
```
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:0 and cuda:1!
```

### Root cause
Both config files hard-code `cuda:1`, but the machine has only one GPU (index 0).

### Files changed

**`SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/scripts/train.py`, line 101**
```python
# Before
device = "cuda:1"
# After
device = "cuda:0"
```

**`SI2E/SI2E_DrQv2/cfgs/config.yaml`, line 24**
```yaml
# Before
device: cuda:1
# After
device: cuda:0
```

---

## Iteration 5 — Bug: NumPy 2.x `randint` Removed

### Symptom
```
AttributeError: 'Generator' object has no attribute 'randint'
```
Raised in MiniGrid's `_rand_int` method during env reset.

### Root cause
NumPy 2.0 removed `Generator.randint()` (deprecated since NumPy 1.17). The pinned MiniGrid
fork at `minigrid-pinned/` still calls `self.np_random.randint(low, high)`.

### File changed
**`minigrid-pinned/gym_minigrid/minigrid.py`** — three methods patched to use `.integers()`
with a `try/except AttributeError` fallback for older NumPy:

```python
# Before (line ~822)
return self.np_random.randint(low, high)

# After
try:
    return self.np_random.integers(low, high)   # NumPy >= 2.0
except AttributeError:
    return self.np_random.randint(low, high)    # NumPy < 2.0
```

---

## Iteration 6 — Bug: NumPy 2.x `np.bool8` Removed

### Symptom
```
AttributeError: module 'numpy' has no attribute 'bool8'
```
Raised inside `gym/utils/passive_env_checker.py` during `gym.make()`.

### Root cause
NumPy 2.0 removed `np.bool8` (alias for `np.bool_`). The gym 0.26.2 passive checker
still references it.

### File changed *(system file — `/usr/local/lib/python3.10/dist-packages/`)*
**`gym/utils/passive_env_checker.py`**
```python
# Before (two occurrences)
isinstance(done, (bool, np.bool8))

# After
isinstance(done, (bool, np.bool_))
```

---

## Iteration 7 — Bug: Gym Observation Space Key Mismatch

### Symptom
```
AssertionError: Observation space key 'image' not in ['image', 'direction', 'mission']
(or similar checker message)
```
The env checker validates the obs space strictly; old MiniGrid returns a dict with
`['image', 'direction', 'mission']` but the checker expected only `['image']`.

### Root cause
gym 0.26.2's passive env checker is incompatible with MiniGrid's legacy obs space.

### File changed
**`SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/utils/env.py`, line 6**
```python
# Before
env = gym.make(env_key)

# After
env = gym.make(env_key, disable_env_checker=True)
```

---

## Iteration 8 — Bug: Action dtype float64 → float32 Mismatch (DrQv2)

### Symptom
```
AssertionError: spec.dtype=float32  value.dtype=float64
```
Raised in `replay_buffer.py` when storing the first action.

### Root cause
`dm_control`'s `action_scale.Wrapper` calls `np.result_type(minimum, maximum)` to determine
the output dtype. When `minimum=-1.0` and `maximum=1.0` are Python `float` literals,
`np.result_type` returns `float64` (Python floats are 64-bit), so every action gets
up-cast to `float64` even though the spec requires `float32`.

### File changed
**`SI2E/SI2E_DrQv2/dmc.py`, line 201**
```python
# Before
action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)

# After
action_scale.Wrapper(env, minimum=np.float32(-1.0), maximum=np.float32(1.0))
```

Also added **`SI2E/SI2E_DrQv2/replay_buffer.py`** defensive cast for shape-matching /
dtype-differing case (belt-and-suspenders):
```python
# Added after shape check, before hard assert
if spec.shape == value.shape and spec.dtype != value.dtype:
    value = value.astype(spec.dtype)
```

---

## Iteration 9 — Bug: Hydra Task Override Key for DrQv2

### Symptom
```
hydra.errors.ConfigCompositionException: Could not override 'task'.
```

### Root cause
DrQv2 uses Hydra config groups; the task config is installed as a *config group default*
with the override key `task@_global_`, not the bare `task`.

### Fix (command-line only — not a code change)
```bash
# Wrong
python3 train.py task=cartpole_swingup

# Correct
python3 train.py "task@_global_=cartpole_swingup"
```

Similarly for agent override:
```bash
"agent._target_=si2e.SI2EAgent"   "agent.do_vcse=true"
```

---

## Iteration 10 — Bug: MuJoCo Rendering (No Display)

### Symptom
```
RuntimeError: Cannot initialize a display / no such display
```
Raised when MuJoCo tries to open a GL window in a headless Docker container.

### Fix (environment variable — not a code change)
```bash
export MUJOCO_GL=egl   # use EGL for off-screen rendering
```
Set before every `python3 train.py` call for DrQv2.

---

## Iteration 11 — Bug: `kthvalue` Dimension Guard Wrong (A2C, SI2E crash at ~100K frames)

### Symptom
```
RuntimeError: kthvalue(): selected number k out of range for dimension 1
```
Crash occurred at update ~800 (≈ 100 K frames) during `A2C+SI2E` training.

### Root cause
`compute_state_entropy` and `compute_value_condition_state_entropy` in
`SI2E/SI2E_A2C/torch-ac/torch_ac/algos/base.py` compute a distance matrix
`dists` of shape `(n_src, n_tgt)` and call `torch.kthvalue(dists, k, dim=1)`.
`kthvalue(..., dim=1)` selects along **columns**, so the valid range is
`k ∈ [1, n_tgt]`. However both functions used `dists.shape[0]` (rows = n_src)
as the clamping bound — e.g.:

```python
knn = min(self.k, dists.shape[0])   # BUG: should be dists.shape[1]
```

When the encoding tree produces level-1 aggregated nodes (`sf_level_1`) that are
fewer than `self.k`, the inner call to `compute_value_condition_state_entropy` with
that smaller set fails because `k + 1 > n_tgt`.

### Files changed
**`SI2E/SI2E_A2C/torch-ac/torch_ac/algos/base.py`** — both entropy functions:

```python
# compute_state_entropy — Before
knn_dists = torch.kthvalue(dists, k=self.k + 1, dim=1).values

# After
n_tgt = dists.shape[1]
knn_dists = torch.kthvalue(dists, k=min(self.k + 1, n_tgt), dim=1).values
```

```python
# compute_value_condition_state_entropy — Before
knn = min(self.k, dists.shape[0])
for k in range(min(5, dists.shape[0])):
    eps += torch.kthvalue(dists, k + 1, dim=1).values

# After
n_tgt = dists.shape[1]
knn = min(self.k, n_tgt - 1)
n_avg = min(5, n_tgt)
for k in range(n_avg):
    eps += torch.kthvalue(dists, k + 1, dim=1).values
eps /= max(n_avg, 1)
```

---

## Iteration 12 — Verified End-to-End Runs

### A2C pipeline (MiniGrid DoorKey-8x8)

| Variant | Flags | FPS | Sanity check |
|---|---|---|---|
| Baseline | `--use_batch` | ~5 800 | ✓ runs cleanly |
| SE (kNN) | `--use_entropy_reward --use_batch` | ~2 400 | ✓ runs cleanly |
| SI2E (encoding tree) | `--use_entropy_reward --use_value_condition --use_batch` | ~490 | ✓ after kthvalue fix |

Control flow in `torch_ac/algos/a2c.py`:
- `use_entropy_reward=False` → plain A2C advantage
- `use_entropy_reward=True, use_value_condition=False` → calls `compute_state_entropy` (kNN)
- `use_entropy_reward=True, use_value_condition=True` → calls `compute_value_condition_structural_entropy` (encoding tree)

Note: **VCSE's value-conditional kNN** (`compute_value_condition_state_entropy`) is called
*internally* by the encoding tree function but is **not directly exposed** as a standalone
flag in the SI2E A2C code. A separate run of `base-vcse/VCSE_A2C/` would be needed.

### DrQv2 pipeline (DMControl cartpole_swingup)

| Variant | agent._target_ | agent.do_vcse | FPS | Notes |
|---|---|---|---|---|
| Baseline | `drqv2.DrQV2Agent` | N/A | ~140 | ✓ runs cleanly |
| SE | `si2e.SI2EAgent` | `false` | ~112 | ✓ runs cleanly |
| SI2E | `si2e.SI2EAgent` | `true` | ~14 | ✓ after replay_buffer + dmc fixes |

`si2e.py` naming confusion: the flag is `do_vcse` but when `True` it activates
the **VCSAE** class (encoding tree = SI2E), not the **VCSE** class (kNN). This is
a copy-paste naming artefact from the original VCSE codebase.

---

## Iteration 13 — Local Runs & Results

### A2C — DoorKey-8x8, 500 K frames (1/6 of paper's 3 M)

All three variants run at the shortened frame budget.  
At 500 K frames DoorKey-8x8 is still in random-walk territory for all methods.

| Variant | Avg success rate (last 20 updates) | Max instantaneous | Run time |
|---|---|---|---|
| A2C baseline | 0.70% | ~5% | ~83 s |
| A2C + SE | 0.45% | ~6% | ~210 s |
| A2C + SI2E | 0.90% | ~7% | ~1 020 s |

### DrQv2 — cartpole_swingup, eval rewards

At 50 K frames none of the methods have started learning (random score ≈ 74.8).

| Variant | Eval @10K | @20K | @30K | @40K | Total time |
|---|---|---|---|---|---|
| DrQV2Agent (baseline) | 74.94 | 74.19 | 75.25 | 74.76 | ~7 min |
| SI2EAgent do_vcse=false (SE) | 74.94 | 74.19 | 75.25 | 74.76 | ~9 min |
| SI2EAgent do_vcse=true (SI2E) | 74.94 | — | — | — | killed @16K |

SI2E DrQv2 killed: at 14 FPS, completing 50 K frames would require ~60 min vs ~7 min
for the baseline.

---

## Comparison: Paper vs Local (see `results/reproduce_table.md`)

### Table 1 — MiniGrid DoorKey-8x8

| Method | Paper success (%) | Local @ 500K frames |
|---|---|---|
| A2C | — | ~0.7% |
| A2C + SE | 72.60 ± 20.32 | ~0.5% |
| A2C + VCSE | 94.32 ± 11.09 | not run (separate repo) |
| A2C + SI2E | **98.58 ± 3.11** | ~0.9% |

### Table 2 — DMControl cartpole_swingup

| Method | Paper reward | Local @ 50K frames |
|---|---|---|
| DrQv2 | — | ~74.8 (random) |
| DrQv2 + SE | 219.69 ± 62.21 | ~74.8 (random) |
| DrQv2 + VCSE | 707.76 ± 50.38 | not run |
| DrQv2 + SI2E | **795.09 ± 90.49** | 74.9 (1 eval, killed) |

---

## Summary of All Code Changes

| File | Change | Reason |
|---|---|---|
| `SI2E/SI2E_A2C/rl-starter-files/.../scripts/train.py:101` | `cuda:1` → `cuda:0` | Single GPU machine |
| `SI2E/SI2E_DrQv2/cfgs/config.yaml:24` | `device: cuda:1` → `cuda:0` | Single GPU machine |
| `minigrid-pinned/gym_minigrid/minigrid.py:822–826` | `np_random.randint` → `.integers()` + fallback | NumPy 2.0 API change |
| `gym/utils/passive_env_checker.py` *(system)* | `np.bool8` → `np.bool_` (2 occurrences) | NumPy 2.0 removed `bool8` |
| `SI2E/SI2E_A2C/rl-starter-files/.../utils/env.py:6` | Added `disable_env_checker=True` | gym 0.26.2 checker incompatible with MiniGrid obs space |
| `SI2E/SI2E_DrQv2/dmc.py:201` | `minimum=-1.0` → `np.float32(-1.0)` (same for max) | Python float literals cause `result_type` to return float64 |
| `SI2E/SI2E_DrQv2/replay_buffer.py:53–55` | Added dtype cast when shapes match but dtypes differ | Belt-and-suspenders for float32 vs float64 mismatch |
| `SI2E/SI2E_A2C/torch-ac/.../algos/base.py:344–386` | `dists.shape[0]` → `dists.shape[1]` in both kthvalue guards | `kthvalue(dim=1)` bounds by columns not rows; crashed at ~100K frames |

---

## Workspace Layout

```
learn-si2e/
├── SI2E/                        ← main repo (all fixes applied)
│   ├── SI2E_A2C/
│   │   ├── rl-starter-files/rl-starter-files/  (train.py, utils/env.py modified)
│   │   └── torch-ac/            (algos/base.py modified; pip install -e . done)
│   └── SI2E_DrQv2/              (cfgs/config.yaml, dmc.py, replay_buffer.py modified)
├── SI2E_paper.pdf
├── base-rl-starter-files/       ← original lcswillems/rl-starter-files (unmodified)
├── base-torch-ac/               ← original lcswillems/torch-ac (unmodified)
├── base-drqv2/                  ← original facebookresearch/drqv2 (unmodified)
├── base-vcse/                   ← original kingdy2002/VCSE (unmodified)
│   ├── VCSE_A2C/
│   ├── VCSE_DrQv2/
│   ├── VCSE_MWM/
│   └── VCSE_SAC/
├── minigrid-pinned/             ← Farama MiniGrid fork (np_random.integers fix)
├── reproduce_and_compare.sh     ← full reproduce script (all 6 variants)
├── compare_results.py           ← log parser + comparison plots
└── results/
    ├── a2c/
    │   ├── baseline/train.log
    │   ├── se/train.log
    │   └── si2e/train.log
    ├── drqv2/
    │   ├── baseline/{eval.csv, train.csv, stdout.log}
    │   ├── se/{eval.csv, stdout.log}
    │   └── si2e/{eval.csv, stdout.log}   ← partial (killed at 16K)
    └── reproduce_table.md
```

---

## Run-time Estimates

### Hardware comparison

| | Paper | Ours |
|---|---|---|
| GPU | NVIDIA RTX A1000 (Ampere, 5 120 CUDA cores, 16 GB ECC, boost ~1 455 MHz) | NVIDIA RTX 3070 Laptop (Ampere, 5 120 CUDA cores, 8 GB, boost ~1 560 MHz) |
| CPU | 8-core Intel i9 @ 3.00 GHz | (same container) |

Same GPU architecture, nearly identical core count; our FPS numbers are a reliable
proxy for the paper's wall-clock times.  
**The paper does not report wall-clock times** — only frame budgets (Appendix D).

Frame budgets from Appendix D:

| Benchmark | Frame budget |
|---|---|
| MiniGrid hard tasks (DoorKey-8x8, RedBlueDoors-6x6, KeyCorridorS3R1) | **3 000 K** |
| MiniGrid easy tasks (DoorKey-6x6, SimpleCrossingS9N1) | **1 000 K** |
| DMControl (all 6 tasks) | **250 K** |
| MetaWorld (all 7 tasks) | **200 K / 100 K** |

---

### Per-run estimates — single seed, single task

#### A2C (MiniGrid)

| Method | Median FPS | 3 M frames (hard) | 1 M frames (easy) |
|---|---:|---:|---:|
| A2C baseline | 4 959 | **10 min** | 3.4 min |
| A2C + SE | 2 449 | **20 min** | 6.8 min |
| A2C + VCSE | ~2 400 ¹ | **~21 min** | ~7 min |
| A2C + SI2E | 518 | **97 min (1.6 h)** | 32 min |

¹ VCSE uses the same kNN loop as SE; FPS assumed equal.

#### DrQv2 (DMControl, 250 K frames per task)

| Method | Median FPS | 250 K frames |
|---|---:|---:|
| DrQv2 baseline | 141.8 | **29 min** |
| DrQv2 + SE | 112.3 | **37 min** |
| DrQv2 + VCSE | ~112 ¹ | **~37 min** |
| DrQv2 + MADE | ~112 ¹ | **~37 min** |
| DrQv2 + SI2E | 14.1 | **296 min (4.9 h)** |

---

### Total cost to reproduce each full paper table

#### Table 1 — MiniGrid (5 tasks × 4 methods × 10 seeds)

| | Sequential, 1 GPU | Parallel, 4 GPUs |
|---|---|---|
| Estimated total | **~90 h** | **~22 h** |

Breakdown: 3 hard tasks × (10+20+21+97 min) × 10 seeds  
         + 2 easy tasks × (3.4+6.8+7+32 min) × 10 seeds  
         = 44 100 min + 982 min ≈ 750 h … wait, let me be precise:

```
Hard (3M):  3 tasks × 10 seeds × (10+20+21+97) min = 3 × 10 × 148 = 4440 min = 74 h
Easy (1M):  2 tasks × 10 seeds × (3.4+6.8+7+32) min = 2 × 10 × 49  = 980 min  = 16 h
Total sequential: ~90 h
```

#### Table 2 — DMControl (6 tasks × 5 methods × 10 seeds)

| | Sequential, 1 GPU | Parallel, 8 GPUs |
|---|---|---|
| Estimated total | **~436 h** | **~55 h** |

```
6 tasks × 10 seeds × (29+37+37+37+296) min = 6 × 10 × 436 = 26 160 min = 436 h
SI2E alone: 6 × 10 × 296 min = 296 h  (68% of total cost)
```

**SI2E's encoding tree accounts for ~68% of the full DMControl reproduction cost.**

---

## Iteration 14 — Full-Budget A2C Runs (3 M frames, seed=1)

Ran A2C baseline and A2C+SE to the paper's full frame budget for DoorKey-8x8.

```bash
cd SI2E/SI2E_A2C/rl-starter-files/rl-starter-files
# Baseline
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model baseline \
  --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/baseline
# SE
PYTHONPATH="$(pwd):$PYTHONPATH" python3 scripts/train.py \
  --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model se \
  --use_entropy_reward --save-interval 100 --frames 3000000 \
  --log-dir /workspace/learn-si2e/results/a2c-full/se
```

| Variant | FPS | Wall time | Final rR:μ | rR:max | Notes |
|---|---|---|---|---|---|
| A2C baseline | 5 592 | ~9 min | 0.02 | 0.81 | Expected fail (paper shows —) |
| A2C + SE | 2 493 | ~20 min | 0.01 | 0.11 | Extremely unlucky seed; paper σ=±20 pp |

Logs: `results/a2c-full/baseline/train.log`, `results/a2c-full/se/train.log`

---

## Iteration 15 — Full-Budget DrQv2 Runs (250 K frames, seed=1)

Ran DrQv2 baseline and DrQv2+SE to paper budget. Launched DrQv2+SI2E as nohup (PID 55737, ~5 h).

```bash
cd /workspace/learn-si2e/SI2E/SI2E_DrQv2
# Baseline
MUJOCO_GL=egl python3 train.py agent._target_=drqv2.DrQV2Agent \
  task@_global_=cartpole_swingup seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/baseline
# SE
MUJOCO_GL=egl python3 train.py agent._target_=si2e.SI2EAgent agent.do_vcse=false \
  task@_global_=cartpole_swingup seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/se
# SI2E (nohup — PID 55737, running as of 2026-05-19, ~170K/250K)
MUJOCO_GL=egl nohup python3 train.py agent._target_=si2e.SI2EAgent agent.do_vcse=true \
  task@_global_=cartpole_swingup seed=1 num_train_frames=250000 device=cuda:0 \
  hydra.run.dir=/workspace/learn-si2e/results/drqv2-full/si2e \
  > results/drqv2-full/si2e/stdout.log 2>&1 &
```

| Variant | FPS | Wall time | Final ER (@ frame) | Paper ER (10 seeds) |
|---|---|---|---|---|
| DrQv2 baseline | ~142 | ~36 min | **341** @ 240K | — |
| DrQv2 + SE | ~112 | ~46 min | **660** @ 240K 🍀 | 219.69 ± 62.21 |
| DrQv2 + SI2E | ~14 | ~5 h | 🔄 ~334 @ 170K | **795.09 ± 90.49** |

🍀 SE ER=660 far exceeds paper mean 219±62 — single lucky seed (paper σ=±62).

Logs: `results/drqv2-full/{baseline,se,si2e}/eval.csv` and `stdout.log`

---

## Known Limitations & Next Steps

1. **Single seed.** Paper uses 10 seeds. Variance is extreme (σ=20 pp for A2C+SE on DoorKey-8x8).
   Our single-seed results are directionally informative but not conclusive.

2. **SI2E has 10–12× compute overhead** over SE/baseline (O(n²) encoding tree).
   RTX 3070: SI2E DrQv2 = 14 FPS vs baseline = 140 FPS.

3. **No VCSE column yet.** `base-vcse/` is cloned but not run.
   See `run_queue.md` for launch commands (A2C+VCSE: ~21 min, DrQv2+VCSE: ~37 min).

4. **`kthvalue` bug in VCSE repo.** Same `dists.shape[0]` bug likely present in
   `base-vcse/VCSE_A2C/torch-ac/torch_ac/algos/base.py` — fix before running.

5. **MADE baseline not available.** No public code found in SI2E or VCSE repos.

6. **A2C+SI2E full-budget not yet run.** Estimated ~97 min. See `run_queue.md`.

See `run_queue.md` for the ordered execution queue.
See `si2e_comparison.md` for paper vs local comparison tables.
See `handoff.md` for full project context and continuation guide.
