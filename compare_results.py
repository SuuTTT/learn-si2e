#!/usr/bin/env python3
"""
compare_results.py — Parse training logs from reproduce_and_compare.sh
and produce tables + plots comparing A2C / VCSE / SI2E.

Usage:
    python3 compare_results.py          # auto-detect results/
    python3 compare_results.py --a2c    # only A2C/MiniGrid
    python3 compare_results.py --drqv2  # only DrQv2/DMControl
"""

import argparse
import re
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Optional: matplotlib for plotting
# ------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not installed — skipping plots (pip install matplotlib)")

RESULTS = Path(__file__).parent / "results"

# ========================
# A2C / MiniGrid parsing
# ========================
# Log lines look like:
#   U 1 | F 000080 | FPS 0403 | D 1 | rR:μσmM 0.00 0.00 0.00 0.00 | ...
A2C_LOG_RE = re.compile(
    r"F\s+(\d+)\s+\|.*rR:μσmM\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

def parse_a2c_log(path):
    frames, mean_returns = [], []
    for line in open(path):
        m = A2C_LOG_RE.search(line)
        if m:
            frames.append(int(m.group(1)))
            mean_returns.append(float(m.group(2)))
    return frames, mean_returns

# ========================
# DrQv2 / DMControl parsing
# ========================
# Log lines: "episode_reward  123.45  frame  50000  episode  10 ..."
DRQ_LOG_RE = re.compile(r"episode_reward\s+([\d.]+).*frame\s+(\d+)")

def parse_drqv2_log(path):
    frames, rewards = [], []
    for line in open(path):
        m = DRQ_LOG_RE.search(line)
        if m:
            rewards.append(float(m.group(1)))
            frames.append(int(m.group(2)))
    return frames, rewards

# ========================
# Summary helpers
# ========================
def summary(name, frames, values):
    if not values:
        print(f"  {name:30s}: no data parsed")
        return
    final = values[-1]
    peak  = max(values)
    print(f"  {name:30s}: final={final:.3f}  peak={peak:.3f}  ({len(values)} checkpoints)")

def compare_a2c(seed=1):
    print("\n" + "="*60)
    print("  A2C / MiniGrid — DoorKey-8x8")
    print("="*60)
    base_dir = RESULTS / "a2c"
    methods = {
        "A2C (baseline)": f"run_a2c_seed{seed}.log",
        "A2C + VCSE":     f"run_vcse_seed{seed}.log",
        "A2C + SI2E":     f"run_si2e_seed{seed}.log",
    }
    all_data = {}
    for name, logfile in methods.items():
        path = base_dir / logfile
        if not path.exists():
            print(f"  [missing] {path}")
            continue
        frames, values = parse_a2c_log(path)
        all_data[name] = (frames, values)
        summary(name, frames, values)

    if HAS_PLOT and all_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, (fr, vals) in all_data.items():
            ax.plot(fr, vals, label=name)
        ax.set_xlabel("Environment frames")
        ax.set_ylabel("Mean return per episode")
        ax.set_title("A2C / MiniGrid DoorKey-8x8 — method comparison")
        ax.legend()
        out = RESULTS / "a2c_comparison.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\n  Plot saved: {out}")
        plt.close(fig)

def compare_drqv2(seed=1):
    print("\n" + "="*60)
    print("  DrQv2 / DMControl — cartpole_swingup")
    print("="*60)
    base_dir = RESULTS / "drqv2"
    methods = {
        "DrQv2 (baseline)": f"run_drqv2_seed{seed}.log",
        "DrQv2 + VCSE":     f"run_vcse_seed{seed}.log",
        "DrQv2 + SI2E":     f"run_si2e_seed{seed}.log",
    }
    all_data = {}
    for name, logfile in methods.items():
        path = base_dir / logfile
        if not path.exists():
            print(f"  [missing] {path}")
            continue
        frames, values = parse_drqv2_log(path)
        all_data[name] = (frames, values)
        summary(name, frames, values)

    if HAS_PLOT and all_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, (fr, vals) in all_data.items():
            ax.plot(fr, vals, label=name)
        ax.set_xlabel("Environment frames")
        ax.set_ylabel("Episode reward")
        ax.set_title("DrQv2 / DMControl cartpole_swingup — method comparison")
        ax.legend()
        out = RESULTS / "drqv2_comparison.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\n  Plot saved: {out}")
        plt.close(fig)

# ========================
# Diff of base vs SI2E
# ========================
def print_key_diffs():
    print("\n" + "="*60)
    print("  Key differences: base repo → SI2E (what changed)")
    print("="*60)
    print("""
  A2C branch (torch-ac):
    base:  torch_ac/algos/base.py  — standard A2C collect/update
    SI2E:  Same file + sip.py imported for encoding-tree reward
           Adds random_encoder, replay_buffer, value-conditional
           structural entropy as intrinsic bonus (β·r_intrinsic)

    Flags controlling the change:
      --use_entropy_reward   enable intrinsic reward
      --use_value_condition  use value-conditional grouping (VCSE-style)
      --use_batch            use encoding-tree partitioning (SI2E only)

  DrQv2 branch (SI2E_DrQv2/):
    base:  drqv2.py   — standard DrQv2 agent
    VCSE:  vcse.py    — DrQv2 + k-NN value-conditional entropy bonus
    SI2E:  si2e.py    — DrQv2 + structural encoding-tree entropy bonus
           (ICM encoder + bipartite graph + HCSE tree + kNN on communities)

    The ONLY change between VCSE and SI2E is the intrinsic reward:
      VCSE: r_i = H(V0) estimated with kNN
      SI2E: r_i = H(V0) - H(V1), where V1 = communities from encoding tree
""")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2c",   action="store_true")
    ap.add_argument("--drqv2", action="store_true")
    ap.add_argument("--seed",  type=int, default=1)
    ap.add_argument("--diffs", action="store_true", help="Print architectural diffs")
    args = ap.parse_args()

    if args.diffs or (not args.a2c and not args.drqv2):
        print_key_diffs()

    if args.a2c or (not args.a2c and not args.drqv2):
        compare_a2c(args.seed)

    if args.drqv2 or (not args.a2c and not args.drqv2):
        compare_drqv2(args.seed)
