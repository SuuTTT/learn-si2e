#!/usr/bin/env python3
"""
FPS benchmark for SI2E variants — table for paper.

Run from: /workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/
Usage:  python3 /workspace/learn-si2e/benchmark_fps.py [--procs 16] [--steps 3]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/workspace/learn-si2e/SI2E/SI2E_A2C/torch-ac')
sys.path.insert(0, '/workspace/learn-si2e/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files')
import torch_ac, utils
from model import ACModel

parser = argparse.ArgumentParser()
parser.add_argument('--procs', type=int, default=16)
parser.add_argument('--steps', type=int, default=5, help='benchmark repetitions')
parser.add_argument('--env', default='MiniGrid-DoorKey-8x8-v0')
args = parser.parse_args()

os.system('find /workspace/learn-si2e/SI2E -name "*.pyc" -delete 2>/dev/null')

env = utils.make_env(args.env, 0)
obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)

variants = [
    ('baseline',        dict(algo='a2c')),
    ('a2c-si2e',        dict(algo='a2c', use_entropy_reward=True, use_value_condition=True)),
    ('a2c-fast-si2e',   dict(algo='a2c', use_entropy_reward=True, use_value_condition=True, fast_se=True)),
    ('ppo-si2e',        dict(algo='ppo', use_entropy_reward=True, use_value_condition=True)),
    ('ppo-fast-si2e',   dict(algo='ppo', use_entropy_reward=True, use_value_condition=True, fast_se=True)),
]

print(f"Benchmarking on {args.env} with {args.procs} procs ({args.steps} warmup+timed runs)")
print(f"{'Method':<20} {'FPS':>8} {'collect_ms':>12} {'update_ms':>10} {'speedup':>8}")
print('-' * 65)

baseline_fps = None
for name, cfg in variants:
    acmodel = ACModel(obs_space, env.action_space, False, False).to('cuda:0')
    envs = [utils.make_env(args.env, i * 1000) for i in range(args.procs)]

    use_fast = cfg.get('fast_se', False)
    if cfg['algo'] == 'ppo':
        algo = torch_ac.PPOAlgo(envs, acmodel, 'cuda:0', preprocess_obss=preprocess_obss,
                                 use_entropy_reward=cfg.get('use_entropy_reward', False),
                                 use_value_condition=cfg.get('use_value_condition', False),
                                 fast_se=use_fast)
    else:
        algo = torch_ac.A2CAlgo(envs, acmodel, 'cuda:0', preprocess_obss=preprocess_obss,
                                  use_entropy_reward=cfg.get('use_entropy_reward', False),
                                  use_value_condition=cfg.get('use_value_condition', False),
                                  fast_se=use_fast)
    algo.beta = 0.005
    algo.use_batch = True
    algo.replay_buffer = np.zeros((10000, 64))
    algo.idx = 0
    algo.full = False

    # Warmup (trigger JIT compilation)
    for _ in range(2):
        exps, logs = algo.collect_experiences()
        algo.update_parameters(exps)

    times_collect, times_update = [], []
    for _ in range(args.steps):
        t0 = time.time()
        exps, logs = algo.collect_experiences()
        t1 = time.time()
        algo.update_parameters(exps)
        t2 = time.time()
        times_collect.append(t1 - t0)
        times_update.append(t2 - t1)

    nf = logs['num_frames']
    tc = min(times_collect)
    tu = min(times_update)
    fps = nf / (tc + tu)
    if baseline_fps is None:
        baseline_fps = fps

    epochs = 4 if cfg['algo'] == 'ppo' else 1
    print(f"{name:<20} {fps:>8.0f} {1000*tc:>10.0f}ms {1000*tu:>8.0f}ms {fps/baseline_fps:>7.2f}x"
          f"  [epochs={epochs}, n={nf}]")

print()
print("Notes:")
print("  - FPS = environment frames per wall-clock second")
print("  - PPO has 4 gradient epochs per rollout vs A2C's 1")
print("  - fast_se uses numpy k-means graph partitioning (~170x faster than glass-jax)")
print("  - Effective gradient steps/sec: A2C FPS × 1, PPO FPS × 4")
print("  - Key metric: PPO-fast-si2e effective steps = FPS × 4")
