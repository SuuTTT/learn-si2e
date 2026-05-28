#!/usr/bin/env python3
"""
Analyze experiment results and generate paper-ready tables + plots.

Usage:
  python3 analyze_results.py                  # summary tables
  python3 analyze_results.py --plot           # also generate learning curve plots
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

BASE = '/workspace/learn-si2e'

# ──────────────────────────────────────────────
# 1. Original SI2E baselines (from earlier sessions)
# ──────────────────────────────────────────────
def load_old_csv(path, env_name, filter_method=None):
    """Load old-style summary CSV (no env column), tag with env_name."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filter_method and row.get('method') != filter_method:
                continue
            try:
                rows.append({
                    'method': row.get('method', '?'),
                    'env': env_name,
                    'env_short': env_name.replace('MiniGrid-', '').replace('-v0', ''),
                    'seed': row.get('seed', '?'),
                    'success_rate_pct': float(row['success_rate_pct']),
                    'fps': float('nan'),
                    'source': 'original',
                })
            except (ValueError, KeyError):
                pass
    return rows

# Load original SI2E results (method='si2e')
original_rows = []
original_rows += load_old_csv(
    f'{BASE}/results/a2c-multiseed/summary.csv',
    'MiniGrid-DoorKey-8x8-v0', filter_method='si2e')
original_rows += load_old_csv(
    f'{BASE}/results/keycorridor/summary.csv',
    'MiniGrid-KeyCorridorS3R2-v0', filter_method='si2e')
original_rows += load_old_csv(
    f'{BASE}/results/redbluedoors/summary.csv',
    'MiniGrid-RedBlueDoors-6x6-v0', filter_method='si2e')

# Also load VCSE as another baseline
vcse_rows = []
vcse_rows += load_old_csv(
    f'{BASE}/results/a2c-multiseed/summary.csv',
    'MiniGrid-DoorKey-8x8-v0', filter_method='vcse')
vcse_rows += load_old_csv(
    f'{BASE}/results/keycorridor/summary.csv',
    'MiniGrid-KeyCorridorS3R2-v0', filter_method='vcse')
for r in vcse_rows:
    r['method'] = 'vcse-original'

# ──────────────────────────────────────────────
# 2. New experiment results
# ──────────────────────────────────────────────
NEW_RESULTS = {
    'fast-si2e':     f'{BASE}/results/fast-si2e/summary.csv',
    'adaptive-beta': f'{BASE}/results/adaptive-beta/summary.csv',
    'ppo-si2e':      f'{BASE}/results/ppo-si2e/summary.csv',
}

def load_new_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                env = row.get('env', '')
                rows.append({
                    'method': row.get('method', '?'),
                    'env': env,
                    'env_short': env.replace('MiniGrid-', '').replace('-v0', ''),
                    'seed': row.get('seed', '?'),
                    'success_rate_pct': float(row['success_rate_pct']),
                    'fps': float(row.get('fps', 'nan') or 'nan'),
                    'source': 'new',
                })
            except (ValueError, KeyError):
                pass
    return rows

new_rows = []
for label, path in NEW_RESULTS.items():
    new_rows += load_new_csv(path)

all_rows = original_rows + vcse_rows + new_rows

# ──────────────────────────────────────────────
# 3. Aggregate by method × env
# ──────────────────────────────────────────────
data = defaultdict(lambda: defaultdict(list))
fps_data = defaultdict(lambda: defaultdict(list))

for r in all_rows:
    m = r['method']
    e = r['env_short']
    sr = r['success_rate_pct']
    fp = r['fps']
    if not np.isnan(sr):
        data[e][m].append(sr)
    if not np.isnan(fp):
        fps_data[e][m].append(fp)

def fmt(vals, bold_threshold=None):
    if not vals:
        return '—'
    mean = np.mean(vals)
    std = np.std(vals)
    s = f'{mean:.1f}% ± {std:.1f}  (N={len(vals)})'
    if bold_threshold and mean >= bold_threshold:
        return f'**{s}**'
    return s

# ──────────────────────────────────────────────
# 4. Print comparison table
# ──────────────────────────────────────────────
print('\n' + '='*75)
print('RESULTS SUMMARY')
print('='*75)

method_order = [
    'vcse-original', 'si2e',          # original baselines
    'si2e-fixed',    'si2e-adaptive',  # adaptive-beta batch
    'fast-si2e',     'ppo-fast-si2e',  # fast-si2e batch
    'ppo-si2e', 'ppo-si2e-adaptive',   # PPO variants
]

envs = sorted(data.keys())
for env in envs:
    print(f'\n{env}:')
    methods = sorted(data[env].keys(),
                     key=lambda m: method_order.index(m) if m in method_order else 99)
    for m in methods:
        vals = data[env][m]
        tag = ' [orig]' if m in ('si2e', 'vcse-original') else ''
        print(f'  {m:<28}{tag} {fmt(vals)}')

# ──────────────────────────────────────────────
# 5. Concise comparison vs SI2E baseline
# ──────────────────────────────────────────────
print('\n' + '='*75)
print('VS. ORIGINAL SI2E BASELINE')
print('='*75)
si2e_dk  = np.mean(data['DoorKey-8x8'].get('si2e', []))
si2e_kc  = np.mean(data['KeyCorridorS3R2'].get('si2e', []))
si2e_rbd = np.mean(data['RedBlueDoors-6x6'].get('si2e', []))

compare_methods = ['fast-si2e', 'si2e-fixed', 'si2e-adaptive',
                   'ppo-fast-si2e', 'ppo-si2e', 'ppo-si2e-adaptive']
header = f"{'Method':<28} {'DK-8x8':>18} {'KC-S3R2':>18} {'RedBlue-6x6':>18}"
print(header)
print('-' * len(header))

def delta(vals, baseline):
    if not vals or np.isnan(baseline):
        return '—'
    m = np.mean(vals)
    d = m - baseline
    s = f'{m:.1f}% ({d:+.1f})'
    return s

for method in compare_methods:
    dk  = delta(data['DoorKey-8x8'].get(method, []),     si2e_dk)
    kc  = delta(data['KeyCorridorS3R2'].get(method, []), si2e_kc)
    rbd = delta(data['RedBlueDoors-6x6'].get(method, []), si2e_rbd)
    print(f'  {method:<26} {dk:>18} {kc:>18} {rbd:>18}')

print(f'\nOriginal SI2E: DK-8x8={si2e_dk:.1f}% KC={si2e_kc:.1f}% RBD={si2e_rbd:.1f}%')

# ──────────────────────────────────────────────
# 6. FPS comparison
# ──────────────────────────────────────────────
if any(any(v for v in fps_data[e].values()) for e in fps_data):
    print('\n' + '='*75)
    print('FPS (from training logs)')
    print('='*75)
    all_fps = defaultdict(list)
    for env in fps_data:
        for m, vals in fps_data[env].items():
            all_fps[m].extend(vals)
    for m, vals in sorted(all_fps.items()):
        if vals:
            print(f'  {m:<28} {np.mean(vals):.0f} ± {np.std(vals):.0f} FPS  (N={len(vals)})')

# ──────────────────────────────────────────────
# 7. Paper-ready LaTeX table
# ──────────────────────────────────────────────
print('\n' + '='*75)
print('LATEX TABLE')
print('='*75)

env_list   = ['DoorKey-8x8', 'KeyCorridorS3R2', 'RedBlueDoors-6x6']
env_header = ['DoorKey-8x8', 'KeyCorridorS3R2', 'RedBlueDoors-6x6']

method_display = {
    'si2e':               'SI2E (original)',
    'fast-si2e':          r'FastSI2E (k-means, ours)',
    'si2e-fixed':         r'FastSI2E + fixed-$\beta$ (ours)',
    'si2e-adaptive':      r'FastSI2E + adaptive-$\beta$ (ours)',
    'vcse-original':      'VCSE (original)',
    'ppo-si2e':           'PPO-SI2E',
    'ppo-fast-si2e':      'PPO-FastSI2E',
    'ppo-si2e-adaptive':  r'PPO-FastSI2E + adaptive-$\beta$',
}

print(r'\begin{table}[h]')
print(r'\centering')
print(r'\begin{tabular}{l' + 'c'*len(env_list) + '}')
print(r'\toprule')
print('Method & ' + ' & '.join(env_header) + r' \\')
print(r'\midrule')
for m, display in method_display.items():
    cells = []
    for env in env_list:
        vals = data[env].get(m, [])
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            cells.append(f'${mean:.1f} \\pm {std:.1f}$')
        else:
            cells.append('—')
    print(f'{display} & ' + ' & '.join(cells) + r' \\')
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\caption{Success rate (\%) mean $\pm$ std across seeds.}')
print(r'\end{table}')

# ──────────────────────────────────────────────
# 8. Learning curves (optional)
# ──────────────────────────────────────────────
if args.plot:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n[WARN] matplotlib not available')
        sys.exit(0)

    STORAGE = f'{BASE}/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/storage'

    def load_log_csv(model_name):
        path = os.path.join(STORAGE, model_name, 'log.csv')
        if not os.path.exists(path):
            return None, None
        frames, returns = [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frames.append(int(row['frames']))
                    returns.append(float(row['return_mean']))
                except (ValueError, KeyError):
                    pass
        if not frames:
            return None, None
        return np.array(frames), np.array(returns)

    def smooth(y, window=5):
        if len(y) < window:
            return y
        return np.convolve(y, np.ones(window)/window, mode='same')

    def plot_method(ax, prefix, env_short, color, label, n_seeds=3):
        curves = []
        for seed in range(1, n_seeds + 1):
            fx, ry = load_log_csv(f'{prefix}-{env_short}-s{seed}')
            if fx is not None and len(fx) > 5:
                curves.append((fx, smooth(ry)))
        if not curves:
            return
        max_f = max(c[0][-1] for c in curves)
        x_grid = np.linspace(0, max_f, 200)
        interped = [np.interp(x_grid, fx, ry) for fx, ry in curves]
        mean_c = np.mean(interped, axis=0)
        std_c = np.std(interped, axis=0)
        ax.plot(x_grid / 1e6, mean_c, color=color, label=label)
        ax.fill_between(x_grid / 1e6, mean_c - std_c, mean_c + std_c,
                        alpha=0.15, color=color)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_specs = {
        'fastse-si2e':       ('blue',  'A2C-SI2E'),
        'fastse-fast-si2e':  ('green', 'A2C-FastSI2E'),
    }
    for ax, env_short in zip(axes, ['DoorKey-8x8', 'KeyCorridorS3R2']):
        for prefix, (color, label) in plot_specs.items():
            plot_method(ax, prefix, env_short, color, label)
        ax.set_xlabel('Frames (M)')
        ax.set_ylabel('Mean Return')
        ax.set_title(env_short)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(f'{BASE}/results', exist_ok=True)
    out_path = f'{BASE}/results/learning_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nLearning curve saved: {out_path}')
    plt.close()

print('\nDone.')
