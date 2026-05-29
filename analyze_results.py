#!/usr/bin/env python3
"""
Analyze all SI2E experiment results and generate paper-ready tables + plots.

Usage:
  python3 analyze_results.py           # summary tables + LaTeX
  python3 analyze_results.py --plot    # also generate learning curve plots
  python3 analyze_results.py --fps     # also run FPS benchmark (slow)
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true')
parser.add_argument('--fps',  action='store_true')
args = parser.parse_args()

BASE    = '/workspace/learn-si2e'
STORAGE = f'{BASE}/SI2E/SI2E_A2C/rl-starter-files/rl-starter-files/storage'

# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────
def load_csv(path, env_override=None, method_filter=None, method_rename=None):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                m = row.get('method', '?')
                if method_filter and m != method_filter:
                    continue
                if method_rename:
                    m = method_rename.get(m, m)
                env = env_override or row.get('env', '')
                sr  = float(row['success_rate_pct'])
                fps_raw = row.get('fps', '') or ''
                try:
                    fps = float(fps_raw)
                except ValueError:
                    fps = float('nan')
                rows.append({'method': m, 'env': env, 'seed': row.get('seed','?'),
                             'sr': sr, 'fps': fps})
            except (ValueError, KeyError):
                pass
    return rows

def dedup(rows):
    """Keep only the last entry per (env, method, seed) — later entries are from longer runs."""
    seen = {}
    for r in rows:
        key = (r['env'], r['method'], r['seed'])
        seen[key] = r  # overwrite with later (longer-run) value
    return list(seen.values())

def agg(rows):
    """Aggregate list of rows → {env_short: {method: [sr,...]}, fps: ...}"""
    rows = dedup(rows)
    sr_data  = defaultdict(lambda: defaultdict(list))
    fps_data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        e = r['env'].replace('MiniGrid-','').replace('-v0','')
        sr_data[e][r['method']].append(r['sr'])
        if not np.isnan(r['fps']):
            fps_data[e][r['method']].append(r['fps'])
    return sr_data, fps_data

def fmt(vals, bold=False):
    if not vals:
        return '—'
    m, s = np.mean(vals), np.std(vals)
    tag = '**' if bold else ''
    return f'{tag}{m:.1f}%±{s:.1f} (N={len(vals)}){tag}'

def latex_cell(vals):
    if not vals:
        return '—'
    m, s = np.mean(vals), np.std(vals)
    return f'${m:.1f}\\pm{s:.1f}$'

# ──────────────────────────────────────────────────────────────────
# 1. Load all result sources
# ──────────────────────────────────────────────────────────────────
all_rows = []

# (a) Original baselines from older summary CSVs
for env_name, path, filt in [
    ('MiniGrid-DoorKey-8x8-v0',      f'{BASE}/results/a2c-multiseed/summary.csv', None),
    ('MiniGrid-KeyCorridorS3R2-v0',  f'{BASE}/results/keycorridor/summary.csv',   None),
    ('MiniGrid-RedBlueDoors-6x6-v0', f'{BASE}/results/redbluedoors/summary.csv',  None),
]:
    all_rows += load_csv(path, env_override=env_name,
                         method_rename={'vcse': 'vcse-original'})

# (b) Fast-SI2E (kmeans) — exclude 'si2e' rows (1M-frame incomplete, overridden by (a))
for fsi2e_method in ('fast-si2e', 'ppo-fast-si2e'):
    all_rows += load_csv(f'{BASE}/results/fast-si2e/summary.csv',
                         method_filter=fsi2e_method)
# Extra seeds written by batch_phase2 section (A) use method name 'fastse-fast-si2e'
all_rows += load_csv(f'{BASE}/results/fast-si2e/summary.csv',
                     method_filter='fastse-fast-si2e',
                     method_rename={'fastse-fast-si2e': 'fast-si2e'})

# (c) Adaptive-beta experiments
all_rows += load_csv(f'{BASE}/results/adaptive-beta/summary.csv')

# (d) PPO-SI2E
all_rows += load_csv(f'{BASE}/results/ppo-si2e/summary.csv')

# (e) Clustering methods (leiden, infomap)
all_rows += load_csv(f'{BASE}/results/clustering-methods/summary.csv')

# (f) Phase 2 (KC extra seeds, RBD, fast+adaptive)
all_rows += load_csv(f'{BASE}/results/phase2/summary.csv',
                     method_rename={
                         'fastse-rbd-fast-si2e': 'rbd-fast-si2e',
                         'fastse-adapt-kc':       'fast-si2e-adaptive',
                         'fastse-adapt-rbd':      'fast-si2e-adaptive',
                     })
# Phase-2 extra KC seeds — normalise naming to 'fast-si2e' so they pool with existing
all_rows += load_csv(f'{BASE}/results/phase2/summary.csv',
                     method_filter='fastse-fast-si2e',
                     method_rename={'fastse-fast-si2e': 'fast-si2e'})

# (g) Ablations
abl_rows = []
abl_path = f'{BASE}/results/ablations/summary.csv'
if os.path.exists(abl_path):
    with open(abl_path) as f:
        for row in csv.DictReader(f):
            try:
                abl_rows.append({
                    'ablation': row['ablation'],
                    'seed': row['seed'],
                    'sr': float(row['success_rate_pct']),
                    'frames': int(row.get('frames', 0)),
                })
            except (ValueError, KeyError):
                pass

sr_data, fps_data = agg(all_rows)

# Inject SI2E FPS directly from fast-si2e summary (SR values there are 1M-frame incomplete
# so were not included in all_rows, but FPS values are valid)
for fps_row in load_csv(f'{BASE}/results/fast-si2e/summary.csv', method_filter='si2e'):
    if not np.isnan(fps_row['fps']):
        e = fps_row['env'].replace('MiniGrid-','').replace('-v0','')
        fps_data[e]['si2e'].append(fps_row['fps'])

# ──────────────────────────────────────────────────────────────────
# 2. Main comparison table
# ──────────────────────────────────────────────────────────────────
print('\n' + '='*80)
print('FULL RESULTS TABLE')
print('='*80)

METHOD_ORDER = [
    'vcse-original', 'si2e',
    'fast-si2e', 'ppo-fast-si2e',
    'si2e-fixed', 'si2e-adaptive',
    'fast-si2e-adaptive',
    'leiden-si2e', 'infomap-si2e',
    'rbd-fast-si2e',
    'ppo-si2e', 'ppo-si2e-adaptive',
]

ENV_ORDER = ['DoorKey-8x8', 'KeyCorridorS3R2', 'RedBlueDoors-6x6']

# Collect all seen envs + methods
all_envs    = [e for e in ENV_ORDER if e in sr_data]
all_methods = sorted(sr_data.get(all_envs[0] if all_envs else '', {}).keys()|
                     {m for e in all_envs for m in sr_data.get(e,{}).keys()},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

header = f"{'Method':<32}" + "".join(f"  {e:>20}" for e in all_envs)
print(header)
print('-'*len(header))
for m in all_methods:
    line = f'{m:<32}'
    for e in all_envs:
        vals = sr_data[e].get(m, [])
        line += f'  {fmt(vals):>20}'
    print(line)

# ──────────────────────────────────────────────────────────────────
# 3. SI2E baseline deltas
# ──────────────────────────────────────────────────────────────────
print('\n' + '='*80)
print('DELTA VS. ORIGINAL SI2E')
print('='*80)
si2e_means = {e: np.mean(sr_data[e].get('si2e', [float('nan')])) for e in all_envs}

for m in [m for m in all_methods if m not in ('vcse-original','si2e')]:
    parts = []
    for e in all_envs:
        vals = sr_data[e].get(m, [])
        base = si2e_means[e]
        if vals and not np.isnan(base):
            delta = np.mean(vals) - base
            parts.append(f'{e}: {np.mean(vals):.1f}% ({delta:+.1f})')
        elif vals:
            parts.append(f'{e}: {np.mean(vals):.1f}%')
    if parts:
        print(f'  {m:<30} {" | ".join(parts)}')

# ──────────────────────────────────────────────────────────────────
# 4. FPS summary
# ──────────────────────────────────────────────────────────────────
all_fps: dict = defaultdict(list)
for e in fps_data:
    for m, vs in fps_data[e].items():
        all_fps[m].extend(vs)

if any(all_fps.values()):
    print('\n' + '='*80)
    print('FPS (measured during training)')
    print('='*80)
    ref_fps = None
    for m, vals in sorted(all_fps.items()):
        if not vals:
            continue
        mean_fps = np.mean(vals)
        if m in ('si2e', 'fastse-si2e') and ref_fps is None:
            ref_fps = mean_fps
        ratio = f'  ({mean_fps / ref_fps:.1f}×)' if ref_fps else ''
        print(f'  {m:<32} {mean_fps:6.0f} ± {np.std(vals):4.0f} FPS  N={len(vals)}{ratio}')

# ──────────────────────────────────────────────────────────────────
# 5. Ablation summary
# ──────────────────────────────────────────────────────────────────
if abl_rows:
    print('\n' + '='*80)
    print('ABLATIONS (DoorKey-8x8)')
    print('='*80)
    from collections import defaultdict as dd
    abl_agg = defaultdict(list)
    for r in abl_rows:
        abl_agg[r['ablation']].append(r['sr'])
    dk_si2e = np.mean(sr_data['DoorKey-8x8'].get('si2e', [0]))
    for abl, vals in sorted(abl_agg.items()):
        delta = np.mean(vals) - dk_si2e if dk_si2e else 0
        print(f'  {abl:<20} {np.mean(vals):.1f}% ± {np.std(vals):.1f}  N={len(vals)}  (vs SI2E: {delta:+.1f})')

# ──────────────────────────────────────────────────────────────────
# 6. Clustering-method comparison
# ──────────────────────────────────────────────────────────────────
clust_methods = [m for m in all_methods if m in ('fast-si2e','leiden-si2e','infomap-si2e')]
if len(clust_methods) > 1:
    print('\n' + '='*80)
    print('CLUSTERING METHOD COMPARISON')
    print('='*80)
    ref_kmeans = {e: (np.mean(sr_data[e]['fast-si2e']) if sr_data[e].get('fast-si2e') else float('nan'))
                  for e in all_envs}
    for m in clust_methods:
        fps_vals = all_fps.get(m, [])
        fps_str  = f'{np.mean(fps_vals):.0f} FPS' if fps_vals else '— FPS'
        parts = []
        for e in all_envs:
            vals = sr_data[e].get(m, [])
            ref  = ref_kmeans[e]
            if vals:
                d = np.mean(vals) - ref if ref else 0
                parts.append(f'{e}: {np.mean(vals):.1f}% ({d:+.1f})')
        print(f'  {m:<22} {fps_str:>12}  ' + ' | '.join(parts))

# ──────────────────────────────────────────────────────────────────
# 7. LaTeX table (paper-ready)
# ──────────────────────────────────────────────────────────────────
print('\n' + '='*80)
print('LATEX TABLE — Success rate (%) ± std')
print('='*80)

DISPLAY = {
    'vcse-original':       'VCSE (orig.)',
    'si2e':                'SI2E (orig.)',
    'fast-si2e':           r'FastSI2E k-means (ours)',
    'leiden-si2e':         r'FastSI2E Leiden (ours)',
    'infomap-si2e':        r'FastSI2E Infomap (ours)',
    'si2e-adaptive':       r'SI2E + adaptive-$\beta$ (ours)',
    'fast-si2e-adaptive':  r'FastSI2E + adaptive-$\beta$ (ours)',
    'rbd-fast-si2e':       r'FastSI2E RBD (ours)',
}

latex_envs = [e for e in ENV_ORDER if e in sr_data]
print(r'\begin{table}[t]')
print(r'\centering')
print(r'\small')
cols = 'l' + 'c' * len(latex_envs)
print(r'\begin{tabular}{' + cols + '}')
print(r'\toprule')
env_labels = ' & '.join(e.replace('-',' ').replace('DoorKey 8x8','DK-8x8')
                        .replace('KeyCorridorS3R2','KC-S3R2')
                        .replace('RedBlueDoors 6x6','RBD-6x6')
                        for e in latex_envs)
print(r'Method & ' + env_labels + r' \\')
print(r'\midrule')
for m, display in DISPLAY.items():
    cells = [latex_cell(sr_data[e].get(m, [])) for e in latex_envs]
    if all(c == '—' for c in cells):
        continue
    print(f'{display} & ' + ' & '.join(cells) + r' \\')
print(r'\midrule')
# Speed row
fps_row = []
for e in latex_envs:
    km = all_fps.get('fast-si2e', fps_data[e].get('fast-si2e', []))
    si = all_fps.get('si2e', fps_data[e].get('si2e', []))
    if km and si:
        fps_row.append(f'${np.mean(km):.0f}$ vs ${np.mean(si):.0f}$')
    elif km:
        fps_row.append(f'${np.mean(km):.0f}$')
    else:
        fps_row.append('—')
print(r'FPS (FastSI2E vs SI2E) & ' + ' & '.join(fps_row) + r' \\')
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\caption{Success rate (\%) mean$\pm$std. FastSI2E uses \texttt{--fast\_se} '
      r'(k-means graph clustering). Adaptive-$\beta$ scales the intrinsic coefficient '
      r'by $(1-\hat{r}_\text{success})$ during training.}')
print(r'\label{tab:main_results}')
print(r'\end{table}')

# ──────────────────────────────────────────────────────────────────
# 8. Learning curve plots (optional)
# ──────────────────────────────────────────────────────────────────
if args.plot:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[WARN] matplotlib not available — skipping plots')
        sys.exit(0)

    def load_log(model_name, col='return_mean'):
        path = os.path.join(STORAGE, model_name, 'log.csv')
        if not os.path.exists(path):
            return None, None
        frames, vals = [], []
        with open(path) as f:
            first = f.readline()
            has_header = first.startswith('update')
            f.seek(0)
            if has_header:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        frames.append(int(row['frames']))
                        vals.append(float(row[col]))
                    except (ValueError, KeyError):
                        pass
            else:
                # Legacy headerless CSV (may have tensor strings with embedded commas)
                # Column order: update,frames,FPS,duration,rreturn_{mean,std,min,max},
                #   num_frames_{mean,std,min,max},entropy,value,policy_loss,value_loss,
                #   grad_norm,[return_{mean,std,min,max}]
                # grad_norm may be "tensor(x, device='cuda:0')" → use csv.reader to parse
                LEGACY_COLS = ['update','frames','FPS','duration',
                               'rreturn_mean','rreturn_std','rreturn_min','rreturn_max',
                               'num_frames_mean','num_frames_std','num_frames_min','num_frames_max',
                               'entropy','value','policy_loss','value_loss','grad_norm',
                               'return_mean','return_std','return_min','return_max',
                               'update_extr_value_loss']
                try:
                    idx = LEGACY_COLS.index(col)
                except ValueError:
                    idx = 17
                for parts in csv.reader(f):
                    try:
                        frames.append(int(parts[1]))
                        vals.append(float(parts[idx]))
                    except (ValueError, IndexError):
                        pass
        return (np.array(frames), np.array(vals)) if frames else (None, None)

    def smooth(y, w=7):
        return np.convolve(y, np.ones(w)/w, mode='same') if len(y) >= w else y

    def plot_group(ax, model_names, color, label, max_frames=None):
        curves = []
        for mn in model_names:
            fx, ry = load_log(mn)
            if fx is not None and len(fx) > 5:
                if max_frames:
                    mask = fx <= max_frames
                    fx, ry = fx[mask], ry[mask]
                curves.append((fx, smooth(ry)))
        if not curves:
            return
        mf = max(c[0][-1] for c in curves)
        xg = np.linspace(0, mf, 300)
        interped = [np.interp(xg, fx, ry) for fx, ry in curves]
        mean_c = np.mean(interped, axis=0)
        std_c  = np.std(interped, axis=0)
        ax.plot(xg/1e6, mean_c, color=color, label=f'{label} (N={len(curves)})', lw=1.5)
        ax.fill_between(xg/1e6, mean_c-std_c, mean_c+std_c, alpha=0.12, color=color)

    # ── Panel 1: DK-8x8 SI2E vs FastSI2E ──
    # ── Panel 2: KC-S3R2 SI2E vs FastSI2E ──
    # ── Panel 3: RBD SI2E-adaptive vs FastSI2E ──
    # ── Panel 4: clustering methods on DK-8x8 ──

    specs = [
        # (title, [(model_names_list, color, label), ...])
        ('DK-8x8: SI2E vs FastSI2E', [
            ([f'fastse-si2e-DoorKey-8x8-s{i}'       for i in (1,2,3)], 'royalblue', 'SI2E (1M)'),
            ([f'fastse-fast-si2e-DoorKey-8x8-s{i}'  for i in (1,2,3)], 'seagreen',  'FastSI2E (3M)'),
        ]),
        ('KC-S3R2: SI2E vs FastSI2E', [
            ([f'ppo-exp-a2c-si2e-KeyCorridorS3R2-s{i}' for i in (1,2,3,4)], 'royalblue', 'SI2E'),
            ([f'fastse-fast-si2e-KeyCorridorS3R2-s{i}'  for i in range(1,6)], 'seagreen',  'FastSI2E'),
        ]),
        ('RBD: fixed-β vs adaptive-β', [
            ([f'abeta-si2e-fixed-RedBlueDoors-6x6-s{i}'    for i in (1,2,3,4,5)], 'royalblue',  'SI2E fixed-β'),
            ([f'abeta-si2e-adaptive-RedBlueDoors-6x6-s{i}' for i in (1,2,3,4,5)], 'darkorange', 'SI2E adaptive-β'),
        ]),
        ('DK-8x8: clustering methods', [
            ([f'cm-leiden-si2e-DoorKey-8x8-s{i}'   for i in (1,2,3)], 'mediumorchid', 'Leiden'),
            ([f'cm-infomap-si2e-DoorKey-8x8-s{i}'  for i in (1,2,3)], 'tomato',       'Infomap'),
            ([f'fastse-fast-si2e-DoorKey-8x8-s{i}' for i in (1,2,3)], 'seagreen',     'k-means'),
        ]),
    ]

    # storage model names for adaptive-beta (different prefix)
    def storage_name_for_adaptive(method, env, seed):
        e = env.replace('MiniGrid-','').replace('-v0','')
        return f'{method}-{e}-s{seed}'

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (title, groups) in zip(axes, specs):
        for model_names, color, label in groups:
            plot_group(ax, model_names, color, label)
        ax.set_xlabel('Frames (M)', fontsize=10)
        ax.set_ylabel('Mean Return', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{BASE}/results/learning_curves.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nLearning curves saved: {out}')
    plt.close()

    # ── FPS bar chart ──
    fps_display = {
        'si2e':          ('SI2E\n(PartitionTree)', 'royalblue'),
        'fast-si2e':     ('FastSI2E\n(k-means)',   'seagreen'),
        'leiden-si2e':   ('FastSI2E\n(Leiden)',     'mediumorchid'),
        'infomap-si2e':  ('FastSI2E\n(Infomap)',    'tomato'),
    }
    fps_vals_bar = [(lbl, col, np.mean(all_fps[m])) for m, (lbl, col) in fps_display.items() if all_fps[m]]
    if fps_vals_bar:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        labels_b = [x[0] for x in fps_vals_bar]
        colors_b = [x[1] for x in fps_vals_bar]
        values_b = [x[2] for x in fps_vals_bar]
        bars = ax2.bar(labels_b, values_b, color=colors_b, edgecolor='k', linewidth=0.5)
        for bar, v in zip(bars, values_b):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 30, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
        ax2.set_ylabel('FPS', fontsize=11)
        ax2.set_title('Training Speed Comparison', fontsize=11)
        ax2.set_ylim(0, max(values_b) * 1.2)
        ax2.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        fps_out = f'{BASE}/results/fps_comparison.png'
        plt.savefig(fps_out, dpi=150, bbox_inches='tight')
        print(f'FPS chart saved: {fps_out}')
        plt.close()

print('\nDone.')
