import json
import sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from analyze import COLORS, LABELS
root, out = (Path(sys.argv[1]), Path(sys.argv[2]))
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.axisbelow': True})

def collect(phases, width=50):
    d = defaultdict(lambda: defaultdict(list))
    for ph in phases:
        for f in (root / ph).glob('*.json'):
            r = json.load(open(f))
            j = r['job']
            if j['width'] != width:
                continue
            k = j['knobs']
            lab = j['problem'] + (f" k={k['k']}" if 'k' in k else '')
            if j['method'] == 'plm' and (not j['tag'].startswith('rec2')):
                continue
            d[lab, j['method']][j['tag']].append(r['summary']['final_rel_l2'])
    best = {}
    for key, by_tag in d.items():
        best[key] = by_tag[min(by_tag, key=lambda t: np.median(by_tag[t]))]
    return best
panels = [('linear in u (60 s; helmholtz 300 s)', collect(['H', 'H2', 'L']), ['order1', 'poisson', 'beam', 'multiscale k=16', 'multiscale k=32', 'helmholtz']), ('nonlinear in u (300 s)', collect(['P']), ['nlpoisson', 'allen_cahn', 'burgers', 'reaction'])]
methods = ['lstsq', 'plm', 'adamw_lbfgs', 'muon_lbfgs']
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True, gridspec_kw={'width_ratios': [6, 4]})
bw = 0.19
for ax, (title, data, probs) in zip(axes, panels):
    for i, m in enumerate(methods):
        xs, meds = ([], [])
        for k, p in enumerate(probs):
            v = data.get((p, m))
            if not v:
                continue
            x = k + (i - 1.5) * bw
            xs.append(x)
            meds.append(np.median(v))
            ax.scatter([x] * len(v), v, color=COLORS[m], s=7, alpha=0.5, zorder=3)
        ax.bar(xs, meds, width=bw - 0.02, color=COLORS[m], label=LABELS[m], zorder=2, linewidth=0)
    ax.set_yscale('log')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(probs, rotation=20, ha='right')
    ax.set_title(title, loc='left')
    ax.axhline(1.0, color='#52514e', lw=0.8, ls='--')
axes[0].set_ylabel('final rel. L2 error (median; dots = seeds)')
axes[0].set_ylim(1e-15, 10)
axes[0].legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
fig.suptitle('Frozen-feature least squares wins where random features span the solution; PLM wins where they do not', x=0.01, ha='left')
fig.tight_layout()
fig.savefig(out / 'lstsq_vs_plm.png', dpi=150)
