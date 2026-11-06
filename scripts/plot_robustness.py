import json
import re
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
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 1.5})
data = defaultdict(lambda: defaultdict(list))
for f in root.glob('*.json'):
    r = json.load(open(f))
    j = r['job']
    hp = float(re.sub('^[a-z]+', '', j['tag']))
    data[j['problem'], j['method']][hp].append(r['summary']['final_rel_l2'])
problems = sorted({k[0] for k in data})
methods = ['adamw', 'adamw_lbfgs', 'muon', 'muon_lbfgs', 'plm']
xlab = {'adamw': 'AdamW lr', 'adamw_lbfgs': 'AdamW lr (then L-BFGS)', 'muon': 'Muon lr', 'muon_lbfgs': 'Muon lr (then L-BFGS)', 'plm': 'PLM initial damping μ₀'}
fig, axes = plt.subplots(len(problems), len(methods), figsize=(3.0 * len(methods), 2.6 * len(problems)), sharey='row')
for i, p in enumerate(problems):
    for k, m in enumerate(methods):
        ax = axes[i, k]
        d = data.get((p, m), {})
        xs = sorted(d)
        med = [np.median(d[x]) for x in xs]
        ax.plot(xs, med, 'o-', color=COLORS[m], markersize=4)
        for x in xs:
            ax.scatter([x] * len(d[x]), d[x], color=COLORS[m], s=6, alpha=0.4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if i == 0:
            ax.set_title(LABELS[m], loc='left', fontsize=9)
        if i == len(problems) - 1:
            ax.set_xlabel(xlab[m], fontsize=8)
        if k == 0:
            ax.set_ylabel(f'{p}\nfinal rel. L2 error', fontsize=8)
fig.suptitle("Sensitivity to the main hyperparameter (8 decades for PLM's μ₀, 2 decades of lr for the rest); dots = seeds", x=0.01, ha='left')
fig.tight_layout()
fig.savefig(out / 'robustness.png', dpi=150)
lines = ['| problem | method | best | worst | worst/best |', '|---|---|---|---|---|']
for p in problems:
    for m in methods:
        d = data.get((p, m), {})
        med = {x: np.median(v) for x, v in d.items()}
        b, w = (min(med.values()), max(med.values()))
        lines.append(f'| {p} | {LABELS[m]} | {b:.1e} | {w:.1e} | {w / b:.0f} |')
(out / 'robustness.md').write_text('\n'.join(lines))
print('\n'.join(lines))
