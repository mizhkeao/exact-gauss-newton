"""Phase N: final error vs forcing-noise level sigma, per problem and method.

    python scripts/plot_noise.py results/campaign/N results/figures/N
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import COLORS, LABELS  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.5})

data = defaultdict(lambda: defaultdict(list))  # (problem, method) -> sigma -> errors
for f in root.glob("*.json"):
    r = json.load(open(f))
    j = r["job"]
    data[(j["problem"], j["method"])][float(j["knobs"]["pde_noise"])].append(r["summary"]["final_rel_l2"])

problems = sorted({k[0] for k in data})
methods = [m for m in ["adamw_lbfgs", "muon_lbfgs", "plm"] if any((p, m) in data for p in problems)]
fig, axes = plt.subplots(1, len(problems), figsize=(3.6 * len(problems), 3.0), sharey=True)
for ax, p in zip(np.atleast_1d(axes), problems):
    for m in methods:
        d = data.get((p, m), {})
        xs = sorted(d)
        ax.plot(xs, [np.median(d[x]) for x in xs], "o-", color=COLORS[m], markersize=4, label=LABELS[m])
        for x in xs:
            ax.scatter([x] * len(d[x]), d[x], color=COLORS[m], s=6, alpha=0.4)
    xs = sorted({x for m in methods for x in data.get((p, m), {})})
    ax.plot(xs, [5.4e-3 * x for x in xs], "--", color="#52514e", lw=2.5, alpha=0.5, zorder=0, label="5.4e-3 σ")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forcing noise σ")
    ax.set_title(p, loc="left")
np.atleast_1d(axes)[0].set_ylabel("final rel. L2 error")
np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
fig.suptitle("Noisy forcing: the floor is 5.4e-3 σ; only PLM reaches it on multiscale", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out / "noise_sweep.png", dpi=150)
