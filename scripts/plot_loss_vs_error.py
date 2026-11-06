"""E8: final training loss vs final relative L2 error, convection beta sweep.

    python scripts/plot_loss_vs_error.py results/campaign/E2 results/figures/E2
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyze import COLORS, LABELS  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
MARK = {1.0: "o", 10.0: "s", 30.0: "^", 40.0: "D"}

fig, ax = plt.subplots(figsize=(6.0, 4.0))
seen = set()
for f in sorted(root.glob("*.json")):
    r = json.load(open(f))
    j, s = r["job"], r["summary"]
    b, m = j["knobs"]["beta"], j["method"]
    lab = f"{LABELS[m]}" if m not in seen else None
    seen.add(m)
    ax.scatter(s["final_loss"], s["final_rel_l2"], color=COLORS[m], marker=MARK[b], s=36,
               edgecolor="white", linewidth=0.8, label=lab, zorder=3)
leg1 = ax.legend(fontsize=7, frameon=False, loc="center left", title="method", title_fontsize=7)
ax.add_artist(leg1)
hs = [ax.scatter([], [], color="#52514e", marker=mk, s=30, label=f"β = {b:g}") for b, mk in MARK.items()]
ax.legend(handles=hs, fontsize=7, frameon=False, loc="lower right", title="wave speed", title_fontsize=7)
ax.set_xscale("log")
ax.set_yscale("log")
ax.axhline(1.0, color="#52514e", lw=0.8, ls=":")
ax.annotate("error ≈ 1: no better than u = 0", (0.02, 0.97), xycoords="axes fraction", fontsize=7, color="#52514e")
ax.set_xlabel("final training loss")
ax.set_ylabel("final relative L2 error")
ax.set_title("Convection: loss and error decouple at β ≥ 30", loc="left")
fig.tight_layout()
fig.savefig(out / "loss_vs_error.png", dpi=150)
print(out / "loss_vs_error.png")
