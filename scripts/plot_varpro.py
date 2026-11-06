"""Phase V (round 4): PLM rec2 vs rec2 + variable projection (vp) / collocation resampling (rs) / both (vprs),
with the tuned frozen-feature LS baseline as a reference. Baselines: rec2 from H (1D, 60 s), H2 and P (300 s);
LS from L and P. Every seed is drawn; the bar is the median.

    python scripts/plot_varpro.py results/campaign docs/figures > results/figures/V/report.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

PROBS = ["order1", "poisson", "beam", "multiscale k=16", "multiscale k=32", "nlpoisson", "helmholtz", "allen_cahn", "burgers", "reaction"]
VARIANTS = [("rec2", "PLM rec2"), ("vp", "+ varpro"), ("rs", "+ resample"), ("vprs", "+ both"), ("lstsq", "frozen-feature LS")]
COL = {"rec2": "#1f77b4", "vp": "#ff7f0e", "rs": "#2ca02c", "vprs": "#d62728", "lstsq": "#8c1d40",
       "fs0.1": "#c7b8e8", "fs0.3": "#9467bd", "fs1": "#4b2a7a"}
if "--two-stage" in sys.argv:  # Phase W: two-stage (basis learning, then one output-layer solve) at 10 / 30 / 100 % budget
    VARIANTS = [("rec2", "PLM rec2"), ("rs", "+ resample"), ("fs0.1", "2-stage 10 %"), ("fs0.3", "2-stage 30 %"), ("fs1", "2-stage 100 %"), ("lstsq", "frozen-feature LS")]
    PHASES = ["H", "H2", "P", "L", "V", "W"]
else:
    PHASES = ["H", "H2", "P", "L", "V"]
VARS = {v for v, _ in VARIANTS}


def label(j):
    return j["problem"] + (f" k={j['knobs']['k']}" if "k" in j["knobs"] else "")


def collect():
    d = defaultdict(lambda: defaultdict(list))  # problem -> variant -> [(err, s/iter, jac_evals, val_gap)]
    for ph in PHASES:
        for f in (root / ph).glob("*.json"):
            r = json.load(open(f))
            j = r["job"]
            if j["width"] != 50 or j["method"] not in ("plm", "lstsq"):
                continue
            tag = j["tag"]
            if j["method"] == "plm":
                if not tag.startswith("rec2"):
                    continue
                last_ = tag.split("_")[-1]
                var = last_ if (last_ in ("vp", "rs", "vprs") or last_.startswith("fs")) else "rec2"
                if ph in ("V", "W") and var == "rec2":
                    continue
                if var not in VARS:
                    continue
            else:
                var = f"lstsq:{tag}"
            s, last = r["summary"], r["rows"][-1]
            d[label(j)][var].append((s["final_rel_l2"], s["sec_per_iter"], s["jac_evals"], last.get("val_loss", np.nan) / max(last["loss"], 1e-300)))
    # LS: better feat_scale by median error
    for p, by in d.items():
        ls = {k: v for k, v in by.items() if k.startswith("lstsq:")}
        if ls:
            best = min(ls, key=lambda k: np.median([e for e, *_ in ls[k]]))
            by["lstsq"] = ls[best]
    return d


d = collect()
fig, ax = plt.subplots(figsize=(11, 3.8))
bw = 0.16 if "--two-stage" not in sys.argv else 0.14
lines = ["# V: PLM upgrades, round 4 (width 50)", "", "Final relative L2, median (min–max) over seeds; `bad` = seeds with error > 1e-2; "
         "s/Jac = train time per Jacobian; val/train = validation residual / training loss at the end (fixed grid).", "",
         "| problem | variant | n | median (min–max) | bad | s/Jac | val/train |", "|---|---|---|---|---|---|---|"]
for k, p in enumerate(PROBS):
    for i, (var, lab) in enumerate(VARIANTS):
        v = d.get(p, {}).get(var)
        if not v:
            continue
        errs = np.array([e for e, *_ in v])
        x = k + (i - len(VARIANTS) / 2 + 0.5) * bw
        ax.bar(x, np.median(errs), bw * 0.9, color=COL[var], label=lab if k == 0 else None, alpha=0.85)
        ax.scatter(np.full(len(errs), x), errs, s=7, color="k", zorder=3, alpha=0.6)
        bad = int((errs > 1e-2).sum())
        spj = np.median([s for _, s, *_ in v])
        vg = np.nanmedian([g for *_, g in v]) if var != "lstsq" else np.nan
        lines.append(f"| {p} | {lab} | {len(errs)} | {np.median(errs):.1e} ({errs.min():.0e}–{errs.max():.0e}) | {bad} | {spj:.2f} | {vg:.1f} |")
ax.set_yscale("log")
ax.set_xticks(range(len(PROBS)))
ax.set_xticklabels(PROBS, rotation=20, ha="right")
ax.set_ylabel("final relative L2 error")
two = "--two-stage" in sys.argv
ax.set_title("two-stage: PLM learns the basis for 10 / 30 / 100 % of the budget, then one output-layer solve" if two else
             "PLM rec2 vs variable projection (vp) vs collocation resampling (rs); frozen-feature LS for reference", fontsize=9)
ax.legend(ncol=6, fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(out / ("two_stage.png" if two else "varpro_resample.png"), dpi=150)
print("\n".join(lines))
