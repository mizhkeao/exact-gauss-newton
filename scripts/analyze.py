"""Aggregate campaign results into figures and a markdown report.

    python scripts/analyze.py results/campaign/A --out results/figures/A
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# fixed categorical slots per method (color follows the entity, never its rank)
COLORS = {
    "adamw": "#2a78d6", "muon": "#eb6834", "lbfgs": "#1baf7a", "adamw_lbfgs": "#eda100",
    "muon_lbfgs": "#e87ba4", "plm": "#008300", "lstsq": "#8c1d40",
}
TAG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]
LABELS = {"adamw": "AdamW", "muon": "Muon", "lbfgs": "L-BFGS", "adamw_lbfgs": "AdamW→L-BFGS",
          "muon_lbfgs": "Muon→L-BFGS", "plm": "PLM", "lstsq": "Frozen-feature LS"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.5})


def load(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*.json")):
        with open(p) as f:
            r = json.load(f)
        j = r["job"]
        r["setting"] = (j["problem"], tuple(sorted(j.get("knobs", {}).items())), j["dtype"], j["width"])
        r["series"] = j["method"] + (f":{j['tag']}" if j.get("tag") else "")
        out.append(r)
    return out


def setting_name(s) -> str:
    prob, knobs, dtype, width = s
    k = " ".join(f"{a}={b:g}" if isinstance(b, (int, float)) else f"{a}={b}" for a, b in knobs)
    return f"{prob} {k} w={width} {dtype}".replace("  ", " ")


def series_label(s: str) -> str:
    m, _, tag = s.partition(":")
    return LABELS.get(m, m) + (f" [{tag}]" if tag else "")


VARIANT_COLORS = ["#6250d6", "#e34948", "#52514e", "#8c5a2b", "#0d366b", "#9c27b0", "#5d4037", "#1c5cab", "#b0b0b0"]


def series_color(s: str, i: int, mixed: bool = False) -> str:
    """Untagged methods keep their fixed slot. Tagged variants take slots no base method uses
    when they share a figure with base methods (`mixed`), else the full tag palette."""
    m, _, tag = s.partition(":")
    if not tag:
        return COLORS.get(m, "#52514e")
    pal = VARIANT_COLORS if mixed else TAG_COLORS
    return pal[i % len(pal)]


def series_colors(series: list[str]) -> dict[str, str]:
    """Color follows the entity: a method keeps its slot; a tagged variant keeps its method's slot
    when it is the only variant of that method in the figure, else takes a variant slot."""
    per_method = defaultdict(list)
    for s in series:
        per_method[s.partition(":")[0]].append(s)
    out, i = {}, 0
    for s in sorted(series):
        m = s.partition(":")[0]
        if len(per_method[m]) == 1 and m in COLORS:
            out[s] = COLORS[m]
        else:
            out[s] = VARIANT_COLORS[i % len(VARIANT_COLORS)] if ":" in s else COLORS.get(m, "#52514e")
            i += ":" in s
    return out


def curve_on_grid(rows, grid):
    """Best-so-far rel_l2 sampled on a time grid."""
    t = np.array([r["time"] for r in rows])
    e = np.minimum.accumulate(np.array([r["rel_l2"] for r in rows]))
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.where(idx >= 0, e[np.clip(idx, 0, len(e) - 1)], np.nan)
    return out


def plot_curves(results, out: Path, fname: str, title: str):
    by_series = defaultdict(list)
    for r in results:
        by_series[r["series"]].append(r)
    budget = max(r["job"]["budget"] for r in results)
    grid = np.linspace(0, budget, 300)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cols = series_colors(list(by_series))
    for s, rs in sorted(by_series.items()):
        curves = np.array([curve_on_grid(r["rows"], grid) for r in rs])
        med, lo, hi = np.nanmedian(curves, 0), np.nanmin(curves, 0), np.nanmax(curves, 0)
        c = cols[s]
        ax.plot(grid, med, color=c, label=f"{series_label(s)} (n={len(rs)})")
        if len(rs) > 1:
            ax.fill_between(grid, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.annotate(series_label(s), (grid[-1], med[~np.isnan(med)][-1] if np.any(~np.isnan(med)) else np.nan),
                    xytext=(3, 0), textcoords="offset points", color=c, fontsize=7, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("training time (s)")
    ax.set_ylabel("best relative L2 error so far")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def fmt(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1e}"


def summary_tables(results) -> tuple[str, str]:
    by = defaultdict(list)
    for r in results:
        by[(r["setting"], r["series"])].append(r["summary"])
    settings = sorted({k[0] for k in by})
    series = sorted({k[1] for k in by})
    err = "| setting | " + " | ".join(series_label(s) for s in series) + " |\n|---|" + "---|" * len(series) + "\n"
    cost = "| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |\n|---|---|---|---|---|---|---|---|---|\n"
    for st in settings:
        cells = []
        for s in series:
            ss = by.get((st, s))
            if not ss:
                cells.append("—")
                continue
            f = np.array([x["final_rel_l2"] for x in ss])
            cells.append(f"{np.median(f):.1e} ({f.min():.0e}–{f.max():.0e})")
            med = lambda k: np.median([x[k] for x in ss if x[k] is not None]) if any(x[k] is not None for x in ss) else None
            cost += (f"| {setting_name(st)} | {series_label(s)} | {np.median([x['sec_per_iter'] for x in ss]):.2e} | "
                     f"{fmt(med('peak_mem_mb'))} | {np.median([x['grad_evals'] for x in ss]):.0f} | "
                     f"{np.median([x['jac_evals'] for x in ss]):.0f} | {np.median([x['iters'] for x in ss]):.0f} | "
                     f"{fmt(med('t_0.001'))} | {fmt(med('t_1e-05'))} |\n")
        err += f"| {setting_name(st)} | " + " | ".join(cells) + " |\n"
    return err, cost


def plot_knob_sweep(results, out: Path, problem: str, knob: str, fname: str):
    pts = defaultdict(lambda: defaultdict(list))
    for r in results:
        j = r["job"]
        if j["problem"] == problem and knob in j.get("knobs", {}):
            pts[r["series"]][j["knobs"][knob]].append(r["summary"]["final_rel_l2"])
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cols = series_colors(list(pts))
    for s, d in sorted(pts.items()):
        ks = sorted(d)
        med = [np.median(d[k]) for k in ks]
        c = cols[s]
        ax.plot(ks, med, "o-", color=c, markersize=5, label=series_label(s))
        for k in ks:
            ax.scatter([k] * len(d[k]), d[k], color=c, s=8, alpha=0.5)
    ax.set_yscale("log")
    pos = [k for k in ks if isinstance(k, (int, float)) and k > 0]
    if pos and max(pos) / min(pos) > 20:
        if min(ks) > 0:
            ax.set_xscale("log")
        else:
            ax.set_xscale("symlog", linthresh=min(pos))  # keeps the knob=0 point on the axis
            ax.set_xlim(left=-min(pos) * 0.5)
    ax.set_xlabel(knob)
    ax.set_ylabel("final relative L2 error")
    ax.set_title(f"{problem}: error vs {knob}", loc="left")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def plot_scale(results, out: Path):
    pts = defaultdict(lambda: defaultdict(list))
    for r in results:
        pts[r["series"]][r["job"]["width"]].append(r["summary"])
    if len({w for d in pts.values() for w in d}) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    keys = [("final_rel_l2", "final relative L2 error"), ("sec_per_iter", "seconds per iteration"), ("peak_mem_mb", "peak GPU memory (MB)")]
    cols = series_colors(list(pts))
    for ax, (k, lab) in zip(axes, keys):
        for s, d in sorted(pts.items()):
            ws = sorted(d)
            vals = [np.median([x[k] for x in d[w] if x[k] is not None]) for w in ws]
            Ds = [np.median([x["D"] for x in d[w]]) for w in ws]
            ax.plot(Ds, vals, "o-", color=cols[s], markersize=5, label=series_label(s))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("parameters D")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("scale sweep", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "scale_sweep.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (a.root.parent.parent / "figures" / a.root.name)
    out.mkdir(parents=True, exist_ok=True)
    results = load(a.root)
    print(f"{len(results)} runs")

    by_setting = defaultdict(list)
    for r in results:
        by_setting[r["setting"]].append(r)
    for st, rs in sorted(by_setting.items()):
        name = setting_name(st)
        plot_curves(rs, out, "curves_" + name.replace(" ", "_").replace("=", "") + ".png", name)

    for prob, knob in [("multiscale", "k"), ("noisy", "sigma"), ("convection", "beta"), ("helmholtz", "a2")]:
        plot_knob_sweep(results, out, prob, knob, f"sweep_{prob}_{knob}.png")
    plot_scale(results, out)

    err, cost = summary_tables(results)
    report = f"# {a.root.name}: {len(results)} runs\n\n## Final relative L2 error, median (min–max) over seeds\n\n{err}\n## Cost\n\n{cost}"
    (out / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
