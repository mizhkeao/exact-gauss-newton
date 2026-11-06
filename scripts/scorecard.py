import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, rankdata
root = Path(sys.argv[1])
S = {'adamw': 'A', 'muon': 'M', 'lbfgs': 'L', 'adamw_lbfgs': 'A→L', 'muon_lbfgs': 'M→L', 'plm': 'P', 'lstsq': 'LS'}

def load(phase, problem, knobs=None, width=50, tags=None):
    d = defaultdict(lambda: defaultdict(list))
    for ph in [phase] if isinstance(phase, str) else phase:
        for f in (root / ph).glob('*.json'):
            r = json.load(open(f))
            j = r['job']
            if j['problem'] != problem or j['width'] != width or (knobs and any((j['knobs'].get(k) != v for k, v in knobs.items()))):
                continue
            if tags and j['method'] in tags and (j['tag'] != tags[j['method']]):
                continue
            d[S[j['method']]][j['tag']].append(r['summary']['final_rel_l2'])
    out = {}
    for sym, by_tag in d.items():
        best = min(by_tag, key=lambda t: np.median(by_tag[t]))
        out[sym] = by_tag[best]
    return out

def observed_ranks(d, methods):
    med = {m: np.median(d[m]) for m in methods}
    lo = {m: min(d[m]) for m in methods}
    hi = {m: max(d[m]) for m in methods}
    group = {m: m for m in methods}

    def find(m):
        while group[m] != m:
            m = group[m]
        return m
    for a in methods:
        for b in methods:
            if a < b and lo[a] <= hi[b] and (lo[b] <= hi[a]):
                group[find(a)] = find(b)
    gmed = {g: np.mean([med[m] for m in methods if find(m) == g]) for g in {find(m) for m in methods}}
    return (rankdata([gmed[find(m)] for m in methods], method='average'), med)

def pred_ranks(pred: str, methods):
    r, k = ({}, 1)
    for tier in pred.split('>'):
        ms = [m.strip() for m in tier.split('≈')]
        for m in ms:
            r[m] = k + (len(ms) - 1) / 2
        k += len(ms)
    return np.array([r[m] for m in methods])

def fmt_rank(methods, ranks, med):
    order = sorted(zip(ranks, methods), key=lambda x: (x[0], med[x[1]]))
    out, prev = ([], None)
    for rk, m in order:
        out.append(('≈' if rk == prev else '>') if out else '')
        out.append(m)
        prev = rk
    return ' '.join((x for x in out if x))
ROUND1 = [('E1 order 1 (Phase A)', 'A', 'order1', {}, 'P > A→L ≈ M→L > L > M ≈ A'), ('E1 order 2 (Phase A)', 'A', 'poisson', {}, 'P > A→L ≈ M→L > L > M ≈ A'), ('E1 order 4 (Phase A)', 'A', 'beam', {}, 'P > A→L ≈ M→L > L > M ≈ A'), ('E3 Helmholtz (Phase D)', 'D', 'helmholtz', {'a2': 4}, 'P > A→L > A'), ('E5 k = 1 (Phase E)', 'E', 'multiscale', {'k': 1}, 'P > A→L > A'), ('E5 k = 4 (Phase E)', 'E', 'multiscale', {'k': 4}, 'P > A→L > A'), ('E5 k = 16 (Phase E)', 'E', 'multiscale', {'k': 16}, 'P > A→L > A'), ('E5 k = 32 (Phase E)', 'E', 'multiscale', {'k': 32}, 'P > A→L > A'), ('E9 reaction (Phase D2)', 'D2', 'reaction', {'rho': 5.0}, 'P ≈ A→L ≈ M→L > L'), ('E9 wave (Phase D2)', 'D2', 'wave', {}, 'P ≈ A→L ≈ M→L > L'), ('E9 convection β=30 (Phase D2)', 'D2', 'convection', {'beta': 30.0}, 'P ≈ A→L ≈ M→L > L')]
REC2 = {'plm': 'rec2'}
REC2L = {'plm': 'rec2_lag4'}
ROUND2 = [('L order1 w=50 (H + L)', ['H', 'L'], 'order1', {}, 'LS > P', REC2), ('L poisson w=50 (H + L)', ['H', 'L'], 'poisson', {}, 'LS > P', REC2), ('L beam w=50 (H + L)', ['H', 'L'], 'beam', {}, 'LS > P', REC2), ('L multiscale k=16 w=50 (H + L)', ['H', 'L'], 'multiscale', {'k': 16}, 'P > LS', REC2), ('L multiscale k=32 w=50 (H + L)', ['H', 'L'], 'multiscale', {'k': 32}, 'P > LS', REC2), ('L multiscale k=16 w=200 (L)', ['L'], 'multiscale', {'k': 16}, 'P ≈ LS', REC2), ('L multiscale k=32 w=200 (L)', ['L'], 'multiscale', {'k': 32}, 'P > LS', REC2), ('L helmholtz w=50 (H2 300 s + L)', ['H2', 'L'], 'helmholtz', {'a2': 4}, 'P > LS', REC2L), ('P nlpoisson (P)', ['P'], 'nlpoisson', {}, 'LS > P > A→L ≈ M→L', REC2), ('P burgers (P)', ['P'], 'burgers', {'nu': 0.05}, 'P > A→L ≈ M→L > LS', REC2L), ('P allen_cahn (P)', ['P'], 'allen_cahn', {}, 'P > LS ≈ A→L ≈ M→L', REC2L), ('P reaction (P)', ['P'], 'reaction', {'rho': 5.0}, 'P > M→L > LS ≈ A→L', REC2L), ('H2 helmholtz 5 seeds (H2)', ['H2'], 'helmholtz', {'a2': 4}, 'P > A→L ≈ M→L', REC2L), ('H2 reaction 5 seeds (H2)', ['H2'], 'reaction', {'rho': 5.0}, 'P > M→L > A→L', REC2L), ('H2 wave 5 seeds (H2)', ['H2'], 'wave', {}, 'P > A→L ≈ M→L', REC2L), ('H3 convection β=30 causal (H3)', ['H3'], 'convection', {'beta': 30.0, 'causal': 1}, 'P > A→L ≈ M→L ≈ A', REC2L)]

def load_variants(phases, problem, knobs, tagsym, width=50):
    d = defaultdict(lambda: defaultdict(list))
    for ph in phases:
        for f in (root / ph).glob('*.json'):
            r = json.load(open(f))
            j = r['job']
            if j['problem'] != problem or j['width'] != width or any((j['knobs'].get(k) != v for k, v in knobs.items())):
                continue
            if j['method'] == 'plm':
                if j['tag'] not in tagsym:
                    continue
                d[tagsym[j['tag']]][j['tag']].append(r['summary']['final_rel_l2'])
            elif j['method'] == 'lstsq':
                d['LS'][j['tag']].append(r['summary']['final_rel_l2'])
    return {sym: by[min(by, key=lambda t: np.median(by[t]))] for sym, by in d.items()}
V1 = {'rec2': 'P', 'rec2_vp': 'Pv', 'rec2_rs': 'Pr', 'rec2_vprs': 'Pvr'}
V4 = {'rec2_lag4': 'P', 'rec2_lag4_vp': 'Pv', 'rec2_lag4_rs': 'Pr', 'rec2_lag4_vprs': 'Pvr'}
ROUND4 = [('V order1 (H + L + V)', ['H', 'L', 'V'], 'order1', {}, 'Pv ≈ Pvr ≈ LS > P ≈ Pr', V1), ('V poisson (H + L + V)', ['H', 'L', 'V'], 'poisson', {}, 'Pv ≈ Pvr ≈ LS > P ≈ Pr', V1), ('V beam (H + L + V)', ['H', 'L', 'V'], 'beam', {}, 'Pv ≈ Pvr ≈ LS > P ≈ Pr', V1), ('V multiscale k=16 (H + L + V)', ['H', 'L', 'V'], 'multiscale', {'k': 16}, 'Pv ≈ Pvr ≈ P ≈ Pr > LS', V1), ('V multiscale k=32 (H + L + V)', ['H', 'L', 'V'], 'multiscale', {'k': 32}, 'Pr ≈ Pvr > P ≈ Pv > LS', V1), ('V nlpoisson (P + V)', ['P', 'V'], 'nlpoisson', {}, 'Pv ≈ Pvr ≈ LS > P ≈ Pr', V1), ('V helmholtz (H2 + L + V)', ['H2', 'L', 'V'], 'helmholtz', {'a2': 4}, 'Pv ≈ Pvr ≈ P ≈ Pr > LS', V4), ('V allen_cahn (P + V)', ['P', 'V'], 'allen_cahn', {}, 'Pv ≈ Pvr ≈ P ≈ Pr > LS', V4), ('V burgers (P + V)', ['P', 'V'], 'burgers', {'nu': 0.05}, 'Pv ≈ Pvr ≈ P ≈ Pr > LS', V4), ('V reaction (H2 + P + V)', ['H2', 'P', 'V'], 'reaction', {'rho': 5.0}, 'Pr ≈ Pvr > P ≈ Pv > LS', V4)]
W1 = {'rec2': 'P', 'rec2_vp': 'Pv', 'rec2_fs0.1': 'F1', 'rec2_fs0.3': 'F3', 'rec2_fs1': 'F10'}
W4 = {'rec2_lag4': 'P', 'rec2_lag4_rs': 'Pr', 'rec2_lag4_vp': 'Pv', 'rec2_lag4_rs_fs0.1': 'F1', 'rec2_lag4_rs_fs0.3': 'F3', 'rec2_lag4_rs_fs1': 'F10'}
ROUND5 = [('W order1', ['H', 'L', 'V', 'W'], 'order1', {}, 'F1 ≈ F3 ≈ F10 ≈ LS ≈ Pv > P', W1), ('W poisson', ['H', 'L', 'V', 'W'], 'poisson', {}, 'F1 ≈ F3 ≈ F10 ≈ LS ≈ Pv > P', W1), ('W beam', ['H', 'L', 'V', 'W'], 'beam', {}, 'F1 ≈ F3 ≈ F10 ≈ LS ≈ Pv > P', W1), ('W nlpoisson', ['P', 'V', 'W'], 'nlpoisson', {}, 'F1 ≈ F3 ≈ F10 ≈ LS ≈ Pv > P', W1), ('W multiscale k=16', ['H', 'L', 'V', 'W'], 'multiscale', {'k': 16}, 'F10 > P ≈ F3 > F1 > LS', W1), ('W multiscale k=32', ['H', 'L', 'V', 'W'], 'multiscale', {'k': 32}, 'F10 ≈ P > F3 > F1 ≈ LS', W1), ('W helmholtz', ['H2', 'L', 'V', 'W'], 'helmholtz', {'a2': 4}, 'F10 > F3 ≈ Pr > P > F1 > LS', W4), ('W allen_cahn', ['P', 'V', 'W'], 'allen_cahn', {}, 'F10 > F3 ≈ P ≈ Pr > F1 > LS', W4), ('W burgers', ['P', 'V', 'W'], 'burgers', {'nu': 0.05}, 'F10 > F3 ≈ P ≈ Pr > F1 > LS', W4), ('W reaction', ['H2', 'P', 'V', 'W'], 'reaction', {'rho': 5.0}, 'F10 ≈ F3 ≈ Pr ≈ P > F1 > LS', W4)]

def score_variants(rows, title):
    lines = [f'## {title}', '', '| experiment | predicted | observed (median error) | τ |', '|---|---|---|---|']
    taus = []
    for name, phases, prob, knobs, pred, tagsym in rows:
        d = load_variants(phases, prob, knobs, tagsym)
        methods = [m.strip() for m in pred.replace('≈', '>').split('>') if m.strip()]
        if any((m not in d for m in methods)):
            continue
        _score_row(lines, taus, name, phases, prob, knobs, pred, d)
    if taus:
        lines.append(f'| **mean τ over {len(taus)} rankings** | | | **{np.mean(taus):+.2f}** |')
    return lines

def score(rows, title):
    lines = [f'## {title}', '', '| experiment | predicted | observed (median error) | τ |', '|---|---|---|---|']
    taus = []
    for row in rows:
        name, phase, prob, knobs, pred = row[:5]
        tags = row[5] if len(row) > 5 else None
        d = load(phase, prob, knobs, width=200 if 'w=200' in name else 50, tags=tags)
        _score_row(lines, taus, name, phase, prob, knobs, pred, d)
    lines.append(f'| **mean τ over {len(taus)} rankings** | | | **{np.mean(taus):+.2f}** |')
    return lines

def _score_row(lines, taus, name, phase, prob, knobs, pred, d):
    methods = [m for m in pred.replace('≈', '>').split('>') if m.strip()]
    methods = [m.strip() for m in methods]
    if phase == 'D2':
        d2 = defaultdict(list)
        for f in (root / phase).glob('*.json'):
            r = json.load(open(f))
            j = r['job']
            if j['problem'] != prob or any((j['knobs'].get(k) != v for k, v in knobs.items())):
                continue
            if j['method'] == 'plm' and j['tag'] != 'weighted_lag4_warm':
                continue
            d2[S[j['method']]].append(r['summary']['final_rel_l2'])
        d = d2
    obs, med = observed_ranks(d, methods)
    tau = kendalltau(pred_ranks(pred, methods), obs).statistic
    if np.isnan(tau):
        tau = 0.0
    taus.append(tau)
    obs_str = fmt_rank(methods, obs, med) + '  (' + ', '.join((f'{m} {med[m]:.0e}' for m in methods)) + ')'
    lines.append(f'| {name} | {pred} | {obs_str} | {tau:+.2f} |')
out = ['# Prediction scorecard', '', 'Observed ranking: methods ordered by median final error; two methods tie when their', "seed ranges overlap. Agreement = Kendall's τ-b. LS = frozen-feature least squares at its better feat_scale.", '']
out += score(ROUND1, 'Round 1 (predictions of 2026-11-05, Phases A–G)') + ['']
out += score(ROUND2, 'Rounds 2–3 (predictions of 2026-11-05/06, Phases H–P)') + ['']
out += score_variants(ROUND4, 'Round 4 (predictions of 2026-11-06, Phase V: P = PLM rec2, Pv = + varpro, Pr = + resampling, Pvr = both)') + ['']
out += score_variants(ROUND5, 'Round 5 (predictions of 2026-11-06, Phase W: F1 / F3 / F10 = two-stage at 10 / 30 / 100 % budget, Pr = rec2 + resampling)')
print('\n'.join(out))
