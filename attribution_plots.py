#!/usr/bin/env python3
"""
Figures and robust statistics for attribution runs (attribution.py).

Every comparison is computed per example and summarised by its MEDIAN with a
bootstrap 95% CI (2000 resamples, seed 0). The means in attribution_summary.py
are kept, but ratios are heavy-tailed (one example with a near-zero
denominator can dominate a mean), so the figures use medians of log2 ratios.

Figures (PNG, in --out):
  1_field_density.png     where the attribution goes: median density per prompt field
                          (1 = in proportion to the field's length), per model x method
  2_distractor.png        distractor condition vs the sufficient conditions, by subtype
  3_duplicate_position.png  second vs first copy of the repeated condition (duplicate items)
  4_attention_layers.png  attention density of equation / sufficient / distractor per layer
  5_working_share.png     target (a): share of the attribution that goes to the model's own working
  6_correct_vs_wrong.png  field densities for correct vs wrong answers (models with >= 10 wrong)
  7_ig_completeness.png   IG check: |sum of attributions - (f(x) - f(baseline))| / |f(x) - f(baseline)|
  8_ig_baselines.png      IG check: do the pad and zero baselines give the same field ranking per item?
  9_distractor_order.png  distractor ÷ useful condition, split by whether the distractor comes first or after a
                          useful one (control: 2nd ÷ 1st of two useful conditions)
Figures 1-6 show only the methods used for conclusions (MAIN: Gradient×Input, attention, rollout).
IG is computed but completeness-only (user decision 2026-11-04); it appears only in figures 7 and 8.
Also writes stats.json with every number that is plotted.

  python attribution_plots.py results/attribution/<run> [<run> ...] --out figures/attribution/<name>
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_summary import GRAD, METHODS, PROMPT_FIXED, field_values, prompt_fields  # noqa: E402

METHOD_LABEL = {'gxi': 'Gradient×Input', 'ig_pad': 'IG (pad baseline)', 'ig_mask': 'IG (zero baseline)',
                'attn': 'Attention', 'rollout': 'Attention rollout'}
MODEL_LABEL = {'qwen3-4b-instruct-2507': 'Qwen3-4B', 'qwen2.5-math-1.5b-instruct': 'Qwen2.5-Math-1.5B',
               'phi-4-mini-instruct': 'Phi-4-mini'}
MODEL_ORDER = ['qwen3-4b-instruct-2507', 'qwen2.5-math-1.5b-instruct', 'phi-4-mini-instruct']
MODEL_COLOR = {'qwen3-4b-instruct-2507': '#2a78d6', 'qwen2.5-math-1.5b-instruct': '#eb6834',
               'phi-4-mini-instruct': '#1baf7a'}
MODEL_MARKER = {'qwen3-4b-instruct-2507': 'o', 'qwen2.5-math-1.5b-instruct': 's', 'phi-4-mini-instruct': 'D'}
FIELD_COLOR = {'equation': '#6250d6', 'sufficient': '#008300', 'distractor': '#e87ba4'}
FIELD_STYLE = {'equation': '-', 'sufficient': '--', 'distractor': ':'}
MAIN = ('gxi', 'attn', 'rollout')            # methods shown in figures 1-6
FIELDS = ['instruction', 'equation', 'sufficient', 'distractor', 'answer_instruction']
FIELD_LABEL = {'instruction': 'instruction', 'equation': 'equation', 'sufficient': 'sufficient\ncondition(s)',
               'distractor': 'distractor\ncondition', 'answer_instruction': 'answer\ninstruction'}
TARGET_LABEL = {'c': 'target (c): true solution, no working', 'a': "target (a): model's own answer after its working"}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- per-example measures
def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def densities(r, method):
    """{field: density} over the prompt fields, plus the raw shares; None if unusable."""
    v = field_values(r, method)
    P = prompt_fields(r)
    if not all(finite(v.get(g)) for g in P):
        return None
    tot = sum(v[g] for g in P)
    ntok = sum(r['group_tokens'][g] for g in P)
    if tot <= 0:
        return None
    share = {g: v[g] / tot for g in P}
    dens = {g: share[g] / (r['group_tokens'][g] / ntok) for g in P}
    conds = [g for g in P if g.startswith('cond_')]
    suff = [g for g in conds if r['roles'][g] == 'sufficient']
    dist = [g for g in conds if r['roles'][g] == 'distractor']
    out = {'instruction': dens.get('instruction'), 'equation': dens.get('equation'),
           'answer_instruction': dens.get('answer_instruction'),
           'sufficient': float(np.mean([dens[g] for g in suff])) if suff else None,
           'distractor': dens[dist[0]] if dist else None, '_share': share, '_conds': conds}
    if 'working' in v and finite(v['working']):
        out['working_share'] = v['working'] / (v['working'] + tot)
    if dist and r['subtype'] == 'duplicate':
        d = dist[0]
        twins = [g for g in conds if r['condition_texts'][int(g[5:]) - 1] == r['condition_texts'][int(d[5:]) - 1]]
        if len(twins) == 2:
            a, b = sorted(twins, key=lambda g: int(g[5:]))
            if share[a] > 0 and share[b] > 0:
                out['dup_log2'] = math.log2(share[b] / share[a])
    if out['distractor'] and out['sufficient'] and out['distractor'] > 0 and out['sufficient'] > 0:
        out['distr_log2'] = math.log2(out['distractor'] / out['sufficient'])
    return out


def layer_densities(r):
    """attention density per layer for equation / sufficient / distractor: {field: [L]}."""
    A = r['attn_layers']
    P = prompt_fields(r)
    L = len(A[P[0]])
    ntok = sum(r['group_tokens'][g] for g in P)
    res = {'equation': [], 'sufficient': [], 'distractor': []}
    for l in range(L):
        tot = sum(A[g][l] for g in P)
        if not finite(tot) or tot <= 0:
            return None
        d = {g: (A[g][l] / tot) / (r['group_tokens'][g] / ntok) for g in P}
        conds = [g for g in P if g.startswith('cond_')]
        suff = [d[g] for g in conds if r['roles'][g] == 'sufficient']
        dist = [d[g] for g in conds if r['roles'][g] == 'distractor']
        res['equation'].append(d['equation'])
        res['sufficient'].append(float(np.mean(suff)))
        res['distractor'].append(dist[0] if dist else math.nan)
    return res


def med_ci(xs, n_boot=2000):
    xs = np.asarray([x for x in xs if finite(x)], float)
    if len(xs) == 0:
        return None
    if len(xs) == 1:
        return {'median': float(xs[0]), 'lo': float(xs[0]), 'hi': float(xs[0]), 'n': 1}
    b = np.median(RNG.choice(xs, size=(n_boot, len(xs)), replace=True), axis=1)
    return {'median': float(np.median(xs)), 'lo': float(np.percentile(b, 2.5)),
            'hi': float(np.percentile(b, 97.5)), 'n': int(len(xs))}


# ---------------------------------------------------------------- loading
def load_run(d):
    meta = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
    rows = [json.loads(l) for l in open(os.path.join(d, 'attributions.jsonl'), encoding='utf-8')]
    per = {m: [] for m in METHODS}
    for r in rows:
        for m in METHODS:
            x = densities(r, m)
            if x is not None:
                x['_row'] = r
                per[m].append(x)
    gaps = {}
    for b in ('ig_pad', 'ig_mask'):
        g = [abs(r[b + '_check']['delta']) / max(abs(r[b + '_check']['f_x'] - r[b + '_check']['f_baseline']), 1e-9)
             for r in rows if finite(r[b + '_check']['delta'])]
        gaps[b] = g
    usable = sum(len(v) for v in per.values())
    return {'dir': d, 'meta': meta, 'model': meta['model_key'], 'target': meta['target'], 'rows': rows,
            'per': per, 'gaps': gaps, 'usable': usable > 0,
            'converged': {b: (bool(g) and float(np.median(g)) <= 0.10) for b, g in gaps.items()}}


# ---------------------------------------------------------------- plotting helpers
def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9.5,
        'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 2})
    return plt


def models_present(runs):
    return [m for m in MODEL_ORDER if any(r['model'] == m and r['usable'] for r in runs)]


def get_run(runs, model, target):
    return next((r for r in runs if r['model'] == model and r['target'] == target), None)


def hollow(run, method):
    return method in ('ig_pad', 'ig_mask') and not run['converged'][method]


def ratio_axis(ax, lo=-2.5, hi=2.5):
    ticks = [t for t in (-3, -2, -1, 0, 1, 2, 3) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{2.0 ** t:g}×' for t in ticks])
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.set_xlim(lo, hi)


def forest(ax, runs, target, methods, key, select=lambda r: True, lo=-2.5, hi=2.5):
    """median log2 ratio with CI: one row per method, one point per model."""
    ms = models_present(runs)
    stats = {}
    for i, meth in enumerate(methods):
        for j, mk in enumerate(ms):
            run = get_run(runs, mk, target)
            if run is None or not run['usable']:
                continue
            s = med_ci([x.get(key) for x in run['per'][meth] if select(x['_row'])])
            if s is None:
                continue
            stats[f'{mk} | {meth}'] = s
            y = i + (j - (len(ms) - 1) / 2) * 0.22
            c = MODEL_COLOR[mk]
            ax.plot([max(s['lo'], lo), min(s['hi'], hi)], [y, y], color=c, lw=2, solid_capstyle='round', zorder=2)
            ax.plot(np.clip(s['median'], lo, hi), y, MODEL_MARKER[mk], ms=7, zorder=3,
                    mfc=SURFACE if hollow(run, meth) else c, mec=c, mew=1.6)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in methods])
    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.grid(axis='y', visible=False)
    ratio_axis(ax, lo, hi)
    return stats


def model_legend(fig, runs, extra_hollow=True, y=0.005):
    from matplotlib.lines import Line2D
    ms = models_present(runs)
    h = [Line2D([], [], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=7, lw=2, label=MODEL_LABEL[m]) for m in ms]
    if extra_hollow:
        h.append(Line2D([], [], color=INK2, marker='o', ms=7, lw=0, mfc=SURFACE, mew=1.6,
                        label='hollow = IG not converged (median gap > 10%)'))
    fig.legend(handles=h, loc='lower center', ncol=len(h), bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- figures
def fig_field_density(plt, runs, out, stats):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#ecebe7', '#f2a27f', '#b8461a'])
    targets = [t for t in ('c', 'a') if any(r['target'] == t and r['usable'] for r in runs)]
    ms = models_present(runs)
    fig, axes = plt.subplots(1, len(targets), figsize=(6.4 * len(targets), 0.42 * len(ms) * len(MAIN) + 2.2),
                             squeeze=False)
    for ax, t in zip(axes[0], targets):
        rows, labels = [], []
        for mk in ms:
            run = get_run(runs, mk, t)
            if run is None or not run['usable']:
                continue
            for meth in MAIN:
                vals = []
                for f in FIELDS:
                    s = med_ci([x.get(f) for x in run['per'][meth]])
                    stats[f'{t} | {mk} | {meth} | {f}'] = s
                    vals.append(s['median'] if s else math.nan)
                rows.append(vals)
                labels.append(f"{MODEL_LABEL[mk]} · {METHOD_LABEL[meth]}")
        M = np.log2(np.array(rows))
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -2, 2), aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if math.isfinite(M[i, j]):
                    ax.text(j, i, f'{2 ** M[i, j]:.2f}', ha='center', va='center', fontsize=8.5,
                            color='white' if abs(M[i, j]) > 1.2 else INK)
        ax.set_xticks(range(len(FIELDS)))
        ax.set_xticklabels([FIELD_LABEL[f] for f in FIELDS])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0][0] else [], fontsize=8.5)
        for k in range(1, len(rows) // len(MAIN)):
            ax.axhline(k * len(MAIN) - 0.5, color=SURFACE, lw=4)
        ax.grid(False)
        ax.set_title(TARGET_LABEL[t], pad=34)
    cb = fig.colorbar(im, ax=axes[0].tolist(), orientation='horizontal', fraction=0.035, pad=0.04, aspect=50)
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels(['¼×', '½×', '1× (in proportion to length)', '2×', '4×'])
    cb.outline.set_visible(False)
    fig.suptitle('Where the attribution goes: median density per prompt field', fontweight='bold', y=1.07)
    fig.text(0.5, -0.02, 'density = share of the attribution ÷ share of the prompt tokens.  '
             'Rollout has a known early-token bias (high on the instruction).', ha='center', color=INK2, fontsize=8.5)
    fig.savefig(os.path.join(out, '1_field_density.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_distractor(plt, runs, out, stats):
    targets = [t for t in ('c', 'a') if any(r['target'] == t and r['usable'] for r in runs)]
    subs = ['duplicate', 'rescaled', 'unrelated']
    fig, axes = plt.subplots(len(targets), 4, figsize=(15, 2.0 * len(targets) + 1.1), squeeze=False, sharex=True)
    for i, t in enumerate(targets):
        for j, sub in enumerate(['all'] + subs):
            ax = axes[i][j]
            sel = (lambda r: r['subtype'] != 'none') if sub == 'all' else (lambda r, s=sub: r['subtype'] == s)
            st = forest(ax, runs, t, list(MAIN), 'distr_log2', sel)
            stats.update({f'{t} | distr/suff {sub} | {k}': v for k, v in st.items()})
            ax.set_title(('all distractor items' if sub == 'all' else sub) + f'  ·  target ({t})', fontsize=9.5)
            if j:
                ax.set_yticklabels([])
    fig.supxlabel('distractor density ÷ sufficient-condition density  (median, 95% CI)', y=0.0, fontsize=9.5,
                  color=INK2)
    fig.suptitle('Does the distractor condition get less attribution than the conditions that determine the answer?\n'
                 'left of 1× = the distractor gets less', fontweight='bold', y=1.02)
    fig.tight_layout()
    model_legend(fig, runs, extra_hollow=False, y=-0.08)
    fig.savefig(os.path.join(out, '2_distractor.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_duplicate(plt, runs, out, stats):
    targets = [t for t in ('c', 'a') if any(r['target'] == t and r['usable'] for r in runs)]
    fig, axes = plt.subplots(1, len(targets), figsize=(5.6 * len(targets), 2.8), squeeze=False, sharex=True)
    for ax, t in zip(axes[0], targets):
        st = forest(ax, runs, t, list(MAIN), 'dup_log2', lambda r: r['subtype'] == 'duplicate', -3, 3)
        stats.update({f'{t} | dup 2nd/1st | {k}': v for k, v in st.items()})
        ax.set_title(TARGET_LABEL[t], fontsize=9.5)
        ax.set_xlabel('second copy ÷ first copy  (median, 95% CI)')
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    fig.suptitle('Duplicate items: the same condition written twice. Any difference is a position effect.\n'
                 'left of 1× = the later copy gets less', fontweight='bold', y=1.04)
    fig.tight_layout()
    model_legend(fig, runs, extra_hollow=False, y=-0.14)
    fig.savefig(os.path.join(out, '3_duplicate_position.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_layers(plt, runs, out, stats):
    from matplotlib.lines import Line2D
    targets = [t for t in ('c', 'a') if any(r['target'] == t and r['usable'] for r in runs)]
    ms = [m for m in models_present(runs) if any(r['model'] == m and r['usable'] for r in runs)]
    fig, axes = plt.subplots(len(targets), len(ms), figsize=(4.6 * len(ms), 3.0 * len(targets) + 0.6),
                             squeeze=False, sharey=True)
    for i, t in enumerate(targets):
        for j, mk in enumerate(ms):
            ax = axes[i][j]
            run = get_run(runs, mk, t)
            if run is None or not run['usable']:
                ax.set_visible(False)
                continue
            L = [x for x in (layer_densities(r) for r in run['rows']) if x]
            for f in ('equation', 'sufficient', 'distractor'):
                M = np.array([x[f] for x in L], float)
                med = np.nanmedian(M, axis=0)
                q1, q3 = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
                xs = np.arange(len(med))
                ax.fill_between(xs, q1, q3, color=FIELD_COLOR[f], alpha=0.12, lw=0)
                ax.plot(xs, med, FIELD_STYLE[f], color=FIELD_COLOR[f])
                stats[f'{t} | {mk} | attention by layer | {f}'] = [round(float(v), 4) for v in med]
            ax.axhline(1, color=INK2, lw=1)
            ax.set_yscale('log', base=2)
            ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4])
            ax.set_yticklabels(['⅛×', '¼×', '½×', '1×', '2×', '4×'])
            ax.set_title(f'{MODEL_LABEL[mk]}  ·  target ({t})', fontsize=9.5)
            if i == len(targets) - 1:
                ax.set_xlabel('layer')
            if j == 0:
                ax.set_ylabel('attention density (median)')
    h = [Line2D([], [], color=FIELD_COLOR[f], ls=FIELD_STYLE[f], lw=2, label=lab) for f, lab in
         (('equation', 'equation'), ('sufficient', 'sufficient condition(s)'), ('distractor', 'distractor condition'))]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Attention from the target tokens to each prompt field, layer by layer (band = middle 50% of items)',
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out, '4_attention_layers.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_working(plt, runs, out, stats):
    ms = [m for m in models_present(runs) if (r := get_run(runs, m, 'a')) and r['usable']]
    if not ms:
        return
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for i, meth in enumerate(MAIN):
        for j, mk in enumerate(ms):
            run = get_run(runs, mk, 'a')
            s = med_ci([x.get('working_share') for x in run['per'][meth]])
            if s is None:
                continue
            stats[f'a | working share | {mk} | {meth}'] = s
            y = i + (j - (len(ms) - 1) / 2) * 0.22
            c = MODEL_COLOR[mk]
            ax.plot([s['lo'], s['hi']], [y, y], color=c, lw=2, solid_capstyle='round')
            ax.plot(s['median'], y, MODEL_MARKER[mk], ms=7, mfc=SURFACE if hollow(run, meth) else c, mec=c, mew=1.6)
    ax.set_yticks(range(len(MAIN)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in MAIN])
    ax.set_ylim(len(MAIN) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='y', visible=False)
    ax.set_xlabel("share of the attribution on the model's own working (rest: prompt fields; median, 95% CI)")
    ax.set_title("Target (a): how much of the final answer is explained by the working vs the problem text",
                 loc='left')
    fig.tight_layout()
    model_legend(fig, [get_run(runs, m, 'a') for m in ms], extra_hollow=False, y=-0.2)
    fig.savefig(os.path.join(out, '5_working_share.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_correct_wrong(plt, runs, out, stats, min_wrong=10):
    from matplotlib.lines import Line2D
    panels = [(r['model'], r['target']) for r in runs if r['usable']
              and sum(1 for x in r['rows'] if x.get('correct') is False) >= min_wrong]
    panels.sort(key=lambda p: (MODEL_ORDER.index(p[0]), p[1] != 'c'))
    if not panels:
        return
    meths = list(MAIN)
    fields = ['equation', 'sufficient', 'distractor']
    fig, axes = plt.subplots(len(panels), len(meths), figsize=(12, 2.9 * len(panels) + 0.6), squeeze=False, sharey=True)
    for i, (mk, t) in enumerate(panels):
        run = get_run(runs, mk, t)
        for j, meth in enumerate(meths):
            ax = axes[i][j]
            for k, f in enumerate(fields):
                for dx, ok in ((-0.13, True), (0.13, False)):
                    s = med_ci([x.get(f) for x in run['per'][meth] if x['_row'].get('correct') is ok])
                    if s is None:
                        continue
                    stats[f'{t} | {mk} | {meth} | {f} | {"correct" if ok else "wrong"}'] = s
                    c = MODEL_COLOR[mk]
                    ax.plot([k + dx] * 2, [s['lo'], s['hi']], color=c, lw=2, solid_capstyle='round')
                    ax.plot(k + dx, s['median'], 'o' if ok else 'X', ms=8, color=c, mec=c,
                            mfc=c if ok else SURFACE, mew=1.6)
            ax.axhline(1, color=INK2, lw=1)
            ax.set_yscale('log', base=2)
            ax.set_yticks([0.25, 0.5, 1, 2, 4])
            ax.set_yticklabels(['¼×', '½×', '1×', '2×', '4×'])
            ax.set_xticks(range(len(fields)))
            ax.set_xticklabels(['equation', 'sufficient', 'distractor'])
            ax.grid(axis='x', visible=False)
            nw = sum(1 for x in run['rows'] if x.get('correct') is False)
            ax.set_title(f'{MODEL_LABEL[mk]} · target ({t}) · {METHOD_LABEL[meth]}'
                         + (' *' if hollow(run, meth) else '') + f'\n{len(run["rows"]) - nw} correct, {nw} wrong',
                         fontsize=9)
            if j == 0:
                ax.set_ylabel('density (median, 95% CI)')
    h = [Line2D([], [], color=INK2, marker='o', ms=8, lw=0, label='answer correct'),
         Line2D([], [], color=INK2, marker='X', ms=8, lw=0, mfc=SURFACE, mew=1.6, label='answer wrong')]
    fig.legend(handles=h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f'Correct vs wrong answers (runs with at least {min_wrong} wrong answers)', fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out, '6_correct_vs_wrong.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_ig(plt, runs, out, stats):
    fig, ax = plt.subplots(figsize=(8, 0.5 * 2 * len(runs) + 1.4))
    labels, y = [], 0
    for run in sorted(runs, key=lambda r: (r['target'], MODEL_ORDER.index(r['model']))):
        for b in ('ig_pad', 'ig_mask'):
            g = np.array(run['gaps'][b])
            lab = f"{MODEL_LABEL[run['model']]} · ({run['target']}) · {'pad' if b == 'ig_pad' else 'zero'}" \
                  f" · {run['meta']['ig_steps']} steps"
            labels.append(lab)
            if len(g):
                jit = RNG.uniform(-0.18, 0.18, len(g))
                c = MODEL_COLOR[run['model']]
                ax.scatter(np.clip(g, 1e-3, 1e3), y + jit, s=10, color=c, alpha=0.45, lw=0)
                ax.plot(np.median(g), y, '|', color=INK, ms=16, mew=2.5)
                stats[f'ig gap | {lab}'] = {'median': float(np.median(g)), 'share_under_10pct': float(np.mean(g < 0.1)),
                                            'n': int(len(g))}
            else:
                ax.text(1e-3, y, '  no finite values (run unusable)', va='center', color=INK2, fontsize=8.5)
            y += 1
    ax.axvline(0.1, color=INK2, lw=1, ls='--')
    ax.text(0.1, -0.9, ' 10%', color=INK2, fontsize=8.5)
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1e3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(len(labels) - 0.5, -1.2)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('relative completeness gap per item (log scale; black tick = median)')
    ax.set_title('IG did not converge: the attributions do not add up to f(input) − f(baseline)\n'
                 '(relative completeness gap; left of the dashed 10% line = acceptable)', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, '7_ig_completeness.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


ORDER_ROWS = [('control', 'control: two useful conditions,\nno distractor (2nd ÷ 1st)'),
              ('first', 'distractor written FIRST\n(distractor ÷ useful)'),
              ('after', 'distractor written AFTER a\nuseful condition (distractor ÷ useful)')]


def order_measure(x):
    """('control' | 'first' | 'after', log2 ratio) for one example, or None.
    control = subtype none with two (sufficient) conditions: density of condition 2 / condition 1.
    first / after = rescaled or unrelated items, by the distractor's place among the conditions
    (duplicate items are left out: there the distractor is a copy of a useful condition, see figure 3)."""
    r = x['_row']
    conds = sorted(x['_conds'], key=lambda g: int(g[5:]))
    dens = lambda g: x['_share'][g] / r['group_tokens'][g]
    if r['subtype'] == 'none' and len(conds) == 2:
        a, b = dens(conds[1]), dens(conds[0])
        return ('control', math.log2(a / b)) if a > 0 and b > 0 else None
    if r['subtype'] in ('rescaled', 'unrelated') and 'distr_log2' in x:
        d = [g for g in conds if r['roles'][g] == 'distractor'][0]
        return ('first' if conds.index(d) == 0 else 'after', x['distr_log2'])
    return None


def fig_order(plt, runs, out, stats):
    targets = [t for t in ('c', 'a') if any(r['target'] == t and r['usable'] for r in runs)]
    meths = ['attn', 'gxi']
    ms = models_present(runs)
    fig, axes = plt.subplots(len(meths), len(targets), figsize=(5.8 * len(targets), 2.6 * len(meths) + 1.0),
                             squeeze=False, sharex=True)
    for i, meth in enumerate(meths):
        for j, t in enumerate(targets):
            ax = axes[i][j]
            for k, (key, _) in enumerate(ORDER_ROWS):
                for m, mk in enumerate(ms):
                    run = get_run(runs, mk, t)
                    if run is None or not run['usable']:
                        continue
                    vals = [om[1] for om in (order_measure(x) for x in run['per'][meth]) if om and om[0] == key]
                    s = med_ci(vals)
                    if s is None:
                        continue
                    stats[f'{t} | order {key} | {mk} | {meth}'] = s
                    y = k + (m - (len(ms) - 1) / 2) * 0.22
                    c = MODEL_COLOR[mk]
                    ax.plot([max(s['lo'], -2), min(s['hi'], 2)], [y, y], color=c, lw=2, solid_capstyle='round')
                    ax.plot(np.clip(s['median'], -2, 2), y, MODEL_MARKER[mk], ms=7, color=c)
            ax.axhspan(1.5, 2.5, color=GRID, alpha=0.35, lw=0, zorder=0)
            ax.set_yticks(range(len(ORDER_ROWS)))
            ax.set_yticklabels([lab for _, lab in ORDER_ROWS] if j == 0 else [], fontsize=8.5)
            ax.set_ylim(len(ORDER_ROWS) - 0.5, -0.5)
            ax.grid(axis='y', visible=False)
            ratio_axis(ax, -2, 2)
            ax.set_title(f'{METHOD_LABEL[meth]}  ·  target ({t})', fontsize=9.5)
    fig.supxlabel('ratio per token, as named in each row  (median, 95% CI)', y=0.02, fontsize=9.5, color=INK2)
    fig.suptitle('The distractor is discounted only when it comes AFTER a useful condition\n'
                 '(rescaled + unrelated items; left of 1× = gets less)', fontweight='bold', y=1.02)
    fig.tight_layout()
    model_legend(fig, runs, extra_hollow=False, y=-0.06)
    fig.savefig(os.path.join(out, '9_distractor_order.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def ig_baseline_agreement(run):
    """per item: rank correlation of per-token field attributions, pad vs zero baseline, whether both
    baselines put the most attribution on the same field, and the chance level of that (1 / number of fields)."""
    rho, same, chance = [], [], []
    rank = lambda a: np.argsort(np.argsort(a))
    for r in run['rows']:
        P = prompt_fields(r)
        a = np.array([field_values(r, 'ig_pad').get(g, math.nan) for g in P], float)
        b = np.array([field_values(r, 'ig_mask').get(g, math.nan) for g in P], float)
        if len(P) < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        n = np.array([r['group_tokens'][g] for g in P], float)
        a, b = a / n, b / n
        rho.append(float(np.corrcoef(rank(a), rank(b))[0, 1]))
        same.append(bool(np.argmax(a) == np.argmax(b)))
        chance.append(1 / len(P))
    return rho, same, chance


def fig_ig_baselines(plt, runs, out, stats):
    rs = sorted([r for r in runs if r['usable']], key=lambda r: (r['target'], MODEL_ORDER.index(r['model'])))
    if not rs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.42 * len(rs) + 1.9), sharey=True,
                                   gridspec_kw={'width_ratios': [1.6, 1]})
    labels = []
    for y, run in enumerate(rs):
        rho, same, chance = ig_baseline_agreement(run)
        c = MODEL_COLOR[run['model']]
        labels.append(f"{MODEL_LABEL[run['model']]} · target ({run['target']})")
        if not rho:
            continue
        ax1.scatter(rho, y + RNG.uniform(-0.18, 0.18, len(rho)), s=10, color=c, alpha=0.45, lw=0)
        ax1.plot(np.median(rho), y, '|', color=INK, ms=16, mew=2.5)
        ax2.barh(y, np.mean(same), height=0.55, color=c)
        ax2.plot([np.mean(chance)] * 2, [y - 0.35, y + 0.35], color=INK, lw=1.5)
        ax2.text(np.mean(same) + 0.02, y, f'{np.mean(same):.0%}', va='center', fontsize=8.5, color=INK)
        stats[f'ig baselines | {labels[-1]}'] = {'spearman_median': float(np.median(rho)),
                                                 'top_field_agree': float(np.mean(same)),
                                                 'chance': float(np.mean(chance)), 'n': len(rho)}
    ax1.axvline(0, color=INK2, lw=1)
    ax1.axvline(1, color=INK2, lw=1, ls='--')
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_ylim(len(labels) - 0.5, -0.5)
    ax1.grid(axis='y', visible=False)
    ax1.set_xlabel('rank correlation of field attributions, pad vs zero baseline\n'
                   '(per item; black tick = median; 1 = same ranking)')
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax2.grid(axis='y', visible=False)
    ax2.set_xlabel('items where both baselines pick the same top field\n(black line = chance)')
    fig.suptitle('IG depends on an arbitrary baseline: the pad and zero baselines explain the same item differently',
                 fontweight='bold', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, '8_ig_baselines.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', default='figures/attribution')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = [load_run(d) for d in args.runs]
    for r in runs:
        if not r['usable']:
            print(f"WARNING {os.path.basename(r['dir'])}: no finite attributions (non-finite forward pass?); "
                  'shown only in the IG check figure')
    plt = setup_mpl()
    stats = {'runs': {os.path.basename(r['dir']): {'model': r['model'], 'target': r['target'], 'n': len(r['rows']),
                                                   'usable': r['usable'], 'ig_converged': r['converged']}
                      for r in runs}}
    fig_field_density(plt, runs, args.out, stats)
    fig_distractor(plt, runs, args.out, stats)
    fig_duplicate(plt, runs, args.out, stats)
    fig_layers(plt, runs, args.out, stats)
    fig_working(plt, runs, args.out, stats)
    fig_correct_wrong(plt, runs, args.out, stats)
    fig_order(plt, runs, args.out, stats)
    fig_ig(plt, runs, args.out, stats)
    fig_ig_baselines(plt, runs, args.out, stats)
    with open(os.path.join(args.out, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=1)
    print('wrote figures and stats.json to', args.out)


if __name__ == '__main__':
    main()
