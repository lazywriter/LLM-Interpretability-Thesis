#!/usr/bin/env python3
"""
Attribution report, clean version (user decisions 2026-11-09).

- Target (a) only: the model's own final answer, teacher-forced on its own working. The control target (c) is
  not reported.
- Attention is the method used for every result. Attention rollout is shown only where it answers a different
  question (total flow through the layers: where the attention goes, and how much flows through the working). It is
  left out of every position comparison, because rollout favours early tokens by construction.
- Gradient×Input and Integrated Gradients appear only in the appendix figures (A1, A2): gradient methods measure the
  effect of infinitesimal changes to the input, and a differential equation is extremely sensitive to any change.

Every comparison is per item, summarised by the median with a bootstrap 95% CI (seed 0). The position/role effects
come from a within-item regression (item fixed effects), so position and role are estimated separately.

  python attribution_report.py [--root results/attribution] [--out figures/attribution/report_a]

Used by attribution_results_kaggle.ipynb (built by make_attribution_results_nb.py).
"""
import argparse, glob, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attribution_plots as AP                                    # noqa: E402  (run loading, densities, style)
from attribution_summary import field_values, prompt_fields       # noqa: E402

MODELS = AP.MODEL_ORDER
ML, MC, MM = AP.MODEL_LABEL, AP.MODEL_COLOR, AP.MODEL_MARKER
INK, INK2, GRID, SURFACE = AP.INK, AP.INK2, AP.GRID, AP.SURFACE
FIELDS = ['instruction', 'equation', 'sufficient', 'distractor', 'answer_instruction']
FIELD_LABEL = {'instruction': 'instruction', 'equation': 'equation', 'sufficient': 'useful\ncondition(s)',
               'distractor': 'distractor\ncondition', 'answer_instruction': 'answer-format\ninstruction'}
FIELD_NAME = {f: v.replace('\n', ' ') for f, v in FIELD_LABEL.items()}
# validated with the dataviz validator (all pairs, light surface): passes; CVD 6.1 on one pair, so every field also
# has its own line style
FIELD_COLOR = {'instruction': '#eda100', 'equation': '#6250d6', 'sufficient': '#008300', 'distractor': '#e87ba4',
               'answer_instruction': '#1baf7a'}
FIELD_STYLE = {'instruction': (0, (6, 2, 1, 2)), 'equation': '-', 'sufficient': '--', 'distractor': ':',
               'answer_instruction': '-.'}
METHOD_LABEL = {'attn': 'Attention', 'rollout': 'Attention rollout'}
N_BOOT = 2000


# ------------------------------------------------------------------ statistics
def _rng():
    return np.random.default_rng(0)


def med_ci(xs, log=True, n_boot=N_BOOT):
    """median with bootstrap 95% CI. log=True: xs are log2 values, the result is back on the ratio scale."""
    x = np.asarray([v for v in xs if v is not None and np.isfinite(v)], float)
    if len(x) < 3:
        return None
    b = np.median(x[_rng().integers(0, len(x), (n_boot, len(x)))], axis=1)
    f = (lambda v: float(2 ** v)) if log else float
    return {'est': f(np.median(x)), 'lo': f(np.percentile(b, 2.5)), 'hi': f(np.percentile(b, 97.5)), 'n': int(len(x))}


def fmt(s, pct=False):
    if s is None:
        return '–'
    if pct:
        return f"{s['est']:.0%} [{s['lo']:.0%}, {s['hi']:.0%}]"
    return f"{s['est']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"


def log2s(xs):
    return [math.log2(v) for v in xs if v is not None and np.isfinite(v) and v > 0]


# ------------------------------------------------------------------ loading
def load_runs(root):
    """the newest usable target-(a) run per model (folder names start with a UTC timestamp)."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, '*_a_test'))):
        if not os.path.exists(os.path.join(d, 'meta.json')):
            continue
        r = AP.load_run(d)
        if r['usable'] and r['target'] == 'a':
            runs[r['model']] = r
    return {m: runs[m] for m in MODELS if m in runs}


def run_table(runs):
    return [{'model': ML[m], 'items with a finished answer': len(r['rows']),
             'wrong answers': sum(1 for x in r['rows'] if x.get('correct') is False),
             'dtype': r['meta']['dtype'], 'run folder': os.path.basename(r['dir'])} for m, r in runs.items()]


# ------------------------------------------------------------------ plotting helpers
def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _model_legend(fig, models, y):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MC[m], marker=MM[m], ms=7, lw=2, label=ML[m]) for m in models]
    fig.legend(handles=h, loc='lower center', ncol=len(h), bbox_to_anchor=(0.5, y))


def _forest(ax, rows, models, values, lo=-2, hi=2):
    """rows: [(key, label)]; values[(key, model)] = med_ci dict (ratio scale)."""
    for i, (key, _) in enumerate(rows):
        for j, m in enumerate(models):
            s = values.get((key, m))
            if s is None:
                continue
            y = i + (j - (len(models) - 1) / 2) * 0.22
            a, b, e = (np.clip(np.log2(v), lo, hi) for v in (s['lo'], s['hi'], s['est']))
            ax.plot([a, b], [y, y], color=MC[m], lw=2, solid_capstyle='round', zorder=2)
            ax.plot(e, y, MM[m], ms=7, color=MC[m], zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows], fontsize=9)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis='y', visible=False)
    AP.ratio_axis(ax, lo, hi)


# ------------------------------------------------------------------ 1. where the attention goes
def where_attention_goes(plt, runs, out, stats):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#ecebe7', '#f2a27f', '#b8461a'])
    models = list(runs)
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.62 * len(models) + 2.4))
    for ax, meth in zip(axes, ('attn', 'rollout')):
        M = np.full((len(models), len(FIELDS)), np.nan)
        for i, m in enumerate(models):
            row = {'method': METHOD_LABEL[meth], 'model': ML[m]}
            for j, f in enumerate(FIELDS):
                s = med_ci(log2s([x.get(f) for x in runs[m]['per'][meth]]))
                stats[f'density | {meth} | {m} | {f}'] = s
                row[FIELD_NAME[f]] = fmt(s)
                if s:
                    M[i, j] = math.log2(s['est'])
            table.append(row)
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -2.5, 2.5), aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{2 ** M[i, j]:.2f}', ha='center', va='center', fontsize=10,
                            color='white' if abs(M[i, j]) > 1.3 else INK)
        ax.set_xticks(range(len(FIELDS)))
        ax.set_xticklabels([FIELD_LABEL[f] for f in FIELDS], fontsize=9)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([ML[m] for m in models] if ax is axes[0] else [])
        ax.grid(False)
        ax.set_title({'attn': 'Attention: direct, from the answer tokens',
                      'rollout': 'Attention rollout: through all layers'}[meth], pad=40)
    cb = fig.colorbar(im, ax=list(axes), orientation='horizontal', fraction=0.06, pad=0.06, aspect=50)
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels(['¼×', '½×', '1× (in proportion to its length)', '2×', '4×'])
    cb.outline.set_visible(False)
    fig.suptitle('Where the answer tokens look in the problem text (median density per field)', fontweight='bold', y=1.1)
    return _save(fig, out, '1_where_attention_goes.png'), table


def working_share(plt, runs, out, stats):
    models = list(runs)
    table = []
    fig, ax = plt.subplots(figsize=(8.5, 2.3))
    for i, meth in enumerate(('attn', 'rollout')):
        row = {'method': METHOD_LABEL[meth]}
        for j, m in enumerate(models):
            s = med_ci([x.get('working_share') for x in runs[m]['per'][meth]], log=False)
            stats[f'working share | {meth} | {m}'] = s
            row[ML[m]] = fmt(s, pct=True)
            y = i + (j - (len(models) - 1) / 2) * 0.22
            ax.plot([s['lo'], s['hi']], [y, y], color=MC[m], lw=2, solid_capstyle='round')
            ax.plot(s['est'], y, MM[m], ms=7, color=MC[m])
        table.append(row)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Attention (direct)', 'Attention rollout\n(through all layers)'])
    ax.set_ylim(1.5, -0.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='y', visible=False)
    ax.set_xlabel("share on the model's own working, out of working + problem text  (median, 95% CI)")
    ax.set_title("How much of the answer tokens' attention goes to the model's own working", loc='left')
    fig.tight_layout()
    _model_legend(fig, models, -0.22)
    return _save(fig, out, '2_working_share.png'), table


# ------------------------------------------------------------------ 2. the distractor, by order
ORDER_ROWS = [('control', 'Two useful conditions, no distractor:\nsecond ÷ first'),
              ('first', 'Distractor written FIRST:\ndistractor ÷ useful'),
              ('after', 'Distractor written AFTER a useful one:\ndistractor ÷ useful'),
              ('duplicate', 'Same condition written twice:\nsecond copy ÷ first copy')]


def _order_case(x):
    """('control' | 'first' | 'after' | 'duplicate', subtype, log2 ratio per token) for one item, or None."""
    r = x['_row']
    conds = sorted(x['_conds'], key=lambda g: int(g[5:]))
    dens = lambda g: x['_share'][g] / r['group_tokens'][g]
    st = r['subtype']
    if st == 'none' and len(conds) == 2:
        a, b = dens(conds[1]), dens(conds[0])
        return ('control', st, math.log2(a / b)) if a > 0 and b > 0 else None
    if st in ('rescaled', 'unrelated') and 'distr_log2' in x:
        d = [g for g in conds if r['roles'][g] == 'distractor'][0]
        return ('first' if conds.index(d) == 0 else 'after', st, x['distr_log2'])
    if st == 'duplicate' and 'dup_log2' in x:
        return ('duplicate', st, x['dup_log2'])
    return None


def distractor_order(plt, runs, out, stats):
    models = list(runs)
    vals, table, sub_table = {}, [], []
    cases = {m: [c for c in (_order_case(x) for x in runs[m]['per']['attn']) if c] for m in models}
    for key, lab in ORDER_ROWS:
        row = {'case': lab.replace('\n', ' ')}
        for m in models:
            s = med_ci([v for k, _, v in cases[m] if k == key])
            vals[(key, m)] = s
            stats[f'order | {key} | {m}'] = s
            row[ML[m]] = fmt(s) + (f"  (n={s['n']})" if s else '')
        table.append(row)
    for sub in ('rescaled', 'unrelated'):
        for key in ('first', 'after'):
            row = {'subtype': sub, 'distractor': {'first': 'written first', 'after': 'written after a useful one'}[key]}
            for m in models:
                s = med_ci([v for k, st, v in cases[m] if k == key and st == sub])
                stats[f'order | {sub} | {key} | {m}'] = s
                row[ML[m]] = fmt(s) + (f"  (n={s['n']})" if s else '')
            sub_table.append(row)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    _forest(ax, ORDER_ROWS, models, vals)
    ax.axhspan(1.5, 3.5, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.set_xlabel('ratio of attention per token, as named in each row  (median, 95% CI)')
    ax.set_title('Attention to a condition, by what it is and where it is written\n'
                 '(left of 1× = the condition named first in the row gets less)', loc='left')
    fig.tight_layout()
    _model_legend(fig, models, -0.1)
    return _save(fig, out, '3_distractor_order.png'), table, sub_table


# ------------------------------------------------------------------ 3. position vs role (within-item regression)
COEF_ROWS = [('pos2', 'Position 2 vs position 1\n(same role)'), ('pos3', 'Position 3 vs position 1\n(same role)'),
             ('d_first', 'Distractor written first\nvs a useful condition (same position)'),
             ('d_after', 'Distractor written after a useful one\nvs a useful condition (same position)')]


def _cond_matrix(r, layer=None):
    """rows [log2 density, pos2, pos3, distractor-first, distractor-after] for the conditions of one item."""
    P = prompt_fields(r)
    A = r['attn_layers']
    val = (lambda g: sum(A[g]) / len(A[g])) if layer is None else (lambda g: A[g][layer])
    tot = sum(val(g) for g in P)
    ntok = sum(r['group_tokens'][g] for g in P)
    conds = sorted([g for g in P if g.startswith('cond_')], key=lambda g: int(g[5:]))
    dpos = [k for k, g in enumerate(conds) if r['roles'][g] == 'distractor']
    after = bool(dpos) and dpos[0] > 0
    rows = []
    for k, g in enumerate(conds):
        dens = (val(g) / tot) / (r['group_tokens'][g] / ntok) if tot > 0 else 0
        if not (dens > 0 and math.isfinite(dens)):
            return None
        is_d = r['roles'][g] == 'distractor'
        rows.append([math.log2(dens), k == 1, k == 2, is_d and not after, is_d and after])
    return np.array(rows, float)


def _regression_items(run, layer=None):
    items = []
    for r in run['rows']:
        if r['subtype'] in ('none', 'rescaled', 'unrelated') and len(r['roles']) >= 2:
            M = _cond_matrix(r, layer)
            if M is not None:
                M[:, 0] -= M[:, 0].mean()                        # item fixed effect: demean within the item
                M[:, 1:] -= M[:, 1:].mean(0)
                items.append(M)
    return items


def _fit(items):
    M = np.concatenate(items)
    return np.linalg.lstsq(M[:, 1:], M[:, 0], rcond=None)[0]     # pos2, pos3, d_first, d_after (log2)


def _fit_ci(items, n_boot):
    b = _fit(items)
    rng = _rng()
    boots = np.array([_fit([items[i] for i in rng.integers(0, len(items), len(items))]) for _ in range(n_boot)])
    return b, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)


def position_vs_role(plt, runs, out, stats, n_boot=1000):
    models = list(runs)
    vals, table = {}, []
    for m in models:
        items = _regression_items(runs[m])
        b, lo, hi = _fit_ci(items, n_boot)
        for k, (key, _) in enumerate(COEF_ROWS):
            vals[(key, m)] = {'est': float(2 ** b[k]), 'lo': float(2 ** lo[k]), 'hi': float(2 ** hi[k]), 'n': len(items)}
            stats[f'regression | {key} | {m}'] = vals[(key, m)]
    for key, lab in COEF_ROWS:
        table.append({'effect': lab.replace('\n', ' '), **{ML[m]: fmt(vals[(key, m)]) for m in models}})
    table.append({'effect': 'items in the regression', **{ML[m]: str(vals[('pos2', m)]['n']) for m in models}})
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    _forest(ax, COEF_ROWS, models, vals, -1.5, 1.5)
    ax.axhspan(1.5, 3.5, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.set_xlabel('multiplicative effect on attention per token  (estimate, 95% bootstrap CI)')
    ax.set_title('Position and role separated: within-item regression\n'
                 'log2 attention density of a condition ~ position + role, item fixed effects', loc='left')
    fig.tight_layout()
    _model_legend(fig, models, -0.1)
    return _save(fig, out, '4_position_vs_role.png'), table


# ------------------------------------------------------------------ 4. layer by layer
def layers_fields(plt, runs, out, stats, n_split=200):
    from matplotlib.lines import Line2D
    models = list(runs)
    rel_table = []
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 3.9), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        prof, rel = {}, {'model': ML[m]}
        D = {f: [] for f in FIELDS}
        for r in runs[m]['rows']:
            P = prompt_fields(r)
            A = r['attn_layers']
            L = len(A['equation'])
            ntok = sum(r['group_tokens'][g] for g in P)
            conds = [g for g in P if g.startswith('cond_')]
            per = {f: [] for f in FIELDS}
            for l in range(L):
                tot = sum(A[g][l] for g in P)
                d = {g: (A[g][l] / tot) / (r['group_tokens'][g] / ntok) for g in P}
                per['instruction'].append(d['instruction'])
                per['equation'].append(d['equation'])
                per['answer_instruction'].append(d['answer_instruction'])
                per['sufficient'].append(np.mean([d[g] for g in conds if r['roles'][g] == 'sufficient']))
                ds = [d[g] for g in conds if r['roles'][g] == 'distractor']
                per['distractor'].append(ds[0] if ds else np.nan)
            for f in FIELDS:
                D[f].append(per[f])
        rng = _rng()
        for f in FIELDS:
            X = np.log2(np.array(D[f], float))
            med = np.nanmedian(X, 0)
            prof[f] = med
            ax.plot(np.arange(len(med)), 2 ** med, color=FIELD_COLOR[f], ls=FIELD_STYLE[f], lw=2)
            rs = []
            for _ in range(n_split):
                idx = rng.permutation(len(X))
                h1, h2 = idx[:len(X) // 2], idx[len(X) // 2:]
                rs.append(np.corrcoef(np.nanmedian(X[h1], 0), np.nanmedian(X[h2], 0))[0, 1])
            rel[FIELD_NAME[f]] = f'{np.median(rs):.2f}'
            stats[f'layers | {m} | {f}'] = [round(float(2 ** v), 4) for v in med]
        rel_table.append(rel)
        ax.axhline(1, color=INK2, lw=1)
        ax.set_yscale('log', base=2)
        ax.set_yticks([1 / 32, 1 / 16, 0.125, 0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticklabels(['1/32×', '1/16×', '⅛×', '¼×', '½×', '1×', '2×', '4×', '8×'])
        ax.set_xlabel('layer')
        ax.set_title(ML[m], fontsize=10)
    axes[0][0].set_ylabel('attention density (median over items)')
    h = [Line2D([], [], color=FIELD_COLOR[f], ls=FIELD_STYLE[f], lw=2, label=FIELD_NAME[f]) for f in FIELDS]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle('Layer by layer: which part of the problem the answer tokens attend to', fontweight='bold', y=1.0)
    fig.tight_layout()
    return _save(fig, out, '5_layers_fields.png'), rel_table


def layers_position_role(plt, runs, out, stats, n_boot=300):
    models = list(runs)
    panels = [(0, 'Position 2 vs position 1 (same role)'),
              (2, 'Distractor written first vs useful (same position)'),
              (3, 'Distractor written after a useful one vs useful (same position)')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for m in models:
        L = len(runs[m]['rows'][0]['attn_layers']['equation'])
        est, lo, hi = [], [], []
        for l in range(L):
            b, a, c = _fit_ci(_regression_items(runs[m], l), n_boot)
            est.append(b); lo.append(a); hi.append(c)
        est, lo, hi = np.array(est), np.array(lo), np.array(hi)
        depth = np.arange(L) / (L - 1)
        for ax, (k, _) in zip(axes, panels):
            ax.fill_between(depth, 2 ** lo[:, k], 2 ** hi[:, k], color=MC[m], alpha=0.15, lw=0)
            ax.plot(depth, 2 ** est[:, k], color=MC[m], lw=2)
        for k, key in ((0, 'pos2'), (2, 'd_first'), (3, 'd_after')):
            stats[f'layers regression | {key} | {m}'] = [round(float(2 ** v), 4) for v in est[:, k]]
    for ax, (_, title) in zip(axes, panels):
        ax.axhline(1, color=INK2, lw=1)
        ax.set_yscale('log', base=2)
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(['¼×', '½×', '1×', '2×', '4×'])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.set_xlabel('relative depth (first layer = 0%, last = 100%)')
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel('effect on attention per token\n(estimate, 95% CI)')
    fig.suptitle('Layer by layer: where position and role act (within-item regression per layer)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    _model_legend(fig, models, -0.1)
    return _save(fig, out, '6_layers_position_role.png')


# ------------------------------------------------------------------ 5. correct vs wrong
def correct_vs_wrong(plt, runs, out, stats, min_wrong=10, n_boot=N_BOOT):
    models = [m for m in runs if sum(1 for x in runs[m]['rows'] if x.get('correct') is False) >= min_wrong]
    skipped = [ML[m] for m in runs if m not in models]
    vals, table = {}, []
    for m in models:
        ok = [x for x in runs[m]['per']['attn'] if x['_row'].get('correct') is True]
        bad = [x for x in runs[m]['per']['attn'] if x['_row'].get('correct') is False]
        rng = _rng()
        for f in FIELDS:
            a, b = np.array(log2s([x.get(f) for x in ok])), np.array(log2s([x.get(f) for x in bad]))
            d = [np.median(b[rng.integers(0, len(b), len(b))]) - np.median(a[rng.integers(0, len(a), len(a))])
                 for _ in range(n_boot)]
            s = {'est': float(2 ** (np.median(b) - np.median(a))), 'lo': float(2 ** np.percentile(d, 2.5)),
                 'hi': float(2 ** np.percentile(d, 97.5)), 'n': f'{len(a)} correct / {len(b)} wrong'}
            vals[(f, m)] = s
            stats[f'wrong vs correct | {m} | {f}'] = s
            table.append({'model': ML[m], 'field': FIELD_NAME[f], 'correct (median)': f'{2 ** np.median(a):.2f}',
                          'wrong (median)': f'{2 ** np.median(b):.2f}', 'wrong ÷ correct [95% CI]': fmt(s),
                          'n': s['n']})
    fig, ax = plt.subplots(figsize=(9, 3.9))
    _forest(ax, [(f, FIELD_NAME[f]) for f in FIELDS], models, vals, -1.5, 1.5)
    ax.set_xlabel('attention density, wrong answers ÷ correct answers  (median ratio, 95% CI)')
    ax.set_title('Do wrong answers come with a different reading of the problem?', loc='left')
    fig.tight_layout()
    _model_legend(fig, models, -0.12)
    return _save(fig, out, '7_correct_vs_wrong.png'), table, skipped


# ------------------------------------------------------------------ appendix: gradient methods
def _gxi_distr(r):
    """per-item log2(distractor ÷ useful) with G×I (|Σ| per field), or None."""
    v = field_values(r, 'gxi')
    conds = [g for g in r['roles']]
    d = [g for g in conds if r['roles'][g] == 'distractor']
    s = [g for g in conds if r['roles'][g] == 'sufficient']
    if not d or not s:
        return None
    dd = v[d[0]] / r['group_tokens'][d[0]]
    ss = np.mean([v[g] / r['group_tokens'][g] for g in s])
    return math.log2(dd / ss) if dd > 0 and ss > 0 else None


def appendix_gxi(plt, runs, out, stats):
    models = list(runs)
    table = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={'width_ratios': [1.2, 1]})
    for j, m in enumerate(models):
        att = {x['_row']['id']: x.get('distr_log2') for x in runs[m]['per']['attn']}
        pairs = [(att.get(r['id']), _gxi_distr(r)) for r in runs[m]['rows'] if r['subtype'] in ('rescaled', 'unrelated')]
        pairs = np.array([p for p in pairs if p[0] is not None and p[1] is not None], float)
        rank = lambda a: np.argsort(np.argsort(a))
        rho = float(np.corrcoef(rank(pairs[:, 0]), rank(pairs[:, 1]))[0, 1])
        ax1.scatter(pairs[:, 0], pairs[:, 1], s=14, color=MC[m], alpha=0.6, lw=0, marker=MM[m],
                    label=f'{ML[m]}  (rank corr. {rho:.2f})')
        p = np.exp([r['logp_target'] for r in runs[m]['rows']])
        ax2.scatter(p, j + _rng().uniform(-0.2, 0.2, len(p)), s=12, color=MC[m], alpha=0.5, lw=0)
        ax2.plot(np.median(p), j, '|', color=INK, ms=18, mew=2.5)
        iqr = lambda v: float(np.subtract(*np.percentile(v, [75, 25])))
        canc = [abs(r['gxi'][g][0]) / r['gxi'][g][1] for r in runs[m]['rows'] for g in r['roles'] if r['gxi'][g][1] > 0]
        row = {'model': ML[m], 'spread of distractor ratio, G×I (IQR, log2)': f'{iqr(pairs[:, 1]):.2f}',
               'same, attention': f'{iqr(pairs[:, 0]):.2f}', 'rank corr. G×I vs attention': f'{rho:.2f}',
               'answers with p > 0.99': f'{np.mean(p > 0.99):.0%}',
               'G×I kept after cancellation, |Σ| ÷ Σ|·|': f'{np.median(canc):.2f}'}
        table.append(row)
        stats[f'appendix gxi | {m}'] = row
    lim = 5
    ax1.plot([-lim, lim], [-lim, lim], color=GRID, lw=1, zorder=0)
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.axvline(0, color=INK2, lw=0.8)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel('attention: log2(distractor ÷ useful), per item')
    ax1.set_ylabel('Gradient×Input: log2(distractor ÷ useful)')
    ax1.set_title('Item by item, G×I and attention do not agree', loc='left', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8.5)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([ML[m] for m in models])
    ax2.set_ylim(len(models) - 0.5, -0.5)
    ax2.set_xlim(0, 1.02)
    ax2.grid(axis='y', visible=False)
    ax2.set_xlabel("probability of the model's own answer, after its working\n(black tick = median)")
    ax2.set_title('The answer is almost certain, so its gradient is near zero', loc='left', fontsize=10)
    fig.suptitle('Appendix A1. Gradient×Input', fontweight='bold', y=1.02)
    fig.tight_layout()
    return _save(fig, out, 'A1_gradient_x_input.png'), table


def appendix_ig(plt, runs, out, stats):
    models = list(runs)
    table = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.6), gridspec_kw={'width_ratios': [1.4, 1]})
    labels = []
    y = 0
    for m in models:
        run = runs[m]
        rho, same, chance = AP.ig_baseline_agreement(run)
        row = {'model': ML[m], 'IG steps': run['meta']['ig_steps']}
        for b, name in (('ig_pad', 'pad'), ('ig_mask', 'zero')):
            g = np.array(run['gaps'][b])
            ax1.scatter(np.clip(g, 1e-3, 1e3), y + _rng().uniform(-0.18, 0.18, len(g)), s=10, color=MC[m], alpha=0.45, lw=0)
            ax1.plot(np.median(g), y, '|', color=INK, ms=16, mew=2.5)
            labels.append(f'{ML[m]} · {name} baseline')
            row[f'median completeness gap, {name}'] = f'{np.median(g):.0%}'
            y += 1
        k = models.index(m)
        ax2.barh(k, np.mean(same), height=0.55, color=MC[m])
        ax2.plot([np.mean(chance)] * 2, [k - 0.35, k + 0.35], color=INK, lw=1.5)
        ax2.text(np.mean(same) + 0.02, k, f'{np.mean(same):.0%}', va='center', fontsize=9)
        row['pad vs zero: rank corr. of fields (median)'] = f'{np.median(rho):.2f}'
        row['pad vs zero: same top field'] = f'{np.mean(same):.0%} (chance {np.mean(chance):.0%})'
        table.append(row)
        stats[f'appendix ig | {m}'] = row
    ax1.axvline(0.1, color=INK2, lw=1, ls='--')
    ax1.set_xscale('log')
    ax1.set_xlim(1e-3, 1e3)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_ylim(len(labels) - 0.5, -0.5)
    ax1.grid(axis='y', visible=False)
    ax1.set_xlabel('completeness gap per item (log scale; dashed = 10%)')
    ax1.set_title('IG does not converge', loc='left', fontsize=10)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([ML[m] for m in models])
    ax2.set_ylim(len(models) - 0.5, -0.5)
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax2.grid(axis='y', visible=False)
    ax2.set_xlabel('items where both baselines pick the same top field\n(black line = chance)')
    ax2.set_title('The two baselines disagree', loc='left', fontsize=10)
    fig.suptitle('Appendix A2. Integrated Gradients', fontweight='bold', y=1.04)
    fig.tight_layout()
    return _save(fig, out, 'A2_integrated_gradients.png'), table


# ------------------------------------------------------------------ everything
def summary_table(stats, runs):
    models = list(runs)
    g = lambda key, m, pct=False: fmt(stats.get(key.format(m=m)), pct)
    rows = [
        ("Attention on the model's own working (share)", 'working share | attn | {m}', True),
        ("Rollout on the model's own working (share)", 'working share | rollout | {m}', True),
        ('Position 2 vs 1, same role (regression)', 'regression | pos2 | {m}', False),
        ('Distractor written first vs useful, same position', 'regression | d_first | {m}', False),
        ('Distractor written after a useful one vs useful, same position', 'regression | d_after | {m}', False),
        ('Same condition twice: second copy ÷ first copy', 'order | duplicate | {m}', False),
        ('Wrong ÷ correct answers: attention on the equation', 'wrong vs correct | {m} | equation', False),
        ('Wrong ÷ correct answers: attention on the useful conditions', 'wrong vs correct | {m} | sufficient', False),
    ]
    return [{'result': lab, **{ML[m]: g(key, m, pct) for m in models}} for lab, key, pct in rows]


def build(root='results/attribution', out='figures/attribution/report_a'):
    os.makedirs(out, exist_ok=True)
    plt = AP.setup_mpl()
    runs = load_runs(root)
    assert runs, f'no usable target-(a) runs under {root}'
    stats, R = {}, {'runs': runs, 'figures': {}, 'tables': {}}
    R['tables']['runs'] = run_table(runs)
    R['figures']['where'], R['tables']['where'] = where_attention_goes(plt, runs, out, stats)
    R['figures']['working'], R['tables']['working'] = working_share(plt, runs, out, stats)
    R['figures']['order'], R['tables']['order'], R['tables']['order_subtype'] = distractor_order(plt, runs, out, stats)
    R['figures']['regression'], R['tables']['regression'] = position_vs_role(plt, runs, out, stats)
    R['figures']['layers'], R['tables']['layers_reliability'] = layers_fields(plt, runs, out, stats)
    R['figures']['layers_effects'] = layers_position_role(plt, runs, out, stats)
    R['figures']['wrong'], R['tables']['wrong'], R['skipped_wrong'] = correct_vs_wrong(plt, runs, out, stats)
    R['figures']['gxi'], R['tables']['gxi'] = appendix_gxi(plt, runs, out, stats)
    R['figures']['ig'], R['tables']['ig'] = appendix_ig(plt, runs, out, stats)
    R['tables']['summary'] = summary_table(stats, runs)
    R['stats'] = stats
    with open(os.path.join(out, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=1)
    try:
        import pandas as pd
        for name, t in R['tables'].items():
            pd.DataFrame(t).to_csv(os.path.join(out, f'table_{name}.csv'), index=False)
    except ImportError:
        pass
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/attribution')
    ap.add_argument('--out', default='figures/attribution/report_a')
    a = ap.parse_args()
    R = build(a.root, a.out)
    print('figures:', *R['figures'].values(), sep='\n  ')
