#!/usr/bin/env python3
"""
Summarise attribution runs (attribution.py) at the field level.

For every example and method, the attribution to the PROMPT fields
(instruction, equation, each condition, answer instruction, labels) is turned
into shares. Gradient methods use |sum of the field's token attributions|;
attention methods use attention mass. Shares are reported two ways:
  share     the field's part of the total over the prompt fields
  density   share / (field tokens / prompt tokens): 1 = in proportion to its length

Main comparisons (the thesis questions in CLAUDE.md):
  distractor vs sufficient   density of the distractor condition / mean density of the
                             sufficient conditions (< 1: the distractor gets less)
  duplicate copies           second copy / first copy of the repeated condition
                             (identical text, so any difference is a position effect)
  working (target a)         part of the total that goes to the model's own working

Each comparison is broken down by subtype, by correct vs wrong, and for
`unrelated` items by whether the answer fell into the z-trap.

  python attribution_summary.py results/attribution/<run> [<run> ...]
Writes <run>/summary.json and prints the tables.
"""
import json, math, os, sys
from collections import defaultdict

PROMPT_FIXED = ('instruction', 'equation', 'answer_instruction', 'prompt_other')
GRAD = ('gxi', 'ig_pad', 'ig_mask')
METHODS = GRAD + ('attn', 'rollout')


def field_values(r, method):
    """{group: non-negative magnitude} for one example and method."""
    if method in GRAD:
        return {g: abs(v[0]) for g, v in r[method].items()}   # |signed sum| per field
    if method == 'attn':
        return {g: sum(v) / len(v) for g, v in r['attn_layers'].items()}
    return dict(r['rollout'])


def prompt_fields(r):
    return [g for g in r['group_tokens'] if g in PROMPT_FIXED or g.startswith('cond_')]


def example_measures(r, method):
    v = field_values(r, method)
    P = prompt_fields(r)
    tot = sum(v[g] for g in P)
    ntok = sum(r['group_tokens'][g] for g in P)
    if tot <= 0:
        return None
    share = {g: v[g] / tot for g in P}
    dens = {g: share[g] / (r['group_tokens'][g] / ntok) for g in P}
    conds = [g for g in P if g.startswith('cond_')]
    suff = [g for g in conds if r['roles'][g] == 'sufficient']
    dist = [g for g in conds if r['roles'][g] == 'distractor']
    m = {'share_' + k: share[k] for k in PROMPT_FIXED if k in share}
    m.update({'density_' + k: dens[k] for k in PROMPT_FIXED if k in dens})
    m['share_conditions'] = sum(share[g] for g in conds)
    m['density_sufficient'] = sum(dens[g] for g in suff) / len(suff)
    if dist:
        d = dist[0]
        m['density_distractor'] = dens[d]
        m['distractor_over_sufficient'] = dens[d] / m['density_sufficient']
        if r['subtype'] == 'duplicate':
            twins = [g for g in conds if r['condition_texts'][int(g[5:]) - 1] == r['condition_texts'][int(d[5:]) - 1]]
            if len(twins) == 2:
                first, second = sorted(twins, key=lambda g: int(g[5:]))
                m['duplicate_second_over_first'] = share[second] / share[first] if share[first] > 0 else math.nan
    if 'working' in v:
        m['share_working'] = v['working'] / (v['working'] + tot)
    return m


def mean_sem(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {'mean': mu, 'sem': sd / math.sqrt(n), 'n': n}


def slices(r):
    out = ['all', f"subtype={r['subtype']}", f"order={r['order']}"]
    if r.get('correct') is not None:
        out.append('correct' if r['correct'] else 'wrong')
    if r['subtype'] == 'unrelated' and r.get('equals_trap') is not None:
        out.append('unrelated: trap' if r['equals_trap'] else 'unrelated: no trap')
    return out


def summarise_run(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, 'attributions.jsonl'), encoding='utf-8')]
    meta = json.load(open(os.path.join(run_dir, 'meta.json'), encoding='utf-8'))
    acc = defaultdict(lambda: defaultdict(list))         # (method, slice) -> measure -> values
    for r in rows:
        for m in METHODS:
            ms = example_measures(r, m)
            if ms is None:
                continue
            for s in slices(r):
                for k, v in ms.items():
                    acc[(m, s)][k].append(v)
    report = {f'{m} | {s}': {k: mean_sem(v) for k, v in d.items()} for (m, s), d in acc.items()}
    ig = {}
    for b in ('ig_pad', 'ig_mask'):
        rel = sorted(abs(r[b + '_check']['delta']) / max(abs(r[b + '_check']['f_x'] - r[b + '_check']['f_baseline']), 1e-9)
                     for r in rows)
        ig[b] = {'median_relative_gap': rel[len(rel) // 2] if rel else None,
                 'share_under_10pct': sum(x < .1 for x in rel) / len(rel) if rel else None}
    summary = {'meta': meta, 'n_examples': len(rows), 'ig_completeness': ig, 'report': report}
    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary


def fmt(x):
    return '   -    ' if x is None else f"{x['mean']:5.2f}±{x['sem']:.2f}"


def print_summary(s):
    m = s['meta']
    print(f"\n=== {m['model_key']} | target {m['target']} | {m['split']} | {s['n_examples']} examples | "
          f"IG gap (median, relative): pad {s['ig_completeness']['ig_pad']['median_relative_gap']:.3f}, "
          f"mask {s['ig_completeness']['ig_mask']['median_relative_gap']:.3f}")
    cols = [('density_equation', 'equation'), ('density_sufficient', 'sufficient'),
            ('density_distractor', 'distractor'), ('distractor_over_sufficient', 'distr/suff'),
            ('duplicate_second_over_first', 'dup 2nd/1st'), ('share_conditions', 'cond share'),
            ('share_working', 'working')]
    rep = s['report']
    slices_seen = sorted({k.split(' | ')[1] for k in rep}, key=lambda x: (x != 'all', x))
    for meth in METHODS:
        print(f'\n  {meth}  (density: 1 = in proportion to the field length)')
        print('  ' + 'slice'.ljust(22) + ''.join(c[1].rjust(13) for c in cols))
        for sl in slices_seen:
            d = rep.get(f'{meth} | {sl}')
            if d:
                print('  ' + sl.ljust(22) + ''.join(fmt(d.get(c[0])).rjust(13) for c in cols))


if __name__ == '__main__':
    for run in sys.argv[1:]:
        print_summary(summarise_run(run))
