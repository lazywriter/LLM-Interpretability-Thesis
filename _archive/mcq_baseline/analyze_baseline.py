#!/usr/bin/env python3
"""
Item-by-item analysis of the canonical test runs (results/CANONICAL.json).

  1. headline numbers with 95% Wilson intervals; paired model comparison
     (exact McNemar test)
  2. overlap of the two models' correct / robust-correct item sets
  3. which WRONG option gets picked: trap, extra-match (satisfies a
     contradicting condition), near-miss (satisfies a sufficient condition, not
     enough to determine y) or far-miss (satisfies none), against what a
     random wrong pick would give
  4. difficulty: accuracy vs. value_complexity of the sufficient conditions,
     vs. using only x0 = 0, vs. number of conditions
  5. confidence: is p(chosen letter) informative about correctness?

Accuracy per item is the mean over the 6 option orders (n_correct / 6) where
noted, which removes letter bias and is less noisy than one order.

  python analyze_baseline.py            # prints the report, writes results/analysis/<stamp>/
"""
import datetime, json, math, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ivp_mcq'))
from grade_mcq import load, LETTERS  # noqa: E402

RES = os.path.join(HERE, 'results')


def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8')]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact p for b vs. c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def option_kind(it, letter):
    """correct / trap / extra_match / near_miss / far_miss."""
    o = next(o for o in it['options'] if o['letter'] == letter)
    if o['role'] in ('correct', 'trap'):
        return o['role']
    roles = {k: c['role'] for k, c in enumerate(it['conditions'], 1)}
    sat = [roles[k] for k in o['satisfies_conditions']]
    if 'contradicting' in sat:
        return 'extra_match'
    if sat:
        return 'near_miss'
    return 'far_miss'


def pct(x):
    return f'{100 * x:5.1f}'


def main():
    canon = json.load(open(os.path.join(RES, 'CANONICAL.json'), encoding='utf-8'))
    items = {it['id']: it for it in load(os.path.join(HERE, 'ivp_mcq', 'ivp_mcq_test.jsonl'))}
    M = {}
    for m, v in canon['models'].items():
        M[m] = {'base': {r['id']: r for r in jl(os.path.join(RES, v['baseline'], 'predictions.jsonl'))},
                'perm': {r['id']: r for r in jl(os.path.join(RES, v['option_permutations'], 'items.jsonl'))}}
    names = list(M)
    ids = sorted(items)
    out = {'models': names, 'n_items': len(ids)}
    P = print

    # 1 ---------------------------------------------------------------- headline
    P('=' * 90); P('1. HEADLINE (test, n=792)'); P('=' * 90)
    out['headline'] = {}
    for m in names:
        b, p = M[m]['base'], M[m]['perm']
        k = sum(b[i]['correct'] for i in ids)
        lo, hi = wilson(k, len(ids))
        mean6 = sum(p[i]['n_correct'] for i in ids) / (6 * len(ids))
        rob = sum(p[i]['robust_correct'] for i in ids)
        ge5 = sum(p[i]['n_correct'] >= 5 for i in ids)
        cw = sum(p[i]['content_consistent'] and not p[i]['robust_correct'] for i in ids)
        out['headline'][m] = dict(acc=k / len(ids), ci95=[lo, hi], acc_mean6=mean6, robust=rob, ge5=ge5,
                                  consistent_wrong=cw)
        P(f'{m:22s} acc {pct(k / len(ids))} [{pct(lo)},{pct(hi)}]  mean-over-6-orders {pct(mean6)}  '
          f'robust 6/6: {rob}  >=5/6: {ge5}  consistently-same-wrong-option: {cw}')
    a, c = names
    b01 = sum(M[a]['base'][i]['correct'] and not M[c]['base'][i]['correct'] for i in ids)
    b10 = sum(M[c]['base'][i]['correct'] and not M[a]['base'][i]['correct'] for i in ids)
    pv = mcnemar_exact(b01, b10)
    P(f'paired comparison (original order): only {a} right {b01}, only {c} right {b10}, exact McNemar p = {pv:.3f}')
    n6a = [M[a]['perm'][i]['n_correct'] for i in ids]
    n6c = [M[c]['perm'][i]['n_correct'] for i in ids]
    better = sum(x > y for x, y in zip(n6a, n6c)); worse = sum(x < y for x, y in zip(n6a, n6c))
    pv6 = mcnemar_exact(better, worse)
    P(f'paired, per-item n_correct of 6: {a} higher on {better}, {c} higher on {worse}, sign test p = {pv6:.3f}')
    out['paired'] = dict(only_first=b01, only_second=b10, mcnemar_p=pv, n6_first_higher=better,
                         n6_second_higher=worse, sign_p=pv6)

    # 2 ---------------------------------------------------------------- overlap
    P('\n' + '=' * 90); P('2. OVERLAP OF THE TWO MODELS'); P('=' * 90)
    out['overlap'] = {}
    for label, f in [('correct, original order', lambda m, i: M[m]['base'][i]['correct']),
                     ('robust 6/6', lambda m, i: M[m]['perm'][i]['robust_correct']),
                     ('>=5/6', lambda m, i: M[m]['perm'][i]['n_correct'] >= 5)]:
        A = {i for i in ids if f(a, i)}; C = {i for i in ids if f(c, i)}
        inter, union = len(A & C), len(A | C)
        # expected intersection if the two sets were independent
        exp = len(A) * len(C) / len(ids)
        out['overlap'][label] = dict(first=len(A), second=len(C), both=inter, either=union,
                                     expected_both_if_independent=exp, jaccard=inter / union if union else 0)
        P(f'{label:24s} {a}: {len(A):3d}  {c}: {len(C):3d}  both: {inter:3d} (independent: {exp:5.1f})  '
          f'either: {union:3d}  Jaccard {inter / union:.2f}')
    both_rob = sorted(i for i in ids if M[a]['perm'][i]['robust_correct'] and M[c]['perm'][i]['robust_correct'])
    out['robust_both_ids'] = both_rob
    bv = Counter(items[i]['variant'] for i in both_rob)
    P(f'robust for BOTH models: {len(both_rob)} items (B {bv["B"]}, C {bv["C"]}); by ode_type: '
      + ', '.join(f'{k} {v}' for k, v in sorted(Counter(items[i]['ode_type'] for i in both_rob).items())))
    # when both wrong, same wrong option?
    bw = [i for i in ids if not M[a]['base'][i]['correct'] and not M[c]['base'][i]['correct']]
    same_wrong = sum(M[a]['base'][i]['pred'] == M[c]['base'][i]['pred'] for i in bw)
    P(f'both wrong (original order): {len(bw)} items; same wrong option on {same_wrong} '
      f'({pct(same_wrong / len(bw))}%, 50% if unrelated)')
    out['both_wrong'] = dict(n=len(bw), same_option=same_wrong)

    # 3 ---------------------------------------------------------------- wrong-option kinds
    P('\n' + '=' * 90); P('3. WHICH WRONG OPTION IS PICKED'); P('=' * 90)
    P('kinds: trap (z read as y) | extra_match (satisfies a contradicting condition) |\n'
      '       near_miss (satisfies a sufficient condition, too few to determine y) | far_miss (none)')
    P('"random" = share expected if the wrong pick were random among the 2 wrong options\n')
    out['wrong_kinds'] = {}
    for m in names:
        b, p = M[m]['base'], M[m]['perm']
        out['wrong_kinds'][m] = {}
        for scope, sel in [('original order, all wrong picks', None), ('consistently-wrong items (same option x6)', 'cw')]:
            for var in ('B', 'C'):
                obs, exp, n = Counter(), Counter(), 0
                for i in ids:
                    it = items[i]
                    if it['variant'] != var:
                        continue
                    if sel is None:
                        if b[i]['correct']:
                            continue
                        pick = b[i]['pred']
                    else:
                        r = p[i]
                        if not (r['content_consistent'] and not r['robust_correct']):
                            continue
                        pick = r['majority_pick']
                    n += 1
                    obs[option_kind(it, pick)] += 1
                    for L in LETTERS:
                        if L != it['answer']:
                            exp[option_kind(it, L)] += 0.5
                kinds = [k for k in ('trap', 'extra_match', 'near_miss', 'far_miss') if exp[k] > 0]
                out['wrong_kinds'][m][f'{scope} | {var}'] = {'n': n, 'observed': dict(obs), 'random': dict(exp)}
                P(f'{m:22s} {var} {scope:42s} n={n:3d}  ' + '  '.join(
                    f'{k} {pct(obs[k] / n) if n else "  -  "}% (random {pct(exp[k] / n) if n else "  -  "})' for k in kinds))
        P()
    # near-miss by order (only 2nd order items have near misses in B)
    P('near_miss exists only where a single condition does not determine y (2nd order items).')

    # 4 ---------------------------------------------------------------- difficulty
    P('\n' + '=' * 90); P('4. WHAT MAKES AN ITEM HARD (accuracy = mean over 6 orders)'); P('=' * 90)
    feats = {}
    for i in ids:
        it = items[i]
        suff = [c for c in it['conditions'] if c['role'] == 'sufficient']
        feats[i] = {'vc_mean': sum(c['value_complexity'] for c in suff) / len(suff),
                    'vc_max': max(c['value_complexity'] for c in suff),
                    'x0_only_zero': all(c['point'] == '0' for c in suff),
                    'uses_derivative': any(c['derivative_order'] > 0 for c in suff),
                    'n_conditions': it['n_conditions'], 'order': it['order']}
    out['difficulty'] = {}
    for m in names:
        p = M[m]['perm']
        acc6 = {i: p[i]['n_correct'] / 6 for i in ids}
        res = {}
        for o in (1, 2):
            sub = [i for i in ids if feats[i]['order'] == o]
            rho = spearman([feats[i]['vc_max'] for i in sub], [acc6[i] for i in sub])
            res[f'spearman_vcmax_order{o}'] = rho
        P(f'{m}:')
        P(f'  Spearman(max value_complexity of sufficient conditions, accuracy): '
          f'order 1 {res["spearman_vcmax_order1"]:+.3f}, order 2 {res["spearman_vcmax_order2"]:+.3f}')
        for label, key, bins in [('max value_complexity', 'vc_max', [(0, 0), (1, 2), (3, 4), (5, 9)]),
                                 ('n_conditions', 'n_conditions', [(2, 2), (3, 3), (4, 4)])]:
            parts = []
            for lo, hi in bins:
                sub = [i for i in ids if lo <= feats[i][key] <= hi]
                if sub:
                    v = sum(acc6[i] for i in sub) / len(sub)
                    parts.append(f'{lo}-{hi}: {pct(v)} (n={len(sub)})' if lo != hi else f'{lo}: {pct(v)} (n={len(sub)})')
                    res[f'{key}_{lo}_{hi}'] = [v, len(sub)]
            P(f'  by {label:22s} ' + '   '.join(parts))
        for label, key in [('sufficient conditions all at x0=0', 'x0_only_zero'),
                           ('sufficient set uses a derivative condition', 'uses_derivative')]:
            for o in (1, 2):
                yes = [i for i in ids if feats[i][key] and feats[i]['order'] == o]
                no = [i for i in ids if not feats[i][key] and feats[i]['order'] == o]
                if yes and no:
                    vy, vn = sum(acc6[i] for i in yes) / len(yes), sum(acc6[i] for i in no) / len(no)
                    res[f'{key}_order{o}'] = {'yes': [vy, len(yes)], 'no': [vn, len(no)]}
                    P(f'  order {o}, {label:44s} yes {pct(vy)} (n={len(yes):3d})  no {pct(vn)} (n={len(no):3d})')
        out['difficulty'][m] = res

    # 5 ---------------------------------------------------------------- confidence
    P('\n' + '=' * 90); P('5. IS CONFIDENCE INFORMATIVE? (original order, p of chosen letter among A/B/C)'); P('=' * 90)
    out['confidence'] = {}
    for m in names:
        b, p = M[m]['base'], M[m]['perm']
        rows = []
        for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
            sub = [i for i in ids if lo <= b[i]['probs_over_letters'][b[i]['pred']] < hi]
            if sub:
                acc = sum(b[i]['correct'] for i in sub) / len(sub)
                rob = sum(p[i]['robust_correct'] for i in sub) / len(sub)
                rows.append((f'{lo:.1f}-{min(hi, 1):.1f}', len(sub), acc, rob))
        out['confidence'][m] = [dict(bin=r[0], n=r[1], acc=r[2], robust=r[3]) for r in rows]
        P(f'{m}:  ' + '   '.join(f'p {r[0]}: n={r[1]:3d} acc {pct(r[2])} robust {pct(r[3])}' for r in rows))

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    d = os.path.join(RES, 'analysis', f'{stamp}_baseline_items')
    os.makedirs(d, exist_ok=False)
    json.dump(out, open(os.path.join(d, 'analysis.json'), 'w'), indent=2)
    for m in names:  # item lists for attribution, per model
        p = M[m]['perm']
        lists = {'robust_6of6': sorted(i for i in ids if p[i]['robust_correct']),
                 'ge5of6': sorted(i for i in ids if p[i]['n_correct'] >= 5),
                 'consistently_wrong': sorted(i for i in ids if p[i]['content_consistent'] and not p[i]['robust_correct'])}
        json.dump(lists, open(os.path.join(d, f'item_sets_{m}.json'), 'w'), indent=1)
    P(f'\nwrote {d}')


if __name__ == '__main__':
    main()
