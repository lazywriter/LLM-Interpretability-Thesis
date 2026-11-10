#!/usr/bin/env python3
"""
Independent verifier for IVP-MCQ. Uses ONLY the rendered text of each item
(equation, conditions, options) plus sympy.dsolve -- none of the generator's
internal constants -- and re-derives every label:

  1. every option solves the ODE                       (checkodesol)
  2. every size-n subset of the y-conditions is solved for the constants of
     dsolve's general solution; an option is "achievable" iff it equals the
     unique solution of some subset
  3. exactly one option is achievable, and it is the labelled answer
  4. variant C: the full set of y-conditions has no solution (truly inconsistent)
  5. variant B: the answer satisfies every y-condition; for 'unrelated', the
     trap option becomes achievable if z is (wrongly) read as y
Then prints balance statistics and writes distribution.json.

usage: python check_ivp_mcq.py ivp_mcq_all.jsonl [--outdir .]
"""
import argparse, itertools, json, os, re
from collections import Counter, defaultdict
import sympy as sp

x = sp.symbols('x')
y = sp.Function('y')
LOC = {'x': x, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'pi': sp.pi,
       'sqrt': sp.sqrt, 'E': sp.E}
COND_RE = re.compile(r"^(?:(-?\d+)\*)?([yz])('*)\((.+?)\) = (.+)$")
XS = [sp.Rational(3, 10), sp.Rational(7, 10), sp.Rational(13, 10)]


def parse_expr(s):
    return sp.sympify(s.replace('^', '**'), locals=LOC)


def parse_ode(text):
    lhs, rhs = text.split(' = ')
    def conv(t):
        t = t.replace("y''", 'D2').replace("y'", 'D1')
        t = re.sub(r'\by\b', 'Y0', t).replace('^', '**')
        return sp.sympify(t, locals={**LOC, 'D2': sp.Derivative(y(x), (x, 2)),
                                     'D1': sp.Derivative(y(x), x), 'Y0': y(x)})
    return sp.Eq(conv(lhs), conv(rhs))


def parse_cond(text):
    m = COND_RE.match(text)
    assert m, text
    k, fn, primes, pt, val = m.groups()
    return dict(scale=int(k) if k else 1, fn=fn, d=len(primes),
                p=parse_expr(pt), v=parse_expr(val))


def close(a, b):
    return abs(complex(sp.N(a - b, 40))) < 1e-18


def same_fn(e1, e2):
    return all(close(e1.subs(x, xv), e2.subs(x, xv)) for xv in XS)


_cache = {}
def general(ode_text):
    if ode_text not in _cache:
        ode = parse_ode(ode_text)
        sol = sp.dsolve(ode, y(x))
        consts = sorted(sol.rhs.free_symbols - {x}, key=str)
        _cache[ode_text] = (ode, sol.rhs, consts)
    return _cache[ode_text]


def subset_solution(G, consts, conds):
    eqs = [sp.Eq(c['scale'] * sp.diff(G, x, c['d']).subs(x, c['p']), c['v'])
           for c in conds]
    sols = sp.solve(eqs, consts, dict=True)
    if len(sols) != 1 or set(sols[0]) != set(consts):
        return None
    s = G.subs(sols[0])
    return None if (s.free_symbols - {x}) else s


def check(item):
    errs = []
    ode, G, consts = general(item['equation'])
    n = item['n_required']
    if len(consts) != n:
        errs.append(f'dsolve gave {len(consts)} constants, expected {n}')
    conds = [parse_cond(c['text']) for c in item['conditions']]
    opts = [parse_expr(o['text'].split('=', 1)[1]) for o in item['options']]
    for L, o in zip('ABC', opts):
        ok = sp.checkodesol(ode, sp.Eq(y(x), o))
        if not (ok[0] if isinstance(ok, tuple) else ok):
            errs.append(f'option {L} does not solve the ODE')

    def achievable_set(cs):
        sols = [s for sub in itertools.combinations(cs, n)
                if (s := subset_solution(G, consts, list(sub))) is not None]
        return [any(same_fn(o, s) for s in sols) for o in opts]

    ycs = [c for c in conds if c['fn'] == 'y']
    ach = achievable_set(ycs)
    if sum(ach) != 1:
        errs.append(f'{sum(ach)} achievable options (expected 1): {ach}')
    elif 'ABC'[ach.index(True)] != item['answer']:
        errs.append('achievable option is not the labelled answer')

    if item['variant'] == 'C':
        if sp.solve([sp.Eq(c['scale'] * sp.diff(G, x, c['d']).subs(x, c['p']), c['v'])
                     for c in ycs], consts, dict=True):
            errs.append('C item: conditions are jointly satisfiable')
    else:
        ans = opts['ABC'.index(item['answer'])]
        for c in ycs:
            if not close(c['scale'] * sp.diff(ans, x, c['d']).subs(x, c['p']), c['v']):
                errs.append('B item: answer violates a y-condition')
        if item['subtype'] == 'unrelated':
            misread = [dict(c, fn='y') for c in conds]
            t = 'ABC'.index(item['info']['trap_letter'])
            if not achievable_set(misread)[t]:
                errs.append('trap option not achievable under z->y misreading')
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path)]
    bad = 0
    for r in rows:
        e = check(r)
        if e:
            bad += 1
            print(r['id'], e)
    print(f'\nVERIFIED {len(rows) - bad}/{len(rows)} items '
          f'({len(_cache)} distinct equations solved with dsolve)')

    T = len(rows)
    def tab(key):
        c = Counter(key(r) for r in rows)
        return {str(k): {'count': v, 'pct': round(100 * v / T, 1)}
                for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
    dist = {
        'total': T,
        'by_split': tab(lambda r: r['split']),
        'by_variant': tab(lambda r: r['variant']),
        'by_ode_type': tab(lambda r: r['ode_type']),
        'by_category': tab(lambda r: r['category']),
        'by_variant_subtype': tab(lambda r: f"{r['variant']}:{r['subtype']}"),
        'by_answer_letter': tab(lambda r: r['answer']),
        'by_order': tab(lambda r: r['order']),
        'by_n_conditions': tab(lambda r: r['n_conditions']),
        'B_distractor_position': tab(lambda r: r['info'].get('distractor_position')
                                     if r['variant'] == 'B' else 'n/a'),
        'B_trap_letter': tab(lambda r: r['info'].get('trap_letter', 'n/a')),
        'option_roles': dict(Counter(o['role'] for r in rows for o in r['options'])),
        'condition_points': dict(Counter(c['point'] for r in rows for c in r['conditions'])),
        'condition_derivative_order': dict(Counter(c['derivative_order']
                                                   for r in rows for c in r['conditions'])),
    }
    # letter balance inside each (variant, ode_type) stratum
    strat = defaultdict(Counter)
    for r in rows:
        strat[f"{r['variant']}|{r['ode_type']}"][r['answer']] += 1
    dist['answer_letter_by_variant_ode_type'] = {k: dict(sorted(v.items()))
                                                 for k, v in sorted(strat.items())}
    # does the correct subset ever include non-textbook points? (C)
    dist['C_sufficient_set_uses_x0_only'] = dict(Counter(
        all(r['conditions'][i - 1]['point'] in ('0',) for i in r['info']['sufficient_positions'])
        for r in rows if r['variant'] == 'C'))
    with open(os.path.join(args.outdir, 'distribution.json'), 'w') as f:
        json.dump(dist, f, indent=2)
    for k in ('by_variant', 'by_ode_type', 'by_variant_subtype', 'by_answer_letter',
              'B_distractor_position', 'by_split'):
        print(f'\n== {k} ==')
        for kk, v in dist[k].items():
            print(f'  {kk:38s} {v["count"]:4d}  {v["pct"]:5.1f}%')
    print('\noption roles:', dist['option_roles'])
    print('C sufficient set only at x0=0:', dist['C_sufficient_set_uses_x0_only'])


if __name__ == '__main__':
    main()
