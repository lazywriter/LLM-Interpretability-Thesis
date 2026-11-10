#!/usr/bin/env python3
"""
Independent check of IVP-MCQ gold answers, from the rendered text only.

For every option `y = f(x)` of an item:
  1. substitute f into the differential equation and simplify the residual
  2. evaluate every condition `k*y^(d)(x0) = v` on f; `z` conditions are about
     another function and never count for y
Then derive the answer without looking at the label:
  the option that solves the ODE and whose satisfied y-conditions DETERMINE the
  solution (they fix all integration constants of the general solution from
  `sympy.dsolve`: rank of the conditions in the constants = order; e.g.
  y'(1) and y'(-1) for y'' = f(x) fix only one constant). B: it must also
  satisfy all y-conditions. Exactly one option may qualify, and it must equal
  `answer`. The per-option condition sets are also compared with
  the dataset's `satisfies_conditions` metadata.

  python gold_check.py ivp_mcq/ivp_mcq_dev.jsonl            # summary
  python gold_check.py ivp_mcq/ivp_mcq_dev.jsonl --show 3   # + full tables
"""
import argparse, json, re, sys
import sympy as sp

x = sp.Symbol('x')
_F = sp.Function('f')
LOCALS = {'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'pi': sp.pi, 'x': x}
COND_RE = re.compile(r"^\s*(?:(-?\d+)\*)?([yz])('*)\(([^)]*)\)\s*=\s*(.+?)\s*$")


def parse(s):
    return sp.sympify(s.replace('^', '**'), locals=LOCALS)


def is_zero(e):
    e = sp.simplify(e)
    if e == 0:
        return True
    try:  # numeric fallback at a few points, for expressions simplify misses
        return all(abs(complex(e.subs(x, v).evalf())) < 1e-9 for v in (0.3, 1.7, 2.9))
    except (TypeError, ValueError):
        return False


def ode_residual(equation, f):
    lhs, rhs = equation.split('=')
    D2, D1, Y = sp.symbols('D2 D1 Y')
    expr = lambda t: parse(t.replace("y''", 'D2').replace("y'", 'D1').replace('y', 'Y'))
    r = expr(lhs) - expr(rhs)
    return r.subs({D2: sp.diff(f, x, 2), D1: sp.diff(f, x), Y: f})


_GENERAL = {}


def general_solution(equation):
    """General solution and its constants, from dsolve (cached per equation)."""
    if equation not in _GENERAL:
        lhs, rhs = equation.split('=')
        f = _F(x)
        expr = lambda t: parse(t.replace("y''", 'D2').replace("y'", 'D1').replace('y', 'Y')).subs(
            {sp.Symbol('D2'): f.diff(x, 2), sp.Symbol('D1'): f.diff(x), sp.Symbol('Y'): f})
        sol = sp.dsolve(sp.Eq(expr(lhs), expr(rhs)), f).rhs
        consts = sorted(sol.free_symbols - {x}, key=str)
        _GENERAL[equation] = (sol, consts)
    return _GENERAL[equation]


def determines(equation, conds, positions):
    """Do the conditions at `positions` (1-based) fix every constant?"""
    sol, consts = general_solution(equation)
    rows = [[sp.diff(conds[p - 1]['coef'] * sp.diff(sol, x, conds[p - 1]['deriv'])
                     .subs(x, conds[p - 1]['point']), c) for c in consts] for p in positions]
    return bool(rows) and sp.Matrix(rows).rank(simplify=True) == len(consts)


def parse_condition(text):
    m = COND_RE.match(text)
    if not m:
        raise ValueError(f'cannot parse condition {text!r}')
    k, fn, primes, pt, val = m.groups()
    return {'coef': int(k) if k else 1, 'function': fn, 'deriv': len(primes),
            'point': parse(pt), 'value': parse(val)}


def check_item(it):
    conds = [parse_condition(c['text']) for c in it['conditions']]
    rows = []
    for o in it['options']:
        f = parse(o['text'].split('=', 1)[1])
        solves = is_zero(ode_residual(it['equation'], f))
        sat = []
        for pos, c in enumerate(conds, 1):
            if c['function'] != 'y':
                continue
            lhs = c['coef'] * sp.diff(f, x, c['deriv']).subs(x, c['point'])
            if is_zero(lhs - c['value']):
                sat.append(pos)
        rows.append({'letter': o['letter'], 'text': o['text'], 'role': o['role'],
                     'solves_ode': solves, 'satisfies': sat,
                     'determined': determines(it['equation'], conds, sat),
                     'meta_satisfies': o['satisfies_conditions']})
    n_y = sum(c['function'] == 'y' for c in conds)
    need_all = it['variant'] == 'B'
    qualifying = [r['letter'] for r in rows if r['solves_ode'] and r['determined']
                  and (not need_all or len(r['satisfies']) == n_y)]
    derived = qualifying[0] if len(qualifying) == 1 else None
    return {
        'id': it['id'], 'variant': it['variant'], 'subtype': it['subtype'],
        'gold': it['answer'], 'derived': derived, 'qualifying': qualifying,
        'gold_ok': derived == it['answer'],
        'all_solve_ode': all(r['solves_ode'] for r in rows),
        'meta_ok': all(sorted(r['satisfies']) == sorted(r['meta_satisfies']) for r in rows),
        'need': f'all {n_y} y-conditions, determining' if need_all else 'a determining subset',
        'options': rows,
        'conditions': [(i, c['text'], c['role']) for i, c in enumerate(it['conditions'], 1)],
    }


def show(it, res):
    print(f"\n=== {it['id']}  variant {it['variant']}:{it['subtype']}  {it['ode_type']}")
    print(f"  equation: {it['equation']}")
    for i, t, role in res['conditions']:
        print(f'  cond {i}: {t:32s} [{role}]')
    print(f"  {'opt':4s}{'text':40s}{'solves ODE':12s}{'satisfies (mine)':18s}"
          f"{'determines y':14s}{'metadata':10s}role")
    for r in res['options']:
        print(f"  {r['letter']:4s}{r['text']:40s}{str(r['solves_ode']):12s}"
              f"{str(r['satisfies']):18s}{str(r['determined']):14s}{str(r['meta_satisfies']):10s}{r['role']}")
    print(f"  needs {res['need']} -> derived answer {res['derived']}, "
          f"gold {res['gold']}  {'OK' if res['gold_ok'] else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--show', type=int, default=0, help='print full tables for the first N items')
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.path, encoding='utf-8')]
    bad = []
    for k, it in enumerate(items):
        res = check_item(it)
        if k < a.show or not (res['gold_ok'] and res['meta_ok'] and res['all_solve_ode']):
            show(it, res)
        if not (res['gold_ok'] and res['meta_ok'] and res['all_solve_ode']):
            bad.append(it['id'])
    print(f'\n{len(items) - len(bad)}/{len(items)} items: gold re-derived from text, every option '
          f'solves the ODE, condition sets match metadata. Problems: {bad or "none"}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
