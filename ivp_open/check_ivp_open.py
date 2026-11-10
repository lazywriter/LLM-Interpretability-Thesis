#!/usr/bin/env python3
"""
Independent check of IVP-OPEN, from the rendered text (equation, condition
texts, answer_text) only; the generator's internals are not used.

For every item:
  1. the answer solves the ODE (SymPy residual)
  2. it satisfies every y-condition
  3. the y-conditions DETERMINE the solution (rank in dsolve's constants)
  4. distractor checks:
       none       exactly `order` conditions, no distractor
       duplicate  the distractor text equals another condition's text
       rescaled   without the distractor, the remaining y-conditions still
                  determine the same solution
       unrelated  the z-condition is the only z, the trap solution solves the
                  ODE, satisfies the z-condition read as y, and differs from
                  the answer
  5. the prompt contains the equation and every condition at its span, and
     the grader marks the true answer correct and the trap wrong

  python ivp_open/check_ivp_open.py ivp_open/ivp_open_all.jsonl
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import sympy as sp  # noqa: E402
from gold_check import parse, parse_condition, ode_residual, is_zero, determines, x  # noqa: E402
from grade_open import grade_reply  # noqa: E402


def check(it):
    errs = []
    conds = [parse_condition(c['text']) for c in it['conditions']]
    f = parse(it['answer_text'].split('=', 1)[1])
    if not is_zero(ode_residual(it['equation'], f)):
        errs.append('answer does not solve the ODE')
    ypos = [k for k, c in enumerate(conds, 1) if c['function'] == 'y']
    for k in ypos:
        c = conds[k - 1]
        if not is_zero(c['coef'] * sp.diff(f, x, c['deriv']).subs(x, c['point']) - c['value']):
            errs.append(f'answer violates condition {k}')
    if not determines(it['equation'], conds, ypos):
        errs.append('y-conditions do not determine the solution')
    roles = [c['role'] for c in it['conditions']]
    sub = it['subtype']
    if sub == 'none':
        if len(conds) != it['order'] or 'distractor' in roles:
            errs.append('none item has extra conditions')
    else:
        dk = roles.index('distractor') + 1
        if sub == 'duplicate':
            others = [c['text'] for k, c in enumerate(it['conditions'], 1) if k != dk]
            if it['conditions'][dk - 1]['text'] not in others:
                errs.append('duplicate is not a verbatim repeat')
        if sub == 'rescaled' and not determines(it['equation'], conds, [k for k in ypos if k != dk]):
            errs.append('rescaled: the remaining conditions do not determine y')
        if sub == 'unrelated':
            z = [k for k, c in enumerate(conds, 1) if c['function'] == 'z']
            if z != [dk]:
                errs.append('unrelated: z-condition is not the distractor')
            t = parse(it['info']['trap_solution'].split('=', 1)[1])
            zc = conds[dk - 1]
            if not is_zero(ode_residual(it['equation'], t)):
                errs.append('trap does not solve the ODE')
            if not is_zero(zc['coef'] * sp.diff(t, x, zc['deriv']).subs(x, zc['point']) - zc['value']):
                errs.append('trap does not satisfy the z-condition read as y')
            if is_zero(t - f):
                errs.append('trap equals the answer')
    p = it['prompt']
    if p[slice(*it['spans']['equation'])] != it['equation']:
        errs.append('equation span wrong')
    for (s, e), c in zip(it['spans']['conditions'], it['conditions']):
        if p[s:e] != c['text']:
            errs.append('condition span wrong')
    g = grade_reply(it, 'Some working.\nAnswer: ' + it['answer_text'])
    if not (g['correct'] and g['solves_ode'] and g['satisfies'] == ypos):
        errs.append(f'grader does not accept the true answer: {g}')
    if sub == 'unrelated':
        gt = grade_reply(it, 'Answer: ' + it['info']['trap_solution'])
        if gt['correct'] or not gt['equals_trap'] or not gt['z_as_y']:
            errs.append(f'grader mishandles the trap: {gt}')
    return errs


def main():
    items = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
    bad = 0
    for it in items:
        e = check(it)
        if e:
            bad += 1
            print(it['id'], it['subtype'], it['category'], e)
    print(f'VERIFIED {len(items) - bad}/{len(items)} items')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
