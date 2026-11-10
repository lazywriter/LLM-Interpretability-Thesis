#!/usr/bin/env python3
"""
Generate IVP-OPEN: open-ended initial value problems (no options), for
measuring how well each model solves them when it may show its working.

Problems are built with the same construction as IVP-MCQ (imported from
ivp_mcq/gen_ivp_mcq_dataset.py): every family has a general solution that is
affine in its constants, the true constants are small non-zero integers, and
condition values are derived from them, so the labels are exact.

Design (864 items by default)
  18 categories (6 ode_types x 3), as in IVP-MCQ
  x 4 subtypes:
      none       exactly the n conditions needed (a plain IVP; control)
      duplicate  + one condition restated verbatim
      rescaled   + one condition restated as k*y(x0) = k*v
      unrelated  + a condition on a different function z; `info.trap_solution`
                 is what you get if you wrongly read z as y
  x 12 items per (category, subtype)
  dev: 1 per (category, subtype) = 72, test 792
  distractor position balanced inside every (category, subtype)

Prompt: solve, show the working, end with "Answer: y = <expression>" in plain
text. `spans` marks the instruction, the equation, each condition and the
answer instruction (character offsets into `prompt`).

  python gen_ivp_open_dataset.py                 # -> ivp_open/ (seed 23)
"""
import argparse, json, os, random, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ivp_mcq'))
import sympy as sp  # noqa: E402
from gen_ivp_mcq_dataset import (CATEGORIES, fam, build_B, cond_slots, pick_sufficient,  # noqa: E402
                                 rand_consts, make_cond, sol_expr, txt, point_txt, rank_ok,
                                 satisfies)

PROMPT_VERSION = 'open-v1'
INSTRUCTION = 'Solve the initial value problem below.'
ANSWER_INSTRUCTION = ('Show your working, then finish with a final line of the form '
                      '"Answer: y = <expression>", writing the expression in plain text '
                      '(for example: y = 3*exp(2*x) - x^2).')
SUBTYPES = ['none', 'duplicate', 'rescaled', 'unrelated']


def render(ode_text, cond_texts):
    spans = {}
    s = INSTRUCTION
    spans['instruction'] = [0, len(s)]
    s += '\n\nDifferential equation: '
    a = len(s); s += ode_text; spans['equation'] = [a, len(s)]
    s += '\nConditions:\n'
    spans['conditions'] = []
    for i, c in enumerate(cond_texts):
        s += f'{i + 1}. '
        a = len(s); s += c; spans['conditions'].append([a, len(s)])
        s += '\n'
    s += '\n'
    a = len(s); s += ANSWER_INSTRUCTION; spans['answer_instruction'] = [a, len(s)]
    return s, spans


def build(rng, F, sub):
    """(conditions, true constants, info) or None."""
    n = F['n']
    if sub == 'none':
        S = pick_sufficient(rng, F, cond_slots(F))
        if S is None:
            return None
        C = rand_consts(rng, n)
        return [make_cond(F, d, p, C, role='sufficient') for d, p in S], C, {'distractor_kind': None}
    out = build_B(rng, F, sub)
    if out is None:
        return None
    conds, opts, info = out
    C = opts[0][1]
    if sub == 'unrelated':
        T = next(o[1] for o in opts if o[0] == 'trap')
        info['trap_constants'] = [int(t) for t in T]
        info['trap_solution'] = 'y = ' + txt(sol_expr(F, T))
    return conds, C, info


def make_item(rng, cat, sub, dpos):
    for _ in range(300):
        F = fam(rng, cat)
        out = build(rng, F, sub)
        if out is None:
            continue
        conds, C, info = out
        n = F['n']
        ycond = [c for c in conds if c['fn'] == 'y']
        # the y-conditions determine the solution, and the true one satisfies all
        if not rank_ok([c['a'] for c in ycond], n) or not all(satisfies(c, C) for c in ycond):
            continue
        if sub == 'unrelated' and info['trap_solution'] == 'y = ' + txt(sol_expr(F, C)):
            continue
        real = [c for c in conds if c['role'] != 'distractor']
        dis = [c for c in conds if c['role'] == 'distractor']
        rng.shuffle(real)
        if dis:
            pos = dpos % (len(real) + 1)
            ordered = real[:pos] + dis + real[pos:]
            info['distractor_position'] = pos + 1
        else:
            ordered = real
            info['distractor_position'] = None
        prompt, spans = render(F['ode_text'], [c['text'] for c in ordered])
        return dict(
            variant='B', subtype=sub, ode_type=F['ode_type'], category=cat, order=n,
            n_required=n, n_conditions=len(ordered), equation=F['ode_text'],
            conditions=[dict(text=c['text'], role=c['role'], function=c['fn'],
                             derivative_order=c['d'], point=point_txt(c['p']),
                             value_complexity=int(sp.count_ops(c['raw_v']))) for c in ordered],
            answer_text='y = ' + txt(sol_expr(F, C)), constants=[int(c) for c in C],
            prompt=prompt, prompt_version=PROMPT_VERSION, spans=spans, info=info,
            sympy_ode=sp.srepr(F['ode']), sympy_solution=sp.srepr(sol_expr(F, C)),
            sympy_general_solution=sp.srepr(sol_expr(F, sp.symbols(f'C1:{n + 1}'))),
        )
    raise RuntimeError(f'could not build {cat} {sub}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-cell', type=int, default=12, help='items per (category, subtype)')
    ap.add_argument('--dev-per-cell', type=int, default=1)
    ap.add_argument('--seed', type=int, default=23)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'ivp_open'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()
    items, seen = [], set()
    for otype, cats in CATEGORIES.items():
        for cat in cats:
            for sub in SUBTYPES:
                for k in range(args.per_cell):
                    for _ in range(50):
                        it = make_item(rng, cat, sub, dpos=k)
                        key = (it['equation'], tuple(c['text'] for c in it['conditions']))
                        if key not in seen:
                            break
                    else:
                        raise RuntimeError(f'no unique item for {cat} {sub}')
                    seen.add(key)
                    items.append(it)
            print(f'  built {cat:24s} total={len(items)}', flush=True)

    by_cell = defaultdict(list)
    for it in items:
        by_cell[(it['category'], it['subtype'])].append(it)
    dev, test = [], []
    for rows in by_cell.values():
        rng.shuffle(rows)
        dev += rows[:args.dev_per_cell]
        test += rows[args.dev_per_cell:]
    for split, rows in (('dev', dev), ('test', test)):
        for r in rows:
            r['split'] = split
    for i, r in enumerate(sorted(dev + test, key=lambda r: (r['split'], r['ode_type'], r['category'],
                                                            SUBTYPES.index(r['subtype'])))):
        r['id'] = f'ivo-{i:04d}'

    def dump(rows, name):
        with open(os.path.join(args.outdir, name), 'w', encoding='utf-8') as f:
            for r in sorted(rows, key=lambda r: r['id']):
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    dump(dev + test, 'ivp_open_all.jsonl')
    dump(dev, 'ivp_open_dev.jsonl')
    dump(test, 'ivp_open_test.jsonl')
    dist = {'split': Counter(r['split'] for r in items), 'subtype': Counter(r['subtype'] for r in items),
            'ode_type': Counter(r['ode_type'] for r in items), 'order': Counter(r['order'] for r in items),
            'n_conditions': Counter(r['n_conditions'] for r in items),
            'distractor_position': Counter(str(r['info']['distractor_position']) for r in items)}
    json.dump({k: dict(v) for k, v in dist.items()}, open(os.path.join(args.outdir, 'distribution.json'), 'w'),
              indent=2)
    print(f'DONE {len(items)} items (dev {len(dev)}, test {len(test)}) in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
