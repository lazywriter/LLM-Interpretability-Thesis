#!/usr/bin/env python3
"""
Easy-arithmetic control set in the IVP-MCQ format: a sanity check that the
letter-score pipeline reads correct answers out of the models when the maths
is trivial.

144 items, 3 options, exactly one correct, one split ('test'):
  6 categories x 24: add, sub, mul, div (exact), pow (small), mixed (a + b * c)
  answer letter 8/8/8 inside every category
  the correct value is the smallest / middle / largest option 8/8/8 inside every
  category (so "pick the middle number" scores 33%)
Distractors are near values (+-1 .. +-10) and typical slips (a*b for a**b,
(a + b) * c for a + b * c, ...).

Same fields the IVP pipeline reads: id, split, variant, subtype, ode_type,
category, order (= number of operations), n_conditions (0), prompt, answer,
answer_text, options[letter, text, role], spans (instruction, expression,
options, answer_instruction), info.

  python gen_easy_mcq.py            # writes easy_mcq/easy_mcq_test.jsonl (seed 7)
"""
import json, os, random
from collections import Counter

SEED = 7
N_PER_CAT = 24
INSTRUCTION = 'Compute the value of the expression below and choose the correct result.'
ANSWER_LINE = 'Answer with the letter of the correct option only.'
LETTERS = 'ABC'


def make(cat, rng):
    """(expression text, value, list of slip values)"""
    if cat == 'add':
        a, b = rng.randint(3, 60), rng.randint(3, 40)
        return f'{a} + {b}', a + b, [a + b + 10, a + b - 10]
    if cat == 'sub':
        b = rng.randint(3, 30); a = b + rng.randint(5, 50)
        return f'{a} - {b}', a - b, [a + b]
    if cat == 'mul':
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return f'{a} * {b}', a * b, [a * b + a, a * b - b, a + b]
    if cat == 'div':
        b, q = rng.randint(2, 10), rng.randint(4, 12)
        return f'{b * q} / {b}', q, [b * q - b, q * 2]
    if cat == 'pow':
        b, e = rng.choice([(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 2), (3, 3), (3, 4),
                           (4, 2), (4, 3), (5, 2), (5, 3), (6, 2), (6, 3), (7, 2), (8, 2), (9, 2), (10, 2),
                           (10, 3), (10, 4), (11, 2), (12, 2)])
        return f'{b} ** {e}', b ** e, [b * e, b ** (e - 1), b ** e + b]
    if cat == 'mixed':
        a, b, c = rng.randint(2, 20), rng.randint(2, 9), rng.randint(2, 9)
        return f'{a} + {b} * {c}', a + b * c, [(a + b) * c, a * b + c]
    raise ValueError(cat)


def pick_distractors(v, slips, rank, rng):
    """Two distinct positive distractors so that v is the `rank` ('low',
    'mid', 'high') value among the three options."""
    cands = set(slips) | {v + d for d in (1, 2, 3, 4, 5, 10)} | {v - d for d in (1, 2, 3, 4, 5, 10)}
    cands = sorted(c for c in cands if c > 0 and c != v)
    below, above = [c for c in cands if c < v], [c for c in cands if c > v]
    if rank == 'low' and len(above) >= 2:
        return rng.sample(above, 2)
    if rank == 'high' and len(below) >= 2:
        return rng.sample(below, 2)
    if rank == 'mid' and below and above:
        return [rng.choice(below), rng.choice(above)]
    return None


def build_prompt(expr, opts):
    spans = {}
    p = INSTRUCTION
    spans['instruction'] = [0, len(p)]
    p += '\n\nExpression: '
    spans['expression'] = [len(p), len(p) + len(expr)]
    p += expr + '\n\nOptions:\n'
    spans['options'] = []
    for L, t in zip(LETTERS, opts):
        p += f'{L}) '
        spans['options'].append([len(p), len(p) + len(t)])
        p += t + '\n'
    spans['answer_instruction'] = [len(p), len(p) + len(ANSWER_LINE)]
    p += ANSWER_LINE
    return p, spans


def main():
    rng = random.Random(SEED)
    cats = ['add', 'sub', 'mul', 'div', 'pow', 'mixed']
    rows, k = [], 0
    for cat in cats:
        # 24 slots: every letter 8x and every rank 8x (Latin-square cycling)
        letters = [LETTERS[i % 3] for i in range(N_PER_CAT)]
        ranks = [('low', 'mid', 'high')[(i // 3 + i) % 3] for i in range(N_PER_CAT)]
        slots = list(zip(letters, ranks)); rng.shuffle(slots)
        seen = set()
        for letter, rank in slots:
            for _ in range(1000):
                expr, v, slips = make(cat, rng)
                d = pick_distractors(v, slips, rank, rng)
                if d and expr not in seen:
                    break
            else:
                raise RuntimeError(f'no unique {cat} item with rank {rank}')
            seen.add(expr)
            vals = d[:]; rng.shuffle(vals)
            vals.insert(LETTERS.index(letter), v)
            opts = [str(x) for x in vals]
            prompt, spans = build_prompt(expr, opts)
            rows.append({
                'id': f'easy-{k:04d}', 'split': 'test', 'variant': 'E', 'subtype': cat,
                'ode_type': cat, 'category': cat, 'order': 2 if cat == 'mixed' else 1,
                'n_conditions': 0, 'n_required': 0, 'expression': expr, 'value': v,
                'conditions': [],
                'options': [{'letter': L, 'text': t, 'role': 'correct' if L == letter else 'distractor',
                             'satisfies_conditions': []} for L, t in zip(LETTERS, opts)],
                'answer': letter, 'answer_text': str(v), 'prompt': prompt, 'spans': spans,
                'info': {'correct_rank': rank},
            })
            k += 1
    os.makedirs('easy_mcq', exist_ok=True)
    with open('easy_mcq/easy_mcq_test.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {len(rows)} items to easy_mcq/easy_mcq_test.jsonl')


if __name__ == '__main__':
    main()
