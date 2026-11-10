#!/usr/bin/env python3
"""
Load IVP-MCQ and score model answers.

  load(path)                    -> list of item dicts
  extract_letter(text)          -> 'A' | 'B' | 'C' | None   (parse a generation)
  letter_from_scores(scores)    -> letter with the highest score, e.g. the
                                   log-probs of the tokens 'A','B','C' after the
                                   prompt (recommended for small models: no
                                   parsing failures, no format sensitivity)
  evaluate(items, letters)      -> overall accuracy + per-slice breakdown,
                                   including how often the TRAP option is
                                   picked on B:unrelated items
"""
import json, re, sys
from collections import defaultdict

LETTERS = 'ABC'


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]


def extract_letter(text):
    if text is None:
        return None
    t = text.strip()
    m = re.match(r'^\(?([ABC])\)?[.):]?\s*$', t)            # "B", "B)", "(B)."
    if m:
        return m.group(1)
    m = re.match(r'^\(?([ABC])\)\s', t)                      # "B) because ..."
    if m:
        return m.group(1)
    m = re.search(r'(?:answer|option)\s*(?:is|:)?\s*\(?([ABC])\)?\b', t, re.I)
    if m:
        return m.group(1).upper()
    found = re.findall(r'\b([ABC])\b', t)
    return found[0] if len(set(found)) == 1 else None


def letter_from_scores(scores):
    """scores: dict {'A': s, 'B': s, 'C': s} or list aligned with A,B,C."""
    if isinstance(scores, dict):
        return max(LETTERS, key=lambda L: scores[L])
    return LETTERS[max(range(3), key=lambda i: scores[i])]


def evaluate(items, letters):
    slices = defaultdict(lambda: [0, 0])
    trap = [0, 0]
    for it, L in zip(items, letters):
        ok = (L == it['answer'])
        for key in ('all', f"variant={it['variant']}", f"ode_type={it['ode_type']}",
                    f"category={it['category']}",
                    f"subtype={it['variant']}:{it['subtype']}",
                    f"order={it['order']}"):
            slices[key][0] += ok
            slices[key][1] += 1
        if it['variant'] == 'B' and it['subtype'] == 'unrelated':
            trap[0] += (L == it['info']['trap_letter'])
            trap[1] += 1
    report = {k: {'acc': round(c / n, 4), 'n': n} for k, (c, n) in sorted(slices.items())}
    if trap[1]:
        report['B:unrelated trap-pick rate'] = {'rate': round(trap[0] / trap[1], 4),
                                                'n': trap[1]}
    return report['all']['acc'], report


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'ivp_mcq_dev.jsonl'
    items = load(path)
    print(f'loaded {len(items)} items from {path}\n')
    print(items[0]['prompt'])
    print('\ngold:', items[0]['answer'])
    # sanity: gold letters score 100%, a constant guess scores ~33%
    print('gold acc  :', evaluate(items, [i['answer'] for i in items])[0])
    print('always-A  :', evaluate(items, ['A'] * len(items))[0])
    print('parse test:', [extract_letter(s) for s in ['B', 'B)', '(C).', 'The answer is A', 'A or B']])
