#!/usr/bin/env python3
"""
Map IVP-OPEN prompt fields (`spans`) to token positions in a model input.

The dataset stores character spans of the instruction, the equation, each
condition and the answer instruction inside `item['prompt']`. The model sees
the prompt inside its chat template (shifted by `prompt_offset`, from
baseline_letter_logprobs.build_text), possibly followed by more text (the
model's working, a target answer). These helpers turn character ranges into
token indices via the fast tokenizer's offset_mapping; each token goes to the
group it overlaps most, and tokens touching no group go to 'other'.

  groups = item_char_groups(item, prompt_offset)            # [(name, start, end), ...]
  enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
  tokmap = char_groups_to_tokens(enc['offset_mapping'], groups, text)   # {name: [token idx]}

Field names: 'instruction', 'equation', 'cond_1' ... 'cond_k',
'answer_instruction'; `condition_roles(item)` gives {'cond_k': role}.
Text of the prompt outside the fields ("Differential equation: ",
"Conditions:", "1. ") is 'prompt_other'; the chat template is 'template'.

  python field_tokens.py --model qwen2.5-1.5b-instruct     # self-check on all 864 prompts
"""
import argparse, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS_FIXED = ('instruction', 'equation', 'answer_instruction')


def condition_roles(item):
    return {f'cond_{k}': c['role'] for k, c in enumerate(item['conditions'], 1)}


def item_char_groups(item, prompt_offset=0, text_len=None):
    """Character groups (name, start, end) of one item inside a model input."""
    sp, o = item['spans'], prompt_offset
    groups = [(f, sp[f][0] + o, sp[f][1] + o) for f in FIELDS_FIXED]
    groups += [(f'cond_{k}', s + o, e + o) for k, (s, e) in enumerate(sp['conditions'], 1)]
    # the rest of the prompt (labels, numbering, blank lines) and the chat template around it
    taken = sorted((s, e) for _, s, e in groups)
    pos, p_end = o, o + len(item['prompt'])
    for s, e in taken:
        if s > pos:
            groups.append(('prompt_other', pos, s))
        pos = max(pos, e)
    if p_end > pos:
        groups.append(('prompt_other', pos, p_end))
    if o > 0:
        groups.append(('template', 0, o))
    if text_len is not None and text_len > p_end:
        groups.append(('template', p_end, text_len))
    return groups


def char_groups_to_tokens(offsets, groups, text=None):
    """Assign each token to the group its characters overlap most -> {name: [token indices]}.

    With `text`, only non-whitespace characters count, so a token like ".\\n\\n"
    belongs to the field that ends with "." (raw overlap breaks ties)."""
    out = {}
    for t, (a, b) in enumerate(offsets):
        best, best_ov = 'other', (0, 0)
        for name, s, e in groups:
            lo, hi = max(a, s), min(b, e)
            if hi <= lo:
                continue
            ov = (sum(not ch.isspace() for ch in text[lo:hi]) if text is not None else 0, hi - lo)
            if ov > best_ov:
                best, best_ov = name, ov
        out.setdefault(best, []).append(t)
    return out


def field_token_map(tok, text, item, prompt_offset, extra_groups=()):
    """Tokenize `text` once; return (input_ids, {group: [token idx]}).

    `extra_groups`: more (name, start, end) ranges in `text` coordinates, e.g.
    ('working', a, b) and ('target', b, c) for text appended after the prompt."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    groups = item_char_groups(item, prompt_offset) + list(extra_groups)
    covered = max(e for _, s, e in groups)
    if len(text) > covered:
        groups.append(('template', covered, len(text)))
    return enc['input_ids'], char_groups_to_tokens(enc['offset_mapping'], groups, text)


def check_item(tok, text, item, prompt_offset):
    """Problems with the mapping for one item (empty list = fine):
    every field has tokens, they are contiguous, and together they cover the
    field's text with only whitespace extra (no token merges two fields)."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc['offset_mapping']
    tokmap = char_groups_to_tokens(offs, item_char_groups(item, prompt_offset, len(text)), text)
    errs = []
    names = list(FIELDS_FIXED) + [f'cond_{k}' for k in range(1, len(item['conditions']) + 1)]
    spans = {f: item['spans'][f] for f in FIELDS_FIXED}
    spans.update({f'cond_{k}': se for k, se in enumerate(item['spans']['conditions'], 1)})
    for f in names:
        idx = tokmap.get(f, [])
        s, e = spans[f][0] + prompt_offset, spans[f][1] + prompt_offset
        if not idx:
            errs.append(f'{f}: no tokens'); continue
        if idx != list(range(idx[0], idx[-1] + 1)):
            errs.append(f'{f}: tokens not contiguous')
        a, b = offs[idx[0]][0], offs[idx[-1]][1]
        if a > s or b < e:
            errs.append(f'{f}: tokens cover {text[a:b]!r}, field is {text[s:e]!r}')
        elif text[a:s].strip() or text[e:b].strip():
            errs.append(f'{f}: a token merges the field with neighbouring text: {text[a:b]!r}')
    return errs


def main():
    sys.path.insert(0, HERE)
    from baseline_letter_logprobs import MODELS, TEMPLATE_KWARGS, build_text
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MODELS))
    ap.add_argument('--data', default=os.path.join(HERE, 'ivp_open', 'ivp_open_all.jsonl'))
    args = ap.parse_args()
    hf_name, fmt = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(hf_name)
    items = [json.loads(l) for l in open(args.data, encoding='utf-8')]
    bad, sizes = 0, Counter()
    for it in items:
        text, off = build_text(tok, it['prompt'], fmt, None, None, TEMPLATE_KWARGS.get(args.model))
        errs = check_item(tok, text, it, off)
        if errs:
            bad += 1
            if bad <= 10:
                print(it['id'], errs)
        _, tm = field_token_map(tok, text, it, off)
        for g, idx in tm.items():
            sizes[g.split('_')[0] if g.startswith('cond_') else g] += len(idx)
    n = len(items)
    print(f'{args.model}: field -> token mapping OK for {n - bad}/{n} prompts')
    print('mean tokens per prompt by group:', {g: round(v / n, 1) for g, v in sizes.items()})
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
