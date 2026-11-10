#!/usr/bin/env python3
"""
Open-answer arithmetic: can the model calculate, without options?

Greedy generation on easy_open/easy_open_test.jsonl (the same 144 expressions as
the easy MCQ control). The number is read from the text and compared with the
exact value.

  --mode number   "Answer with the number only." The first integer in the reply
                  counts, or the first one after the last "=" if the reply
                  restates the sum (one step, comparable to the letter score).
  --mode working  "Show your working briefly, then 'Answer: <number>'." The
                  number after the last "Answer:" counts, else the last integer.

Same model loading and chat template as baseline_letter_logprobs.py; no system
message unless --system is given.

  python open_arithmetic.py --model qwen2.5-3b-instruct --mode number
  python open_arithmetic.py --table results/easy_open/*/summary.json

Output (never overwrites): results/easy_open/<UTC stamp>_<model>_<mode>/
"""
import argparse, datetime, glob, json, os, platform, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ivp_mcq'))
from baseline_letter_logprobs import (MODELS, TEMPLATE_KWARGS, GPU_DTYPE, SYSTEM_PRESETS, SEED,  # noqa: E402
                                      build_text, load_model, model_revision)

INT = re.compile(r'-?\d[\d,]*')
CATS = ['add', 'sub', 'mul', 'div', 'pow', 'mixed']


def to_int(s):
    return int(s.replace(',', ''))


def read_number(text, mode):
    if mode == 'working':
        m = re.findall(r'Answer\s*[:=]\s*\**\s*\$?\\?(?:boxed\{)?\s*(-?\d[\d,]*)', text, re.I)
        if m:
            return to_int(m[-1]), 'answer_line'
        nums = INT.findall(text)
        return (to_int(nums[-1]), 'last_number') if nums else (None, 'none')
    if '=' in text:  # "17 + 40 = 57": the result follows the last "="
        after = INT.findall(text.rsplit('=', 1)[1])
        if after:
            return to_int(after[0]), 'after_equals'
    nums = INT.findall(text)
    return (to_int(nums[0]), 'first_number') if nums else (None, 'none')


def run(args):
    import torch
    import transformers
    torch.manual_seed(SEED)
    hf_name, fmt = MODELS[args.model]
    system = SYSTEM_PRESETS.get(args.system, args.system)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[
        args.dtype or (GPU_DTYPE.get(args.model, 'fp16') if device == 'cuda' else 'fp32')]
    items = [json.loads(l) for l in open(args.data, encoding='utf-8')]
    if args.limit:
        items = items[:args.limit]
    max_new = args.max_new_tokens or (16 if args.mode == 'number' else 256)

    tok, model = load_model(args.model, dtype, device, args.revision)
    key = 'prompt' if args.mode == 'number' else 'prompt_working'
    rows = []
    for k, it in enumerate(items):
        text, _ = build_text(tok, it[key], fmt, None, system, TEMPLATE_KWARGS.get(args.model))
        inp = tok(text, return_tensors='pt', add_special_tokens=False).to(device)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out = tok.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
        num, how = read_number(out, args.mode)
        rows.append({'id': it['id'], 'category': it['category'], 'expression': it['expression'],
                     'value': it['value'], 'read': num, 'read_by': how, 'correct': num == it['value'],
                     'number_only_reply': bool(re.fullmatch(r'\s*-?\d[\d,]*\.?\s*', out)),
                     'n_new_tokens': int(g.shape[1] - inp['input_ids'].shape[1]), 'reply': out})
        print(f'\r{k + 1}/{len(items)}', end='', flush=True)
    print()

    n = len(rows)
    per = defaultdict(list)
    for r in rows:
        per[r['category']].append(r['correct'])
    summary = {
        'meta': {'experiment': 'open_arithmetic', 'model_key': args.model, 'model': hf_name,
                 'revision': args.revision or model_revision(model, hf_name), 'mode': args.mode,
                 'system': system, 'template_kwargs': TEMPLATE_KWARGS.get(args.model),
                 'dtype': str(dtype).replace('torch.', ''), 'device': device, 'greedy': True,
                 'max_new_tokens': max_new, 'example_input': build_text(tok, items[0][key], fmt, None, system,
                                                                     TEMPLATE_KWARGS.get(args.model))[0],
                 'data': os.path.relpath(args.data, HERE).replace(os.sep, '/'), 'n_items': n,
                 'limit': args.limit, 'seed': SEED, 'torch': torch.__version__,
                 'transformers': transformers.__version__, 'python': platform.python_version(),
                 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')},
        'accuracy': sum(r['correct'] for r in rows) / n,
        'per_category': {c: {'acc': sum(v) / len(v), 'n': len(v)} for c, v in sorted(per.items())},
        'no_number_found': sum(r['read'] is None for r in rows),
        'number_only_replies': sum(r['number_only_reply'] for r in rows),
        'read_by': {h: sum(r['read_by'] == h for r in rows) for h in {r['read_by'] for r in rows}},
        'mean_new_tokens': sum(r['n_new_tokens'] for r in rows) / n,
        'wrong_examples': [{k: r[k] for k in ('expression', 'value', 'read', 'reply')}
                           for r in rows if not r['correct']][:15],
    }
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = os.path.join(args.outdir, f"{stamp}_{args.model}{'_sys' if system else ''}_{args.mode}")
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, 'generations.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"{args.model} [{args.mode}, n={n}] acc={summary['accuracy']:.4f}  -> {out_dir}")
    return out_dir


def table(paths):
    sums = [json.load(open(p, encoding='utf-8')) for p in paths]
    heads = [f"{s['meta']['model_key'].replace('-instruct', '')} [{s['meta']['mode']}]" for s in sums]
    print('category'.ljust(12) + ''.join(h[:30].rjust(32) for h in heads))
    for c in CATS + ['all']:
        cells = []
        for s in sums:
            v = s['accuracy'] if c == 'all' else s['per_category'].get(c, {}).get('acc')
            n = s['meta']['n_items'] if c == 'all' else s['per_category'].get(c, {}).get('n', 0)
            cells.append((f'{100 * v:5.1f}  (n={n})' if v is not None else '-').rjust(32))
        print(c.ljust(12) + ''.join(cells))
    print('no number'.ljust(12) + ''.join(str(s['no_number_found']).rjust(32) for s in sums))
    print('mean tokens'.ljust(12) + ''.join(f"{s['mean_new_tokens']:.1f}".rjust(32) for s in sums))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=sorted(MODELS))
    ap.add_argument('--mode', choices=['number', 'working'], default='number')
    ap.add_argument('--data', default=os.path.join(HERE, 'easy_open', 'easy_open_test.jsonl'))
    ap.add_argument('--system', help='system message or preset (default: none)')
    ap.add_argument('--max-new-tokens', type=int)
    ap.add_argument('--revision')
    ap.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'])
    ap.add_argument('--device')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'results', 'easy_open'))
    ap.add_argument('--table', nargs='+', metavar='SUMMARY_JSON')
    args = ap.parse_args()
    if args.table:
        table(sorted(p for pat in args.table for p in glob.glob(pat)))
    elif args.model:
        run(args)
    else:
        ap.error('give --model or --table')


if __name__ == '__main__':
    main()
