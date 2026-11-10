#!/usr/bin/env python3
"""
Open-ended IVP accuracy of one model, with working allowed (IVP-OPEN).

Greedy, batched generation on ivp_open/ivp_open_<split>.jsonl; every reply is
graded by ivp_open/grade_open.py (final "Answer: y = ..." line, SymPy check
against the true solution). Besides correctness, each reply records whether
the answer solves the ODE, which conditions it satisfies, and for `unrelated`
items whether it follows the z-condition read as y.

Prompt: the item prompt as the user turn of the model's chat template (no
system message unless --system). Base models (format 'raw') get
prompt + "\\n\\nSolution:\\n". --thinking turns on Qwen3's thinking mode
(the <think> block is removed before grading).

  python eval_open_ivp.py --model qwen2.5-3b-instruct --split dev
  python eval_open_ivp.py --model qwen3-1.7b --split dev --thinking --max-new-tokens 2048
  python eval_open_ivp.py --table results/open_eval/*/summary.json

Output (never overwrites): results/open_eval/<UTC stamp>_<model>[_think]_<split>/
  generations.jsonl   one line per item, written batch by batch
  summary.json        metadata, accuracy, per-slice report
"""
import argparse, datetime, glob, json, os, platform, re, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'ivp_open'), os.path.join(HERE, 'ivp_mcq')]
from grade_open import load, grade_reply, evaluate  # noqa: E402
from baseline_letter_logprobs import (MODELS, TEMPLATE_KWARGS, GPU_DTYPE, SYSTEM_PRESETS, SEED,  # noqa: E402
                                      build_text, load_model, model_revision)

RAW_SOLUTION = '\n\nSolution:\n'
THINK_RE = re.compile(r'<think>.*?</think>', re.S)


def model_input(tok, item, fmt, system, tkw):
    if fmt == 'raw':
        return item['prompt'] + RAW_SOLUTION
    return build_text(tok, item['prompt'], fmt, None, system, tkw)[0]


def run(args):
    import torch
    import transformers
    torch.manual_seed(SEED)
    hf_name, fmt = MODELS[args.model]
    system = SYSTEM_PRESETS.get(args.system, args.system)
    tkw = dict(TEMPLATE_KWARGS.get(args.model) or {})
    if args.thinking:
        tkw['enable_thinking'] = True
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[
        args.dtype or (GPU_DTYPE.get(args.model, 'fp16') if device == 'cuda' else 'fp32')]
    data = args.data or os.path.join(HERE, 'ivp_open', f'ivp_open_{args.split}.jsonl')
    items = load(data)
    if args.subtypes:
        items = [it for it in items if it['subtype'] in args.subtypes.split(',')]
    if args.limit:
        items = items[:args.limit]

    tok, model = load_model(args.model, dtype, device, args.revision)
    texts = [model_input(tok, it, fmt, system, tkw) for it in items]
    n_tok = [len(tok(t, add_special_tokens=False)['input_ids']) for t in texts]
    order = sorted(range(len(items)), key=lambda i: n_tok[i])

    rows = [None] * len(items)
    if args.resume:  # continue an unfinished run in its own folder; keep the replies already written
        out_dir = args.resume
        gen_path = os.path.join(out_dir, 'generations.jsonl')
        prev = {}
        for line in open(gen_path, encoding='utf-8'):
            try:
                r = json.loads(line)
                prev[r['id']] = r
            except json.JSONDecodeError:  # a line cut off when the run died
                pass
        rows = [prev.get(it['id']) for it in items]
        order = [i for i in order if rows[i] is None]
        with open(gen_path, 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows if r is not None)
        print(f'resuming {out_dir}: {len(items) - len(order)} done, {len(order)} to go')
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out_dir = os.path.join(args.outdir, f"{stamp}_{args.model}{'_think' if args.thinking else ''}"
                                            f"{'_sys' if system else ''}_{args.split}")
        os.makedirs(out_dir, exist_ok=False)
        gen_path = os.path.join(out_dir, 'generations.jsonl')
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = ({tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])) - {None}
    t0 = time.time()
    with open(gen_path, 'a', encoding='utf-8') as fout, torch.no_grad():
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True,
                      add_special_tokens=False).to(device)
            g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            new = g[:, enc['input_ids'].shape[1]:]
            for j, i in enumerate(idx):
                ids = new[j].tolist()
                ended = any(t in eos for t in ids)
                n_new = next((k for k, t in enumerate(ids) if t in eos), len(ids))
                raw = tok.decode(ids, skip_special_tokens=True)
                reply = THINK_RE.sub('', raw).strip()
                it = items[i]
                gr = grade_reply(it, reply)
                rows[i] = {'id': it['id'], 'split': it['split'], 'subtype': it['subtype'],
                           'ode_type': it['ode_type'], 'category': it['category'], 'order': it['order'],
                           'answer_text': it['answer_text'], **gr,
                           'n_new_tokens': n_new, 'truncated': not ended, 'reply': reply,
                           'raw_reply': raw if raw != reply else None,
                           'gen_ids': ids[:n_new]}  # exact generated tokens (eos excluded), for teacher forcing
                fout.write(json.dumps(rows[i]) + '\n')
            fout.flush()
            done = min(b + args.batch_size, len(order))
            print(f'\r{done}/{len(order)}  ({time.time() - t0:.0f}s)', end='', flush=True)
    print()

    meta = {'experiment': 'open_eval', 'model_key': args.model, 'model': hf_name,
                 'revision': args.revision or model_revision(model, hf_name),
                 'prompt_version': items[0]['prompt_version'], 'prompt_format': fmt,
                 'raw_suffix': RAW_SOLUTION if fmt == 'raw' else None, 'system': system,
                 'template_kwargs': tkw or None, 'thinking': args.thinking,
                 'example_input': texts[0], 'dtype': str(dtype).replace('torch.', ''),
                 'device': device, 'greedy': True, 'max_new_tokens': args.max_new_tokens,
                 'batch_size': args.batch_size, 'data': os.path.relpath(data, HERE).replace(os.sep, '/'),
                 'split': args.split, 'subtypes': args.subtypes, 'n_items': len(items), 'limit': args.limit, 'resumed': bool(args.resume),
                 'seed': SEED, 'torch': torch.__version__, 'transformers': transformers.__version__,
                 'python': platform.python_version(),
                 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                 'seconds': round(time.time() - t0),
                 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
    return write_summary(out_dir, meta, items, rows)


def write_summary(out_dir, meta, items, rows):
    acc, report = evaluate(items, rows)
    n = len(rows)
    wrong = [r for r in rows if not r['correct']]
    summary = {
        'meta': meta,
        'accuracy': acc,
        'report': report,
        'truncated': sum(r['truncated'] for r in rows),
        'mean_new_tokens': sum(r['n_new_tokens'] for r in rows) / n,
        'answer_found_by': dict(Counter(r['found_by'] for r in rows)),
        'wrong_but_solves_ode': sum(bool(r['solves_ode']) for r in wrong),
        'wrong_but_satisfies_some_condition': sum(bool(r['satisfies']) for r in wrong),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"{meta['model_key']}{' (thinking)' if meta['thinking'] else ''} [{meta['split']}, n={n}] acc={acc:.4f} "
          f"truncated={summary['truncated']} no_answer={report['answer_status'].get('no_answer', 0)}  -> {out_dir}")
    return out_dir


def regrade(run_dir):
    """Re-grade a run's saved replies with the current grader; writes <run_dir>_regraded<stamp>/."""
    old = json.load(open(os.path.join(run_dir, 'summary.json'), encoding='utf-8'))
    items = {it['id']: it for it in load(os.path.join(HERE, old['meta']['data']))}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = run_dir.rstrip('/\\') + f'_regraded{stamp}'
    os.makedirs(out_dir, exist_ok=False)
    rows = []
    with open(os.path.join(out_dir, 'generations.jsonl'), 'w', encoding='utf-8') as fout:
        for line in open(os.path.join(run_dir, 'generations.jsonl'), encoding='utf-8'):
            r = json.loads(line)
            r.update(grade_reply(items[r['id']], r['reply']))
            rows.append(r)
            fout.write(json.dumps(r) + '\n')
    meta = dict(old['meta'], regraded_from=os.path.basename(run_dir.rstrip('/\\')),
                regraded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                accuracy_before_regrade=old['accuracy'])
    return write_summary(out_dir, meta, [items[r['id']] for r in rows], rows)


ROWS = ['all', 'subtype=none', 'subtype=duplicate', 'subtype=rescaled', 'subtype=unrelated',
        'order=1', 'order=2', 'ode_type=first_order_direct', 'ode_type=first_order_separable',
        'ode_type=first_order_linear', 'ode_type=second_order_direct',
        'ode_type=second_order_homogeneous', 'ode_type=second_order_nonhomogeneous']


def table(paths):
    sums = [json.load(open(p, encoding='utf-8')) for p in paths]
    sums.sort(key=lambda s: -s['accuracy'])
    heads = [s['meta']['model_key'].replace('-instruct', '').replace('-2507', '')
             + ('+think' if s['meta'].get('thinking') else '') for s in sums]
    w = 38
    print('slice'.ljust(w) + ''.join(h[:17].rjust(19) for h in heads))
    for r in ROWS:
        print(r.ljust(w) + ''.join((f"{100 * s['report'][r]['acc']:5.1f} ({s['report'][r]['n']})"
                                    if r in s['report'] else '-').rjust(19) for s in sums))
    extra = [('unrelated: follows z read as y %', lambda s: 100 * s['report'].get(
                 'unrelated: answer satisfies z-condition read as y', float('nan'))),
             ('unrelated: equals trap %', lambda s: 100 * s['report'].get(
                 'unrelated: answer equals the trap solution', float('nan'))),
             ('no answer found', lambda s: s['report']['answer_status'].get('no_answer', 0)),
             ('unparsable answer', lambda s: sum(v for k, v in s['report']['answer_status'].items()
                                                 if not k.startswith('ok') and k != 'no_answer')),
             ('answers in t instead of x (accepted)', lambda s: s['report']['answer_status'].get('ok_variable_t', 0)),
             ('truncated at max tokens', lambda s: s['truncated']),
             ('mean new tokens', lambda s: s['mean_new_tokens']),
             ('minutes', lambda s: s['meta']['seconds'] / 60)]
    for lab, f in extra:
        print(lab.ljust(w) + ''.join((f'{f(s):.1f}' if isinstance(f(s), float) else str(f(s))).rjust(19)
                                     for s in sums))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=sorted(MODELS))
    ap.add_argument('--split', default='dev', choices=['dev', 'test', 'all'])
    ap.add_argument('--data', help='default: ivp_open/ivp_open_<split>.jsonl')
    ap.add_argument('--subtypes', help='comma-separated subset of none,duplicate,rescaled,unrelated')
    ap.add_argument('--system', help='system message or preset (default: none)')
    ap.add_argument('--thinking', action='store_true', help="Qwen3: turn thinking mode on")
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--revision')
    ap.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'])
    ap.add_argument('--device')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--resume', metavar='RUN_DIR', help='continue an unfinished run (same model and split)')
    ap.add_argument('--outdir', default=os.path.join(HERE, 'results', 'open_eval'))
    ap.add_argument('--table', nargs='+', metavar='SUMMARY_JSON')
    ap.add_argument('--regrade', nargs='+', metavar='RUN_DIR', help='re-grade saved replies with the current grader')
    args = ap.parse_args()
    if args.regrade:
        for d in sorted(p for pat in args.regrade for p in glob.glob(pat)):
            regrade(d)
    elif args.table:
        table(sorted(p for pat in args.table for p in glob.glob(pat)))
    elif args.model:
        run(args)
    else:
        ap.error('give --model or --table')


if __name__ == '__main__':
    main()
