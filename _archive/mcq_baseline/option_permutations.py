#!/usr/bin/env python3
"""
Option-order consistency on IVP-MCQ: is the model's choice driven by option
content or by letter position?

Every item is scored under all 6 orders of its three options (the identity
order is the original prompt), with the same prompt format and letter-log-prob
scoring as baseline_letter_logprobs.py. The options are reordered in memory
via the dataset `spans`; the data files are not touched. Each option appears
under each letter exactly twice, so letter bias averages out.

This is a behavioural reliability check (which items are answered from the
content), not an attribution method; the target and inputs of any attribution
run are unaffected.

Per item:
  n_correct (0-6), the original option picked under each order,
  content_consistent  same option picked under all 6 orders
  robust_correct      correct under all 6 orders (chance 1/729 = 0.14%)
  letter_consistent   same letter picked under all 6 orders (pure position)

  python option_permutations.py --model qwen2.5-3b-instruct --split test
  python option_permutations.py --model phi-4-mini-instruct --split test --system letter-only
  python option_permutations.py --table results/option_permutations/*/summary.json

Output (never overwrites): results/option_permutations/<UTC stamp>_<model>[_sys]_<split>/
"""
import argparse, datetime, glob, itertools, json, math, os, platform, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ivp_mcq'))
from grade_mcq import load, letter_from_scores, LETTERS  # noqa: E402
from baseline_letter_logprobs import (MODELS, LETTER_FORMS, TEMPLATE_KWARGS, GPU_DTYPE,  # noqa: E402
                                      SYSTEM_PRESETS, PROMPT_VERSION, SEED, build_text, letter_ids,
                                      supports_system, load_model, score_texts, model_revision)

PERMS = list(itertools.permutations(range(3)))  # PERMS[0] = (0, 1, 2) = original order


def permuted_prompt(it, perm):
    """Prompt with original option perm[k] shown under letter k."""
    p, spans = it['prompt'], it['spans']['options']
    texts = [o['text'] for o in it['options']]
    for (s, e), k in sorted(zip(spans, range(3)), reverse=True):
        p = p[:s] + texts[perm[k]] + p[e:]
    return p


def run(args):
    import torch
    import transformers
    torch.manual_seed(SEED)
    hf_name, fmt = MODELS[args.model]
    system = SYSTEM_PRESETS.get(args.system, args.system)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[
        args.dtype or (GPU_DTYPE.get(args.model, 'fp16') if device == 'cuda' else 'fp32')]
    data_path = args.data or os.path.join(HERE, 'ivp_mcq', f'ivp_mcq_{args.split}.jsonl')
    items = load(data_path)
    if args.limit:
        items = items[:args.limit]

    tok, model = load_model(args.model, dtype, device, args.revision)
    ids = letter_ids(tok, LETTER_FORMS[fmt][0])
    jobs = [(i, q) for i in range(len(items)) for q in range(len(PERMS))]
    texts = [build_text(tok, permuted_prompt(items[i], PERMS[q]), fmt, None, system,
                        TEMPLATE_KWARGS.get(args.model))[0] for i, q in jobs]
    # the identity order must reproduce the original prompt exactly
    assert all(texts[k * len(PERMS)] == build_text(tok, it['prompt'], fmt, None, system,
                                                   TEMPLATE_KWARGS.get(args.model))[0]
               for k, it in enumerate(items))
    scores, _ = score_texts(model, tok, texts, [ids], args.batch_size, device)

    rows = []
    by_item = defaultdict(list)
    for (i, q), s in zip(jobs, scores):
        by_item[i].append((q, s['logprobs'][0]))
    for i, it in enumerate(items):
        gold_orig = LETTERS.index(it['answer'])
        per = []
        for q, lp in sorted(by_item[i]):
            perm = PERMS[q]
            pred = letter_from_scores(lp)
            picked = perm[LETTERS.index(pred)]                    # original option index
            z = sum(math.exp(v) for v in lp.values())
            per.append({'perm': ''.join(LETTERS[k] for k in perm),  # original letters shown as A,B,C
                        'gold_letter': LETTERS[perm.index(gold_orig)], 'pred': pred,
                        'picked_original': LETTERS[picked], 'correct': picked == gold_orig,
                        'logprobs': lp, 'letter_mass': z,
                        'p_pred': math.exp(lp[pred]) / z})
        picks = [r['picked_original'] for r in per]
        n_correct = sum(r['correct'] for r in per)
        trap = it['info'].get('trap_letter') if it['variant'] == 'B' else None
        rows.append({
            'id': it['id'], 'split': it['split'], 'variant': it['variant'], 'subtype': it['subtype'],
            'ode_type': it['ode_type'], 'category': it['category'], 'order': it['order'],
            'answer': it['answer'], 'trap_letter': trap,
            'original_order_pred': per[0]['pred'], 'original_order_correct': per[0]['correct'],
            'n_correct': n_correct,
            'picked_original_counts': {L: picks.count(L) for L in LETTERS},
            'majority_pick': Counter(picks).most_common(1)[0][0],
            'content_consistent': len(set(picks)) == 1,
            'robust_correct': n_correct == len(PERMS),
            'letter_consistent': len({r['pred'] for r in per}) == 1,
            'trap_picks': picks.count(trap) if trap else None,
            'per_order': per,
        })

    n = len(rows)
    slices = defaultdict(list)
    for r in rows:
        for key in ('all', f"variant={r['variant']}", f"subtype={r['variant']}:{r['subtype']}",
                    f"ode_type={r['ode_type']}", f"order={r['order']}"):
            slices[key].append(r)
    report = {k: {'n': len(v),
                  'acc_original_order': round(sum(r['original_order_correct'] for r in v) / len(v), 4),
                  'acc_mean_over_orders': round(sum(r['n_correct'] for r in v) / (len(PERMS) * len(v)), 4),
                  'robust_correct': round(sum(r['robust_correct'] for r in v) / len(v), 4),
                  'content_consistent': round(sum(r['content_consistent'] for r in v) / len(v), 4),
                  'letter_consistent': round(sum(r['letter_consistent'] for r in v) / len(v), 4)}
              for k, v in sorted(slices.items())}
    all_preds = Counter(p['pred'] for r in rows for p in r['per_order'])
    unrel = [r for r in rows if r['trap_letter']]
    summary = {
        'meta': {'experiment': 'option_permutations', 'model_key': args.model, 'model': hf_name,
                 'revision': args.revision or model_revision(model, hf_name),
                 'dtype': str(dtype).replace('torch.', ''), 'device': device,
                 'prompt_version': PROMPT_VERSION + ('+sys' if system else ''), 'prompt_format': fmt,
                 'system': system, 'system_in_user_turn': bool(system) and not supports_system(tok),
                 'template_kwargs': TEMPLATE_KWARGS.get(args.model),
                 'letter_tokens': {L: [LETTER_FORMS[fmt][0].format(L), ids[L]] for L in LETTERS},
                 'orders': [''.join(LETTERS[k] for k in p) for p in PERMS],
                 'split': args.split, 'data': os.path.relpath(data_path, HERE).replace(os.sep, '/'), 'n_items': n, 'limit': args.limit, 'batch_size': args.batch_size,
                 'seed': SEED, 'torch': torch.__version__, 'transformers': transformers.__version__,
                 'python': platform.python_version(),
                 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')},
        'report': report,
        'n_correct_histogram': {k: sum(r['n_correct'] == k for r in rows) for k in range(len(PERMS) + 1)},
        'n_correct_histogram_if_guessing': {k: round(n * math.comb(6, k) * (1 / 3) ** k * (2 / 3) ** (6 - k), 1)
                                            for k in range(len(PERMS) + 1)},
        'pred_letter_counts_all_orders': {L: all_preds[L] for L in LETTERS},
        'acc_by_order': {''.join(LETTERS[k] for k in p):
                         round(sum(r['per_order'][q]['correct'] for r in rows) / n, 4) for q, p in enumerate(PERMS)},
        'trap_pick_rate_all_orders': (round(sum(r['trap_picks'] for r in unrel) / (6 * len(unrel)), 4)
                                      if unrel else None),
        'robust_correct_ids': [r['id'] for r in rows if r['robust_correct']],
    }

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = os.path.join(args.outdir, f"{stamp}_{args.model}{'_sys' if system else ''}_{args.split}")
    os.makedirs(out, exist_ok=False)
    with open(os.path.join(out, 'items.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    a = report['all']
    print(f"{args.model} [{args.split}, n={n}] original-order acc={a['acc_original_order']:.4f} "
          f"mean-over-orders={a['acc_mean_over_orders']:.4f} robust-correct={a['robust_correct']:.4f} "
          f"content-consistent={a['content_consistent']:.4f} letter-consistent={a['letter_consistent']:.4f}  -> {out}")
    return out


ROWS = ['all', 'variant=B', 'variant=C', 'order=1', 'order=2',
        'subtype=B:duplicate', 'subtype=B:rescaled', 'subtype=B:unrelated',
        'subtype=C:extra1_direct', 'subtype=C:extra1_indirect',
        'subtype=C:extra2_direct', 'subtype=C:extra2_indirect']
COLS = [('acc_original_order', 'acc orig'), ('acc_mean_over_orders', 'acc mean6'),
        ('robust_correct', 'robust'), ('content_consistent', 'content'), ('letter_consistent', 'letter')]


def table(paths):
    sums = [json.load(open(p, encoding='utf-8')) for p in paths]
    rows = ROWS + sorted(k for k in sums[0]['report'] if k.startswith('ode_type='))
    w = max(map(len, rows)) + 2
    for s in sums:
        m = s['meta']
        print(f"\n{m['model_key']}{'+sys' if m.get('system') else ''} [{m['split']}]   (% of items; "
              f"robust = correct under all 6 orders, chance 0.14%)")
        print('slice'.ljust(w) + ''.join(h.rjust(11) for _, h in COLS) + 'n'.rjust(7))
        for r in rows:
            if r in s['report']:
                v = s['report'][r]
                print(r.ljust(w) + ''.join(f'{100 * v[k]:11.1f}' for k, _ in COLS) + f"{v['n']:7d}")
        h, g = s['n_correct_histogram'], s['n_correct_histogram_if_guessing']
        print('items correct under k of 6 orders:  ' + '  '.join(f'k={k}: {h[str(k)] if str(k) in h else h[k]}'
                                                               for k in range(7)))
        print('expected if guessing at random:      ' + '  '.join(f'k={k}: {g[str(k)] if str(k) in g else g[k]}'
                                                               for k in range(7)))
        print('letters predicted over all orders:', s['pred_letter_counts_all_orders'],
              '| trap-pick rate over all orders:', s['trap_pick_rate_all_orders'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=sorted(MODELS))
    ap.add_argument('--split', default='test', choices=['dev', 'test', 'all'])
    ap.add_argument('--data', help='other JSONL in the IVP-MCQ format (default: ivp_mcq/ivp_mcq_<split>.jsonl)')
    ap.add_argument('--system', help='system message or preset: ' + ', '.join(SYSTEM_PRESETS))
    ap.add_argument('--revision')
    ap.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'])
    ap.add_argument('--device')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'results', 'option_permutations'))
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
