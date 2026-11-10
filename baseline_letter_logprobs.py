#!/usr/bin/env python3
"""
Zero-shot accuracy baseline on IVP-MCQ, scored by next-token letter log-probs.

For every item the prompt is fed to the model once; the log-probs of the
answer-letter tokens A/B/C at the next position are compared
(`letter_from_scores`). No generation, no parsing.

  python baseline_letter_logprobs.py --model qwen2.5-0.5b-instruct
  python baseline_letter_logprobs.py --model qwen2.5-1.5b-instruct --split test
  python baseline_letter_logprobs.py --table results/baseline/*/summary.json

Prompt formats (PROMPT_VERSION below; bump it if either changes):
  chat : tokenizer chat template, item prompt as the single user turn,
         generation prompt added. Letter tokens "A","B","C".
  raw  : item prompt + RAW_SUFFIX, for base models. Letter tokens " A"," B"," C".
  --answer-prefix STR (optional, off by default): STR is appended after the
         chat generation prompt / after the raw input, e.g. "The answer is";
         letter tokens are then " A"," B"," C". Recorded as prompt_version
         "v1+prefix" together with the prefix string.
  --system STR (optional, chat only): system message placed before the user
         turn (replaces the template's default one). `--system letter-only`
         uses SYSTEM_PRESETS['letter-only']. Recorded as "+sys". The item
         prompt itself is unchanged, so the dataset `spans` still apply
         (shifted by prompt_char_offset).

Output (never overwrites): results/baseline/<UTC timestamp>_<model key>_<split>/
  predictions.jsonl  one line per item
  summary.json       metadata + per-slice report
"""
import argparse, datetime, glob, json, math, os, platform, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ivp_mcq'))
from grade_mcq import load, letter_from_scores, evaluate, LETTERS  # noqa: E402

PROMPT_VERSION = 'v1'
RAW_SUFFIX = '\nAnswer:'
SEED = 0

SYSTEM_PRESETS = {
    'letter-only': ('You answer multiple-choice questions. Reply with exactly one '
                    'letter (A, B or C) and nothing else. Do not explain or show '
                    'any working.'),
}

MODELS = {
    'qwen2.5-0.5b-instruct': ('Qwen/Qwen2.5-0.5B-Instruct', 'chat'),
    'qwen2.5-1.5b-instruct': ('Qwen/Qwen2.5-1.5B-Instruct', 'chat'),
    'qwen2.5-3b-instruct': ('Qwen/Qwen2.5-3B-Instruct', 'chat'),
    'qwen2.5-math-1.5b': ('Qwen/Qwen2.5-Math-1.5B', 'raw'),
    'qwen2.5-math-1.5b-instruct': ('Qwen/Qwen2.5-Math-1.5B-Instruct', 'chat'),
    'qwen3-1.7b': ('Qwen/Qwen3-1.7B', 'chat'),
    'qwen3-4b': ('Qwen/Qwen3-4B', 'chat'),
    'qwen3-4b-instruct-2507': ('Qwen/Qwen3-4B-Instruct-2507', 'chat'),
    'phi-4-mini-instruct': ('microsoft/Phi-4-mini-instruct', 'chat'),
    # gated: accept the licence on the HF model page and set HF_TOKEN
    'llama-3.2-1b-instruct': ('meta-llama/Llama-3.2-1B-Instruct', 'chat'),
    'llama-3.2-3b-instruct': ('meta-llama/Llama-3.2-3B-Instruct', 'chat'),
    'gemma-2-2b-it': ('google/gemma-2-2b-it', 'chat'),
    'gemma-3-1b-it': ('google/gemma-3-1b-it', 'chat'),
    'gemma-3-4b-it': ('google/gemma-3-4b-it', 'chat'),
}
# extra chat-template arguments. Qwen3: answer directly, no <think> block.
# Llama 3.2: its template writes today's date into the system header; pin it
# so the input is the same on every day.
TEMPLATE_KWARGS = {'qwen3-1.7b': {'enable_thinking': False},
                   'qwen3-4b': {'enable_thinking': False},
                   'llama-3.2-1b-instruct': {'date_string': '26 Jul 2024'},
                   'llama-3.2-3b-instruct': {'date_string': '26 Jul 2024'}}
# from_pretrained arguments (Gemma: eager attention, as Google recommends)
LOAD_KWARGS = {'gemma-2-2b-it': {'attn_implementation': 'eager'},
               'gemma-3-1b-it': {'attn_implementation': 'eager'},
               'gemma-3-4b-it': {'attn_implementation': 'eager'}}
# GPU dtype when --dtype is not given (default fp16). Gemma overflows in fp16
# and the T4 has no fast bf16, so the small Gemmas run in fp32.
GPU_DTYPE = {'gemma-2-2b-it': 'fp32', 'gemma-3-1b-it': 'fp32'}
# letter token spelling expected right after the prompt, and the alternative
# spelling (recorded only as a robustness check)
LETTER_FORMS = {'chat': ('{}', ' {}'), 'raw': (' {}', '{}')}


def build_text(tok, prompt, fmt, answer_prefix=None, system=None, template_kwargs=None):
    """Model input string and the character offset of `prompt` inside it
    (needed later to shift the dataset `spans`)."""
    if fmt == 'chat':
        kw = dict(tokenize=False, add_generation_prompt=True, **(template_kwargs or {}))
        if system and supports_system(tok):
            msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
        elif system:  # e.g. Gemma 2: system text goes at the top of the user turn
            msgs = [{'role': 'user', 'content': system + '\n\n' + prompt}]
        else:
            msgs = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(msgs, **kw)
    elif fmt == 'raw':
        assert not system, 'system message needs the chat format'
        text = prompt + RAW_SUFFIX
    else:
        raise ValueError(fmt)
    if answer_prefix:
        text += (' ' if fmt == 'raw' else '') + answer_prefix
    off = text.find(prompt)
    assert off >= 0 and text.count(prompt) == 1
    return text, off


def supports_system(tok):
    try:
        tok.apply_chat_template([{'role': 'system', 'content': 's'},
                                 {'role': 'user', 'content': 'u'}], tokenize=False)
        return True
    except Exception:  # template raises on a system turn
        return False


def letter_ids(tok, form):
    ids = {}
    for L in LETTERS:
        enc = tok.encode(form.format(L), add_special_tokens=False)
        assert len(enc) == 1, f'{form.format(L)!r} is not a single token: {enc}'
        ids[L] = enc[0]
    return ids


def model_revision(model, hf_name):
    """Commit hash of the loaded weights. transformers 5 dropped
    config._commit_hash, so fall back to the snapshot folder in the HF cache."""
    h = getattr(model.config, '_commit_hash', None)
    if h:
        return h
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(hf_name, 'config.json')
        m = re.search(r'snapshots[\\/]([0-9a-f]{40})', p) if isinstance(p, str) else None
        return m.group(1) if m else None
    except Exception:
        return None


def dtype_kwarg(dtype):
    """`dtype=` from transformers 4.56 on (`torch_dtype` is deprecated there)."""
    import transformers
    major, minor = (int(v) for v in transformers.__version__.split('.')[:2])
    return {'dtype': dtype} if (major, minor) >= (4, 56) else {'torch_dtype': dtype}


def load_model(model_key, dtype, device, revision=None):
    """Tokenizer (left padding) and model in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_name = MODELS[model_key][0]
    tok = AutoTokenizer.from_pretrained(hf_name, revision=revision)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_name, revision=revision, **dtype_kwarg(dtype),
                                                 **LOAD_KWARGS.get(model_key, {})).to(device).eval()
    return tok, model


def score_texts(model, tok, texts, id_sets, batch_size, device):
    """Next-token log-probs after each text, for each {letter: token id} dict in
    `id_sets`. Batched by length with left padding and explicit position ids.
    Returns (scores aligned with texts, token counts)."""
    import torch
    n_tok = [len(tok(t, add_special_tokens=False)['input_ids']) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: n_tok[i])  # less padding
    out = [None] * len(texts)
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True,
                      add_special_tokens=False).to(device)
            # explicit positions so left padding does not shift RoPE positions
            pos = (enc['attention_mask'].cumsum(-1) - 1).clamp(min=0)
            logits = model(**enc, position_ids=pos).logits[:, -1, :].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError(f'non-finite logits ({model.dtype}); rerun with --dtype fp32')
            logp = torch.log_softmax(logits, dim=-1)
            top_p, top_id = logp.max(dim=-1)
            for j, i in enumerate(idx):
                out[i] = {'logprobs': [{L: logp[j, ids[L]].item() for L in LETTERS} for ids in id_sets],
                          'top1_id': top_id[j].item(), 'top1_prob': math.exp(top_p[j].item())}
            print(f'\r{min(b + batch_size, len(order))}/{len(order)}', end='', flush=True)
    print()
    return out, n_tok


def run(args):
    import torch
    import transformers

    torch.manual_seed(SEED)
    hf_name, fmt = MODELS[args.model]
    fmt = args.format or fmt
    system = SYSTEM_PRESETS.get(args.system, args.system)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[
        args.dtype or (GPU_DTYPE.get(args.model, 'fp16') if device == 'cuda' else 'fp32')]

    data_path = args.data or os.path.join(HERE, 'ivp_mcq', f'ivp_mcq_{args.split}.jsonl')
    items = load(data_path)
    if args.limit:
        items = items[:args.limit]

    tok, model = load_model(args.model, dtype, device, args.revision)

    main_form, alt_form = LETTER_FORMS[fmt]
    if args.answer_prefix:
        main_form, alt_form = ' {}', '{}'
    ids_main, ids_alt = letter_ids(tok, main_form), letter_ids(tok, alt_form)

    texts, offsets = zip(*(build_text(tok, it['prompt'], fmt, args.answer_prefix, system,
                                         TEMPLATE_KWARGS.get(args.model)) for it in items))
    scores, n_tok = score_texts(model, tok, texts, [ids_main, ids_alt], args.batch_size, device)

    rows = []
    for it, s, nt, off in zip(items, scores, n_tok, offsets):
        lp, lp_alt = s['logprobs']
        z = math.log(sum(math.exp(v) for v in lp.values()))
        pred = letter_from_scores(lp)
        trap = it['info'].get('trap_letter') if it['variant'] == 'B' else None
        rows.append({
            'id': it['id'], 'split': it['split'], 'variant': it['variant'],
            'subtype': it['subtype'], 'ode_type': it['ode_type'],
            'category': it['category'], 'order': it['order'],
            'n_conditions': it['n_conditions'],
            'answer': it['answer'], 'pred': pred,
            'correct': pred == it['answer'],
            'trap_letter': trap, 'picked_trap': (pred == trap) if trap else None,
            'logprobs': lp,
            'probs_over_letters': {L: math.exp(v - z) for L, v in lp.items()},
            'letter_mass': math.exp(z),
            'logprobs_alt_form': lp_alt,
            'pred_alt_form': letter_from_scores(lp_alt),
            'top1_token': tok.decode([s['top1_id']]),
            'top1_prob': s['top1_prob'],
            'n_tokens': nt, 'prompt_char_offset': off,
        })

    preds = [r['pred'] for r in rows]
    acc, report = evaluate(items, preds)
    n = len(rows)
    summary = {
        'meta': {
            'experiment': 'baseline_letter_logprobs',
            'model_key': args.model, 'model': hf_name,
            'revision': args.revision or model_revision(model, hf_name),
            'dtype': str(dtype).replace('torch.', ''), 'device': device,
            'prompt_version': PROMPT_VERSION + ('+sys' if system else '')
                              + ('+prefix' if args.answer_prefix else ''),
            'prompt_format': fmt, 'answer_prefix': args.answer_prefix,
            'system': system, 'template_kwargs': TEMPLATE_KWARGS.get(args.model),
            'system_in_user_turn': bool(system) and fmt == 'chat' and not supports_system(tok),
            'load_kwargs': LOAD_KWARGS.get(args.model),
            'raw_suffix': RAW_SUFFIX if fmt == 'raw' else None,
            'letter_tokens': {L: [main_form.format(L), ids_main[L]] for L in LETTERS},
            'letter_tokens_alt': {L: [alt_form.format(L), ids_alt[L]] for L in LETTERS},
            'example_input': texts[0],
            'split': args.split, 'data': os.path.relpath(data_path, HERE).replace(os.sep, '/'), 'n_items': n, 'limit': args.limit,
            'batch_size': args.batch_size, 'seed': SEED,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'python': platform.python_version(),
            'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        },
        'accuracy': acc,
        'report': report,
        'pred_letter_counts': dict(Counter(preds)),
        'gold_letter_counts': dict(Counter(it['answer'] for it in items)),
        'confusion_gold_to_pred': {g: dict(Counter(r['pred'] for r in rows if r['answer'] == g))
                                   for g in LETTERS},
        'n_correct': sum(r['correct'] for r in rows), 'n_wrong': sum(not r['correct'] for r in rows),
        'mean_letter_mass': sum(r['letter_mass'] for r in rows) / n,
        'frac_top1_is_letter': sum(r['top1_token'] == main_form.format(r['pred']) for r in rows) / n,
        'accuracy_alt_form': sum(r['pred_alt_form'] == r['answer'] for r in rows) / n,
        'agreement_main_vs_alt_form': sum(r['pred_alt_form'] == r['pred'] for r in rows) / n,
    }

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    tag = ('_sys' if system else '') + ('_prefix' if args.answer_prefix else '')
    out = os.path.join(args.outdir, f'{stamp}_{args.model}{tag}_{args.split}')
    os.makedirs(out, exist_ok=False)
    with open(os.path.join(out, 'predictions.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'{args.model} [{args.split}, n={n}] acc={acc:.4f}  -> {out}')
    return out


TABLE_ROWS = ['all', 'variant=B', 'variant=C', 'order=1', 'order=2',
              'subtype=B:duplicate', 'subtype=B:rescaled', 'subtype=B:unrelated',
              'subtype=C:extra1_direct', 'subtype=C:extra1_indirect',
              'subtype=C:extra2_direct', 'subtype=C:extra2_indirect']


def table(paths):
    sums = [json.load(open(p, encoding='utf-8')) for p in paths]
    rows = TABLE_ROWS + sorted(k for k in sums[0]['report']
                               if k.startswith(('ode_type=', 'category=')))
    heads = [s['meta']['model_key'].replace('-instruct', '-it')
             + ('+sys' if s['meta'].get('system') else '')
             + ('+pfx' if s['meta'].get('answer_prefix') else '') for s in sums]
    w = max(len(r) for r in rows) + 2
    print(''.join(['slice'.ljust(w)] + [h[:24].rjust(26) for h in heads]))
    for r in rows:
        print(''.join([r.ljust(w)] + [
            (f"{s['report'][r]['acc'] * 100:5.1f}  (n={s['report'][r]['n']})" if r in s['report']
             else '-').rjust(26) for s in sums]))
    for label, f in [('B:unrelated trap-pick %', lambda s: s['report'].get(
                         'B:unrelated trap-pick rate', {}).get('rate', float('nan')) * 100),
                     ('pred A/B/C', lambda s: '/'.join(str(s['pred_letter_counts'].get(L, 0))
                                                        for L in LETTERS)),
                     ('mean letter mass', lambda s: s['mean_letter_mass']),
                     ('top-1 is a letter %', lambda s: s['frac_top1_is_letter'] * 100),
                     ('acc, alt letter form %', lambda s: s['accuracy_alt_form'] * 100)]:
        vals = [f(s) for s in sums]
        print(''.join([label.ljust(w)] + [
            (f'{v:.3f}' if isinstance(v, float) and v < 1.5 else
             f'{v:.1f}' if isinstance(v, float) else str(v)).rjust(26) for v in vals]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=sorted(MODELS))
    ap.add_argument('--split', default='test', choices=['dev', 'test', 'all'])
    ap.add_argument('--data', help='other JSONL in the IVP-MCQ format (default: ivp_mcq/ivp_mcq_<split>.jsonl)')
    ap.add_argument('--format', choices=['chat', 'raw'], help='override the model default')
    ap.add_argument('--answer-prefix', help='text the answer starts with, e.g. "The answer is"')
    ap.add_argument('--system', help='system message, or a preset name: '
                    + ', '.join(SYSTEM_PRESETS))
    ap.add_argument('--revision', help='HF revision (default: latest, commit hash recorded)')
    ap.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'],
                    help='default fp16 on GPU, fp32 on CPU')
    ap.add_argument('--device')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--limit', type=int, help='first N items only (smoke tests)')
    ap.add_argument('--outdir', default=os.path.join(HERE, 'results', 'baseline'))
    ap.add_argument('--table', nargs='+', metavar='SUMMARY_JSON',
                    help='print a comparison table of finished runs and exit')
    args = ap.parse_args()
    if args.table:
        table(sorted(p for pat in args.table for p in glob.glob(pat)))
    elif args.model:
        run(args)
    else:
        ap.error('give --model or --table')


if __name__ == '__main__':
    main()
