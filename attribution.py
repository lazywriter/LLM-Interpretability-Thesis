#!/usr/bin/env python3
"""
Field-level attribution for IVP-OPEN: Gradient×Input, Integrated Gradients
(pad and mask baselines), raw attention and attention rollout. No perturbation
or subset methods (LIME, SHAP, occlusion): see CLAUDE.md.

Targets (CLAUDE.md, decided 2026-11-03):
  a  log p of the model's OWN final answer expression, teacher-forced on its
     own greedy working: input = chat prompt + the generated tokens (`gen_ids`
     from an eval_open_ivp.py run, --gens). Replies that were cut off or have
     no answer are skipped.
  c  log p of the TRUE solution, "Answer: y =" + answer expression, right
     after the chat prompt (no working). Needs no generations.

Token groups (field_tokens.py): instruction, equation, cond_1..k,
answer_instruction, prompt_other (labels, numbering), template (chat
template), working (target a: the reply before the answer expression),
answer_prefix (target c: "Answer: y ="), target.

Methods, per example (batch size 1, gradients w.r.t. the input embeddings):
  gxi      sum_d grad * emb
  ig_pad   IG (midpoint rule), baseline = pad-token embedding } on the prompt and working
  ig_mask  IG (midpoint rule), baseline = all-zero embedding  } tokens; template, answer
                                                                 prefix and target stay fixed;
           the completeness gap (sum of attributions - [f(x) - f(baseline)]) is saved
  attn     mean over heads of the attention from the target tokens (as queries)
           to every token, per layer
  rollout  attention rollout (Abnar & Zuidema 2020): A' = (A + I) / 2, rows
           normalised, multiplied from the last layer down; read at the target rows

Output (never overwrites): results/attribution/<UTC stamp>_<model>_<target>_<split>/
  attributions.jsonl   one line per example: group sums (signed and absolute)
                       per method, attention per layer and group, IG
                       completeness, labels, and token-level values for the
                       prompt tokens
  meta.json            settings, revision, timing

  python attribution.py --model qwen2.5-math-1.5b-instruct --target c --split dev
  python attribution.py --model phi-4-mini-instruct --target a --gens results/open_eval/<run>
"""
import argparse, math, datetime, json, os, platform, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, 'ivp_open')]
from grade_open import load, extract_answer  # noqa: E402
from field_tokens import item_char_groups, char_groups_to_tokens, condition_roles  # noqa: E402
from baseline_letter_logprobs import (MODELS, TEMPLATE_KWARGS, SEED, build_text, dtype_kwarg,  # noqa: E402
                                      model_revision)

FIXED_GROUPS = {'template', 'answer_prefix', 'target'}   # never replaced by the IG baseline
# GPU dtype when --dtype is not given. Qwen2.5-Math overflows with eager attention in fp16 (every value NaN in the
# 2026-11-04 run; generation used the default sdpa kernel and was fine), so it runs in fp32 (1.5B fits on a T4).
ATTR_DTYPE = {'qwen2.5-math-1.5b-instruct': 'fp32'}
PROMPT_GROUPS_FIXED = ('instruction', 'equation', 'answer_instruction', 'prompt_other')


# ---------------------------------------------------------------- examples
def answer_expression_span(reply):
    """Character range of the final answer expression in a reply, or None."""
    ans, how = extract_answer(reply)
    if ans is None:
        return None, how
    pos = reply.rfind(ans)
    if pos < 0:
        return None, how
    s = ans.rfind('=') + 1 if '=' in ans else 0
    e = len(ans)
    while s < e and (ans[s].isspace() or ans[s] in '$*'):
        s += 1
    while e > s:  # trailing spaces, full stop, markdown bold, LaTeX closers
        if ans[e - 1].isspace() or ans[e - 1] in '.$*':
            e -= 1
        elif ans[s:e].endswith(('\\]', '\\)')):
            e -= 2
        else:
            break
    return ((pos + s, pos + e) if e > s else None), how


def gen_token_spans(tok, gen_ids):
    """Character span of every generated token inside tok.decode(gen_ids)."""
    ends, prev = [], 0
    for k in range(1, len(gen_ids) + 1):
        n = max(prev, len(tok.decode(gen_ids[:k])))
        ends.append(n)
        prev = n
    starts = [0] + ends[:-1]
    return list(zip(starts, ends)), tok.decode(gen_ids)


def build_example(tok, item, target, tkw, gen_row=None):
    """-> dict(ids, groups {name: [token idx]}, target_pos) or (None, reason)."""
    prompt_text, off = build_text(tok, item['prompt'], 'chat', None, None, tkw)
    if target == 'c':
        prefix, expr = 'Answer: y =', ' ' + item['answer_text'].split('=', 1)[1].strip()
        text = prompt_text + prefix + expr
        a, b = len(prompt_text), len(prompt_text) + len(prefix)
        groups = item_char_groups(item, off, len(prompt_text)) + [('answer_prefix', a, b), ('target', b, len(text))]
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, tokmap = enc['input_ids'], char_groups_to_tokens(enc['offset_mapping'], groups, text)
    else:
        if gen_row is None:
            return None, 'no_generation'
        if gen_row.get('truncated'):
            return None, 'truncated'
        gen = gen_row['gen_ids']
        spans, reply = gen_token_spans(tok, gen)
        tspan, how = answer_expression_span(reply)
        if tspan is None:
            return None, f'no_answer_expression ({how})'
        enc = tok(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
        tokmap = char_groups_to_tokens(enc['offset_mapping'], item_char_groups(item, off, len(prompt_text)),
                                       prompt_text)
        n0 = len(enc['input_ids'])
        gmap = char_groups_to_tokens(spans, [('target', *tspan), ('working', 0, tspan[0])], reply)
        for g, idx in gmap.items():
            if g == 'other':   # tokens after the answer expression: not part of the input we need
                continue
            tokmap.setdefault(g, []).extend(n0 + k for k in idx)
        ids = enc['input_ids'] + gen
    tpos = sorted(tokmap.get('target', []))
    if not tpos:
        return None, 'empty_target'
    end = tpos[-1] + 1                     # input only up to the last target token
    tokmap = {g: [k for k in idx if k < end] for g, idx in tokmap.items()}
    tokmap = {g: idx for g, idx in tokmap.items() if idx}
    return {'ids': ids[:end], 'groups': tokmap, 'target_pos': tpos}, None


# ---------------------------------------------------------------- model
def load_for_attribution(model_key, dtype, device, checkpointing):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_name = MODELS[model_key][0]
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(hf_name, attn_implementation='eager', **dtype_kwarg(dtype)).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():          # attribution must be deterministic: no dropout anywhere
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if hasattr(m, 'attention_dropout') and isinstance(m.attention_dropout, float):
            m.attention_dropout = 0.0
    if checkpointing:                  # recompute activations in backward: long inputs fit on a T4
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.eval()
    return tok, model


def decoder_layers(model):
    for path in ('model.layers', 'model.model.layers', 'transformer.h'):
        obj = model
        try:
            for a in path.split('.'):
                obj = getattr(obj, a)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError('decoder layers not found')


class Attributor:
    def __init__(self, model, tok, ig_steps, checkpointing, ig_batch=1):
        import torch
        self.torch, self.model, self.tok = torch, model, tok
        self.emb = model.get_input_embeddings()
        self.ig_steps, self.ig_batch, self.checkpointing = ig_steps, ig_batch, checkpointing
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.pad_vec = self.emb.weight[pad].detach()
        self.layers = decoder_layers(model)
        self._attn, self._record = [], False
        for layer in self.layers:
            layer.self_attn.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self._record:
            w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if w is None:
                raise RuntimeError('attention weights not returned: the model must use eager attention')
            self._attn.append(w[0].float().mean(0).half())   # mean over heads, [L, L]

    def logp(self, embeds, ids, tpos):
        """log p of the target tokens for each input in the batch -> [k]."""
        torch = self.torch
        logits = self.model(inputs_embeds=embeds, use_cache=False).logits[:, [t - 1 for t in tpos]]
        lp = torch.log_softmax(logits.float(), -1)
        tgt = ids[tpos].view(1, -1, 1).expand(lp.shape[0], -1, 1)
        return lp.gather(2, tgt)[..., 0].sum(1)

    def grad(self, embeds, ids, tpos, weights=None):
        """Gradient of sum_k w_k log p(x_k) w.r.t. the inputs, summed over the batch -> ([1, L, d], log p values)."""
        torch = self.torch
        x = embeds.detach().clone().requires_grad_(True)
        self.model.train(self.checkpointing)      # HF checkpointing is only active in train mode (dropout is off)
        lp = self.logp(x, ids, tpos)
        w = torch.ones_like(lp) if weights is None else weights
        (g,) = torch.autograd.grad((w * lp).sum(), x)
        self.model.eval()
        return g.float().sum(0, keepdim=True), lp.detach().tolist()

    def integrated_gradients(self, E, B, ids, tpos):
        """IG with the midpoint rule on [0, 1], `ig_batch` steps per pass. (Gauss-Legendre, Captum's
        default, did not converge here: the integrand along the path is too rough, see NOTES 2026-11-04.)"""
        import numpy as np
        torch = self.torch
        alphas = (np.arange(self.ig_steps) + 0.5) / self.ig_steps
        w = np.full(self.ig_steps, 1.0 / self.ig_steps)
        total = torch.zeros_like(E, dtype=torch.float32)
        for s in range(0, self.ig_steps, self.ig_batch):
            a = torch.tensor(alphas[s:s + self.ig_batch], device=E.device, dtype=E.dtype).view(-1, 1, 1)
            ww = torch.tensor(w[s:s + self.ig_batch], device=E.device, dtype=torch.float32)
            gk, _ = self.grad(B + a * (E - B), ids, tpos, ww)
            total += gk
        return ((E - B).float() * total).sum(-1)[0]

    def run(self, ex):
        torch = self.torch
        dev = self.emb.weight.device
        ids = torch.tensor(ex['ids'], device=dev)
        tpos = ex['target_pos']
        L = len(ids)
        E = self.emb(ids).detach().unsqueeze(0)                         # [1, L, d]
        varied = torch.zeros(L, dtype=torch.bool, device=dev)
        for g, idx in ex['groups'].items():
            if g not in FIXED_GROUPS:
                varied[idx] = True
        res = {}
        g, (f_x,) = self.grad(E, ids, tpos)
        res['logp_target'] = f_x
        res['gxi'] = (g * E.float()).sum(-1)[0]
        for name, base_vec in (('ig_pad', self.pad_vec), ('ig_mask', torch.zeros_like(self.pad_vec))):
            B = E.clone()
            B[0, varied] = base_vec.to(E.dtype)
            attr = self.integrated_gradients(E, B, ids, tpos)
            with torch.no_grad():
                f_b = float(self.logp(B, ids, tpos)[0])
            res[name] = attr
            res[name + '_check'] = {'f_x': f_x, 'f_baseline': f_b, 'sum_attr': float(attr.sum()),
                                    'delta': float(attr.sum()) - (f_x - f_b)}
        self._attn, self._record = [], True
        with torch.no_grad():
            self.model(inputs_embeds=E, use_cache=False)
        self._record = False
        A = self._attn
        self._attn = []
        res['attn_layers'] = [a[tpos].float().mean(0) for a in A]       # per layer: [L]
        v = torch.zeros(L, device=dev)
        v[tpos] = 1.0 / len(tpos)
        eye = torch.eye(L, device=dev)
        for a in reversed(A):
            r = 0.5 * a.float() + 0.5 * eye
            v = v @ (r / r.sum(-1, keepdim=True))
        res['rollout'] = v
        del A
        return res


def sig(x):
    """6 significant digits (rollout masses on prompt fields can be ~1e-5)."""
    return float(f'{float(x):.6g}')


def group_sums(vec, groups):
    v = vec.detach().float().cpu()
    return {g: [sig(v[idx].sum()), sig(v[idx].abs().sum())] for g, idx in groups.items()}


def summarise(item, ex, res, tok, gen_row):
    groups = ex['groups']
    out = {'id': item['id'], 'split': item['split'], 'subtype': item['subtype'], 'category': item['category'],
           'ode_type': item['ode_type'], 'order': item['order'], 'roles': condition_roles(item),
           'distractor_position': item['info'].get('distractor_position'),
           'restates': item['info'].get('restates_condition'),
           'condition_texts': [c['text'] for c in item['conditions']],
           'correct': gen_row.get('correct') if gen_row else None,
           'equals_trap': gen_row.get('equals_trap') if gen_row else None,
           'answer_status': gen_row.get('status') if gen_row else None,
           'n_tokens': len(ex['ids']), 'n_target_tokens': len(ex['target_pos']),
           'group_tokens': {g: len(idx) for g, idx in groups.items()},
           'logp_target': res['logp_target'],
           'gxi': group_sums(res['gxi'], groups), 'ig_pad': group_sums(res['ig_pad'], groups),
           'ig_mask': group_sums(res['ig_mask'], groups),
           'ig_pad_check': res['ig_pad_check'], 'ig_mask_check': res['ig_mask_check'],
           'rollout': {g: sig(res['rollout'][idx].sum()) for g, idx in groups.items()},
           'attn_layers': {g: [sig(a[idx].sum()) for a in res['attn_layers']] for g, idx in groups.items()}}
    # token-level values for the prompt tokens only (figures); working tokens are summarised by group
    tok_group = {k: g for g, idx in groups.items() for k in idx}
    keep = [k for k in range(len(ex['ids'])) if tok_group.get(k) not in (None, 'template', 'working', 'target', 'answer_prefix')]
    attn_mean = sum(res['attn_layers']) / len(res['attn_layers'])
    out['prompt_tokens'] = [[tok.decode([ex['ids'][k]]), tok_group[k], sig(res['gxi'][k]), sig(res['ig_pad'][k]),
                             sig(res['ig_mask'][k]), sig(attn_mean[k]), sig(res['rollout'][k])] for k in keep]
    out['prompt_token_columns'] = ['token', 'group', 'gxi', 'ig_pad', 'ig_mask', 'attn_mean_over_layers', 'rollout']
    return out


# ---------------------------------------------------------------- main
def select_items(items, per_cell, limit):
    if per_cell:
        seen, out = {}, []
        for it in items:
            key = (it['category'], it['subtype'])
            if seen.get(key, 0) < per_cell:
                seen[key] = seen.get(key, 0) + 1
                out.append(it)
        items = out
    return items[:limit] if limit else items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=sorted(MODELS))
    ap.add_argument('--target', required=True, choices=['a', 'c'])
    ap.add_argument('--gens', help='eval_open_ivp.py run folder with generations.jsonl (target a; optional for c, '
                                   'where it only adds the correct/wrong labels)')
    ap.add_argument('--split', default='dev', choices=['dev', 'test', 'all'])
    ap.add_argument('--per-cell', type=int, help='at most N items per category x subtype')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--ig-steps', type=int, default=64, help='midpoint-rule steps')
    ap.add_argument('--ig-batch', type=int, default=1, help='IG nodes per forward/backward pass')
    ap.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'])
    ap.add_argument('--no-checkpointing', action='store_true', help='faster, more memory (small models / short inputs)')
    ap.add_argument('--outdir', default=os.path.join(HERE, 'results', 'attribution'))
    args = ap.parse_args()
    if args.target == 'a' and not args.gens:
        ap.error('target a needs --gens (an eval_open_ivp.py run folder)')

    import torch, transformers
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[
        args.dtype or (ATTR_DTYPE.get(args.model, 'fp16') if device == 'cuda' else 'fp32')]
    items = load(os.path.join(HERE, 'ivp_open', f'ivp_open_{args.split}.jsonl'))
    gens = {}
    if args.gens:
        for line in open(os.path.join(args.gens, 'generations.jsonl'), encoding='utf-8'):
            r = json.loads(line)
            gens[r['id']] = r
        gmeta = json.load(open(os.path.join(args.gens, 'summary.json'), encoding='utf-8'))['meta']
        assert gmeta['model_key'] == args.model, f"generations are from {gmeta['model_key']}, not {args.model}"
        if args.target == 'a':
            items = [it for it in items if it['id'] in gens]
    items = select_items(items, args.per_cell, args.limit)

    checkpointing = not args.no_checkpointing and device == 'cuda'
    tok, model = load_for_attribution(args.model, dtype, device, checkpointing)
    tkw = TEMPLATE_KWARGS.get(args.model)
    att = Attributor(model, tok, args.ig_steps, checkpointing, args.ig_batch)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = os.path.join(args.outdir, f'{stamp}_{args.model}_{args.target}_{args.split}')
    os.makedirs(out_dir, exist_ok=False)
    skipped, t0, n_done = {}, time.time(), 0
    with open(os.path.join(out_dir, 'attributions.jsonl'), 'w', encoding='utf-8') as fout:
        for i, it in enumerate(items):
            ex, why = build_example(tok, it, args.target, tkw, gens.get(it['id']))
            if ex is None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            try:
                res = att.run(ex)
                if not math.isfinite(res['logp_target']):          # overflow in the forward pass: stop early
                    skipped['non_finite'] = skipped.get('non_finite', 0) + 1
                    if skipped['non_finite'] >= 3 and n_done == 0:
                        raise SystemExit(f'{args.model}: non-finite log p on the first {skipped["non_finite"]} examples '
                                         f'({dtype}); rerun with --dtype fp32')
                    continue
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped['out_of_memory'] = skipped.get('out_of_memory', 0) + 1
                print(f'\n  out of memory on {it["id"]} ({len(ex["ids"])} tokens), skipped', flush=True)
                if skipped['out_of_memory'] >= 3 and n_done == 0:             # every item will fail: stop early
                    raise SystemExit(f'{args.model}: out of memory on the first {skipped["out_of_memory"]} examples '
                                     f'({dtype}, checkpointing={checkpointing}, ig_batch={args.ig_batch}); '
                                     'turn checkpointing on or lower --ig-batch')
                continue
            fout.write(json.dumps(summarise(it, ex, res, tok, gens.get(it['id']))) + '\n')
            fout.flush()
            n_done += 1
            print(f'\r{i + 1}/{len(items)}  done {n_done}  ({time.time() - t0:.0f}s, {len(ex["ids"])} tokens)',
                  end='', flush=True)
    print()
    meta = {'experiment': 'attribution', 'model_key': args.model, 'model': MODELS[args.model][0],
            'revision': model_revision(model, MODELS[args.model][0]), 'target': args.target,
            'gens': os.path.basename(os.path.normpath(args.gens)) if args.gens else None,
            'split': args.split, 'per_cell': args.per_cell, 'limit': args.limit, 'n_selected': len(items),
            'n_done': n_done, 'skipped': skipped, 'ig_steps': args.ig_steps, 'ig_rule': 'midpoint', 'ig_batch': args.ig_batch,
            'ig_baselines': {'ig_pad': 'pad-token embedding', 'ig_mask': 'all-zero embedding'},
            'ig_varied_groups': 'all except ' + ', '.join(sorted(FIXED_GROUPS)),
            'attention': 'eager; mean over heads; target tokens as queries', 'rollout': '(A + I)/2, row-normalised',
            'dtype': str(dtype).replace('torch.', ''), 'checkpointing': checkpointing, 'device': device,
            'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None, 'seed': SEED,
            'prompt_version': items[0]['prompt_version'] if items else None,
            'torch': torch.__version__, 'transformers': transformers.__version__, 'python': platform.python_version(),
            'seconds': round(time.time() - t0),
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"{args.model} target {args.target} [{args.split}] done {n_done}/{len(items)} skipped {skipped}  -> {out_dir}")


if __name__ == '__main__':
    main()
