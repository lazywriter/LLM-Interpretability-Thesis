# IVP-MCQ zero-shot baseline: final report (coding side → discussion side)

*2026-11-03. Covers all coding sessions of 2026-11-02/03. Day-by-day detail: `NOTES_FROM_CODE.md`.*

## 1. Summary

- **The models are Qwen2.5-3B-Instruct and Phi-4-mini-instruct.** Seven other models were tried and dropped: 0.5–1.5B Qwen, Qwen3 and Llama-3.2-3B. They were at chance, or their letter scores did not reflect the answer they would actually give.
- **Test accuracy (n = 792, chance 33.3%):** Qwen **43.4%** [40.0, 46.9], Phi **45.7%** [42.3, 49.2]. Both are clearly above chance. They are **not** significantly different from each other (paired exact McNemar p = 0.32).
- **The skill is real, not letter bias.** Scored under all 6 option orders, accuracy stays at 43.3% / 44.8%.
- **Only a minority of items is solved reliably.** 73 (Qwen) and 96 (Phi) items are correct under all 6 orders; random guessing would give about 1. Most other correct answers are guesses. **This decides which items attribution can use.**
- **The pipeline is verified end to end:**
  - every gold label is re-derived from the item text by SymPy (864/864);
  - the grader is checked;
  - the exact token inputs are checked;
  - the letter score equals what the model writes, whenever it answers directly.
- **New finding:** the models do better when a condition's value is the option with the point substituted term by term (e.g. `y(1) = -5 - 4*exp(-1)` for `y = -5 - 4*exp(-x)`) than when checking needs arithmetic (`y(0) = 2`). x0 = 0 conditions, the textbook case, are among the *harder* ones.
- **Their wrong answers show no systematic heuristic.** Near-miss, trap and extra-match options are picked about as often as a random wrong pick would give.

## 2. What was run

| step | what | where |
|---|---|---|
| Baseline | next-token log-probs of `A`/`B`/`C` after the prompt (`letter_from_scores`), per-slice report | `baseline_letter_logprobs.py` |
| Gold check | independent SymPy re-derivation of every answer from the text: ODE residual, each condition, and whether the satisfied conditions *determine* y (rank in dsolve's constants) | `gold_check.py`: 864/864 OK |
| Diagnostic | 10 steps from prompt to grade, run for Llama | `diagnose_colab.ipynb`, run saved in `diagnose_llama_run.ipynb` |
| Option-order test | every item under all 6 option orders (options reordered in memory via `spans`; data files untouched) | `option_permutations.py` |
| Item analysis | overlap, wrong-option types, difficulty, confidence | `analyze_baseline.py`, output in `results/analysis/` |

Environment for reported runs: Colab T4, fp16, torch 2.11, transformers 5.21, batch size 8. Seeds are fixed. Canonical runs are in `results/CANONICAL.json`.

## 3. How the models were chosen (dev, 72 items; for prompt and model tuning only)

**"Letter mass"** is the share of next-token probability on A/B/C. It must be above 0.9 for the letter score to be the model's actual answer, and for log p(letter) to be a valid attribution target.

| model | prompt v1: acc / letter mass | letter-only system prompt: acc / letter mass | verdict |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 40.3 / 0.05 | 34.7 / 1.00 | chance → dropped |
| Qwen2.5-1.5B-Instruct | 40.3 / 0.61 | 30.6 / 1.00 | chance → dropped |
| Qwen2.5-Math-1.5B (base) | 34.7 / 0.50 | n/a | chance → dropped |
| Qwen3-1.7B | 43.1 / **0.000** | 31.9 / 0.98 | scored from letters it would not output → dropped |
| Qwen3-4B-Instruct-2507 | 52.8 / **0.002** | 37.5 / 1.00 | same → dropped |
| Llama-3.2-3B-Instruct | 31.9 / 0.48 | 29.2 / 1.00 (predicts C 60/72) | chance, strong letter bias → dropped |
| **Qwen2.5-3B-Instruct** | **56.9 / 1.00** | 44.4 / 1.00 | **kept, prompt v1** |
| **Phi-4-mini-instruct** | 40.3 / 0.42 | **54.2 / 1.00** | **kept, letter-only** |

Dev overestimated both kept models by about 10 points (test: 43.4 / 45.7). That is expected after picking the best of 14 configurations (7 models × 2 prompts) on 72 items.

## 4. Pipeline checks: is the low accuracy a bug?

Llama-3.2-3B prompted the check (the user suspected a bug). The full diagnostic found none:

- **Labels:** 864/864 re-derived from the text. One instructive case (ivp-0683): a distractor satisfies `y'(1)` and `y'(-1)`, which for `y'' = f(x)` fix only one constant. So the rule must be "*determining* subset", not "two conditions".
- **Grader** = manual recount. **Input:**
  - one BOS token;
  - template correct;
  - Llama's date pinned;
  - letter token ids round-trip;
  - easy control questions score 8/9.
- **The letter score is what the model writes:** 4/4 items where Llama answers directly give the same letter in free generation.
- **Why Llama fails.** When it wants to explain first (top token "To", letter mass ≈ 0.01), the letter score reads a meaningless tail. On the 2 such items shown, its free-text answer was correct while the letter score was wrong. When it answers directly, it follows position: on C items it predicts C 34/36 times, and in the rotation test it picks the same letter under all 3 orders on 61% of items.
- **Batching:** batched and one-by-one scoring give identical predictions. In fp16, log-probs differ by up to 0.19 on low-probability tokens, so 1–2 near-tie test items flip between runs with different batch composition. This is **accepted as is** (no rerun, user decision).

## 5. Final test results

| | Qwen2.5-3B (v1) | Phi-4-mini (+sys) |
|---|---|---|
| accuracy [95% CI] | 43.4 [40.0, 46.9] | 45.7 [42.3, 49.2] |
| B / C | 44.4 / 42.4 | 48.0 / 43.4 |
| order 1 / order 2 | 45.5 / 41.4 | 50.8 / 40.7 |
| `second_order_direct` | 35.6 | 35.6 |
| `C:extra1_direct` | 37.0 | 34.0 |
| B:unrelated trap-pick | 28.8 | 34.1 |
| letter mass | 0.998 | 0.999 |

Slice SE is about ±4 pp at n ≈ 132 and ±7 pp at n = 44. Dev patterns at n = 4 per category did not replicate on test.

## 6. Option-order test: how much is content, how much is position

| | Qwen | Phi |
|---|---|---|
| accuracy, original order | 43.3 | 45.5 |
| **accuracy averaged over 6 orders** | **43.3** | **44.8** |
| **robust-correct (correct under all 6 orders)** | **73 (9.2%)** | **96 (12.1%)** |
| correct under ≥ 5 of 6 | 146 | 155 |
| same option under all 6 (right or wrong) | 15.7% | 19.3% |
| same letter under all 6 | 5.2% | 8.1% |

The number of orders (out of 6) under which an item is correct is **bimodal**, not binomial. k = 6 has 73 / 96 items (guessing: 1.1). k = 0 has 138 / 130 (guessing: 69.5). Items are either solved from the content or consistently missed.

- **Where robust-correct is high:** Phi first_order_direct 25%, first_order_separable 23.5%; Qwen B:duplicate 18%.
- **Where it is near zero:** `second_order_direct` (4.5% / 1.5%) and `C:extra1_direct` (4% / 4%). In the latter, the same-letter rate rises to 16% / 24%: when two conditions conflict at the same point, the models fall back on position.

## 7. Item-by-item analysis (new)

### 7.1 The two models solve partly the same items

| | Qwen | Phi | both | expected if independent |
|---|---|---|---|---|
| correct (original order) | 344 | 362 | 206 | 157 |
| robust 6/6 | 73 | 96 | **35** | 8.8 |
| ≥ 5/6 | 146 | 155 | 67 | 28.6 |

- **35 items are robust for both** (27 B, 8 C). Most are first-order direct (14) or separable (11). Only 1 is second_order_direct.
- **When both are wrong, they pick the same wrong option 68% of the time** (50% if unrelated). Something item-specific pulls both models, but not by option *type* (see 7.2).

### 7.2 Which wrong option is picked

The table shows observed shares, with the share a random wrong pick would give in brackets. All wrong answers, original order:

| | trap | extra-match (satisfies a contradicting condition) | near-miss (satisfies a sufficient condition, too few to determine y) | far-miss |
|---|---|---|---|---|
| Qwen, B | 17.3 (16.6) | – | 24.5 (21.1) | 58.2 (62.3) |
| Qwen, C | – | 25.0 (27.2) | 13.2 (11.2) | 61.8 (61.6) |
| Phi, B | 21.8 (17.7) | – | 21.8 (23.3) | 56.3 (59.0) |
| Phi, C | – | 25.9 (28.1) | 14.7 (11.6) | 59.4 (60.3) |

- **Wrong answers follow no condition-based heuristic.** Every kind is within a few points of random. There is no "checked only one condition" signature (near-miss) and no "used the contradicting condition" signature (extra-match).
- The trap is at most slightly over-picked by Phi (+4 pp).
- On the consistently-wrong items (same wrong option under all 6 orders, n ≈ 25–30 per model and variant), far-misses dominate, also roughly as random would give.

### 7.3 What makes an item hard

Accuracy here is averaged over the 6 orders.

| feature | Qwen | Phi |
|---|---|---|
| max `value_complexity` of sufficient conditions: 0 → 1–2 → 3–4 → 5+ | 32.5 → 42.8 → 46.9 → 48.7 | 36.8 → 44.0 → 48.0 → 48.7 |
| complex (≥ 3) vs. simple (≤ 2) values | 47.3 vs 39.8 | 48.2 vs 41.9 |
| order 1: sufficient conditions all at x0 = 0, yes vs no | 37.7 vs 47.4 | 44.7 vs 50.4 |
| order 1: sufficient set uses a derivative condition, yes vs no | 39.8 vs 47.1 | 39.8 vs 53.5 |
| number of conditions: 2 → 3 → 4 | 46.5 → 41.7 → 39.8 | 49.4 → 42.7 → 38.9 |

- **"Complex" values are easier, the opposite of the README caveat.** The effect holds within 5 of 6 (Qwen) and 4 of 6 (Phi) ODE types, so it is not an ODE-type confound.
- The no-maths "numbers in conditions" shortcut stays at chance on the complex-value items (34.9%), so a plain number-matching shortcut does not explain it.
- **Hypothesis:** a complex value is the option with x0 substituted term by term, so checking it is symbolic pattern-matching (`y(1) = -5 - 4*exp(-1)` vs `y = -5 - 4*exp(-x)`). A plain value needs evaluation and arithmetic (`y(0) = 2` vs `y = 3 - exp(-3*x)` requires exp(0) = 1 and 3 − 1 = 2). The same would explain why x0 = 0 conditions are harder. **`value_complexity` (SymPy op count) does not measure difficulty for these models.**

### 7.4 Confidence

| p(chosen letter) | Qwen: n / acc / robust | Phi: n / acc / robust |
|---|---|---|
| < 0.5 | 23 / 43.5 / 4.3 | 396 / 41.7 / 4.0 |
| 0.5–0.7 | 135 / 33.3 / 3.0 | 279 / 42.3 / 10.8 |
| 0.7–0.9 | 169 / 37.3 / 6.5 | 101 / 65.3 / 36.6 |
| ≥ 0.9 | 465 / 48.6 / 12.3 | 16 / 81.2 / 81.2 |

- **Phi's confidence is informative.** At p ≥ 0.7, accuracy is 65–81%.
- **Qwen is overconfident.** 465 items have p ≥ 0.9, at 48.6% accuracy. A confidence filter would work for Phi but not for Qwen.

## 8. Decisions needed from the discussion side

1. **Which items attribution should use** (the most important decision):
   - (a) robust-correct 6/6: Qwen 73, Phi 96, both 35. Cleanest, but per-subtype n is about 4–20.
   - (b) ≥ 5/6: 146 / 155. Guessing would contribute at most about 14 items (≤ 10%).
   - (c) the consistently-wrong items (51 / 57) as a **contrast set**: what drives a confident wrong choice.

   The lists are written to `results/analysis/<stamp>/item_sets_<model>.json`.

   Note: this is behavioural item *selection*, not an importance score. The attribution target stays log p(original answer) on the original prompt. Please confirm it fits the rule against "correctness-on-perturbed-input" scores.
2. **How to frame the format limitation.** Single-token answers give no room to compute. Twice, anecdotally (1.5B early on, Llama later), models given room to reason got B items right that they missed as a single letter. Should results be framed as "what the model uses when answering in one step"? Should a reasoning condition be added later? That would be a different target, since there is no single answer token.
3. **The value-complexity finding** (§7.3): is it worth a targeted follow-up? For example, compare substitution-friendly and arithmetic-requiring conditions, matched on everything else. And should the README caveat be revised?
4. **Attribution expectations for B:duplicate:** the two conditions are identical text, so which one is labelled "sufficient" and which "distractor" is arbitrary. Any attribution difference between them is a position effect.
5. **C:extra1_direct and second_order_direct** are essentially unsolved (robust ≈ 1–4%). Leave them out of attribution, or keep them as "not solved" controls?
6. **Llama-3.2-3B** (access granted) does not fit letter-score scoring. Drop it for good?

## 9. Proposed next coding steps, pending the decisions above

1. **Span → token mapping utility** (independent of the decisions). It maps every field to token positions inside the chat-formatted input (using `prompt_char_offset`) and verifies itself on all 864 items for both tokenizers. Tokens that straddle two fields get flagged for a rule.
2. **Attribution harness**: Gradient×Input first, then IG (pad and mask baselines), attention and rollout. Batch size 1 on a T4, on the chosen item set. Built-in checks: IG completeness, and content vs. letter-position attribution.

## 10. Caveats

- **Version hash:** the baseline result files have `revision: null` (a transformers-5 bug, fixed later). The revision is taken from the same-day permutation runs (`results/CANONICAL.json`).
- **Batch flips:** 1–2 near-tie items flip between batch compositions in fp16. Accepted.
- **Older CPU dev runs:** the local CPU dev runs from 2026-11-02 (the +prefix experiments) were deleted on purpose. Their numbers are preserved in `NOTES_FROM_CODE.md`.
- **Small samples:** the dev sample (72) is too small for anything but coarse tuning. The free-text reasoning evidence is anecdotal (≤ 12 items).

## 11. Files

| file | purpose |
|---|---|
| `baseline_letter_logprobs.py` | baseline scoring; shared `load_model` / `score_texts` |
| `option_permutations.py` | 6-order consistency test |
| `gold_check.py` | independent SymPy label check |
| `analyze_baseline.py` | this item analysis |
| `diagnose_colab.ipynb`, `diagnose_llama_run.ipynb` | step-by-step diagnostic and the Llama run |
| `baseline_colab.ipynb`, `permutations_colab.ipynb` | Colab runners |
| `results/CANONICAL.json` | which runs are the reported ones |
| `results/analysis/<stamp>_baseline_items/` | `analysis.json` and per-model item sets |
| `NOTES_FROM_CODE.md` | dated log of every session |
