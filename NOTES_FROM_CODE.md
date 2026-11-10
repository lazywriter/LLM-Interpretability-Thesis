# Notes from the coding side

Claude Code appends a dated entry here after each coding session: what was
built or run, key numbers, surprises, open questions. The discussion chat reads
this file to stay in sync.

---

## 2026-11-02 — setup
- `ivp_mcq/` dataset built and verified (864/864), shortcut-audited. See
  `ivp_mcq/README.md`.
- No model runs yet. Next: zero-shot accuracy baseline (see CLAUDE.md).

## 2026-11-02 — zero-shot baseline: code + dev prompt check (no test runs yet)
**Built**
- `baseline_letter_logprobs.py`: one forward pass per item, compares next-token
  log-probs of the letters (`letter_from_scores`), report via `evaluate`, plus
  predicted-letter counts, gold→pred confusion, mass on the letter tokens,
  whether the top-1 token is a letter, and accuracy under the other letter
  spelling (`A` vs ` A`). Per item it also records `prompt_char_offset`, where
  the dataset prompt starts inside the model input; the span → token mapping
  needs it. Writes `results/baseline/<UTC stamp>_<model>_<split>/` and refuses
  to overwrite. `--table` compares runs.
- Formats: `chat` = chat template, prompt as the single user turn (Qwen's
  default system prompt gets added automatically). `raw` = prompt +
  `"\nAnswer:"` for the base Math model. `--answer-prefix "The answer is"` is
  optional: it starts the assistant turn with that text (`v1+prefix`).
- Batched and bs=1 log-probs agree to 3e-5 (left padding with explicit
  position_ids).
- `baseline_colab.ipynb` runs it on a T4.

**Dev runs** (72 items, CPU fp32, `results/baseline_devtune/`). Dev gold letters
are A29/B25/C18, so **always-A = 40.3% on dev**. n=72 gives SE ≈ 5.7 pp.

| model | acc | pred A/B/C | letter mass | top-1 is letter |
|---|---|---|---|---|
| 0.5B-Instruct v1 | 40.3 | 38/34/0 | 0.05 | 0% (always "To") |
| 0.5B-Instruct +prefix | 33.3 | 18/43/11 | 0.82 | 100% |
| 1.5B-Instruct v1 | 40.3 | 41/19/12 | 0.61 | 56% (else "The") |
| 1.5B-Instruct +prefix | 40.3 | 31/13/28 | 0.96 | 100% |
| Math-1.5B (base, raw) | 33.3 | 56/1/15 | 0.50 | 100% |

**Takeaways / for the discussion side**
1. Every model is at or near chance on dev, and the 40.3% scores equal the
   always-A baseline. Test (balanced letters, n=792) will tell more. Expect
   numbers near 33%.
2. **Prompt v1 is a problem for 0.5B-Instruct:** it puts 95% of its mass on
   starting to reason ("To ..."), so the letter argmax comes from a ~5% tail.
   That also matters for attribution, because the target scalar log p(letter)
   is then the log-prob of a token the model would not produce.
   `+prefix` fixes it (mass 0.82–0.96, top-1 always a letter).
   **Decision needed: v1 or v1+prefix for the reported test baseline?**
   I'd lean to `+prefix` for the chat models.
3. Letter bias is strong and varies by model (Math base: 56 of 72 predictions
   are A). Should we also report a letter-debiased score, e.g. subtract each
   letter's mean log-prob over the split? That is a methodology choice, so it
   is not implemented.
4. Model choice: I used Qwen2.5-Math-1.5B **base**, as CLAUDE.md names it.
   Math-1.5B-*Instruct* is wired up too (`qwen2.5-math-1.5b-instruct`) if
   preferred.

## 2026-11-02 (later): is the chance-level dev accuracy a pipeline issue?
Checked possible causes other than "the model can't do it":
- **Scoring pipeline: OK.** 9 trivially easy MCQs, same template/answer line and
  same code (build_text, letter ids, left-padded batch). 1.5B-Instruct 9/9 (v1
  and +prefix), 0.5B-Instruct 8/9 (v1), Math-1.5B 8/9. Letter mass 0.8–1.0 on
  easy items, so 0.5B's "To ..." on IVP items is a reaction to the task, not
  to the format. **But +prefix drops 0.5B to 5/9 on easy items** (it predicts
  mostly A): the prefix is not a free fix for 0.5B.
- **Labels: OK.** check_ivp_mcq on dev 72/72. My own SymPy substitution: gold
  satisfies every y-condition in 36/36 dev B items. Prediction rows match the
  item id, gold letter and correct flag in all 5 runs.
- **Answer format: probably part of it.** 1.5B-Instruct, reasoning allowed
  (greedy, ≤700 tokens, final "Answer: X"), on 12 dev items (6 B, 6 C):
  - B: 4/6, vs 2/6 single-token on the same items. It does solve and check.
  - C: 1/6, vs 1/6 single-token. 5/6 gave no letter: it checks the options
    against *all* conditions, finds that none fits, and answers "None",
    ignoring the instruction about a determining subset. In ivp-0036 it even
    found the right option satisfying condition 1, then rejected it.
  - Tiny sample, diagnostic only (not a methodology change).
- Open: C may be measuring instruction following more than condition use, at
  least at 1.5B. Single-token zero-shot asks for the answer without any room
  to compute, so near-chance accuracy may say more about the format than the
  models' maths. Worth discussing before the test run.

## 2026-11-02 (later): letter-only prompt + bigger models
- Added `--system` (preset `letter-only`: "Reply with exactly one letter (A, B
  or C) and nothing else. Do not explain or show any working."). The item
  prompt is unchanged, so spans still apply via prompt_char_offset.
- Dev (72), CPU fp32:

| run | acc | pred A/B/C | letter mass |
|---|---|---|---|
| 0.5B-it +sys | 36.1 | 41/31/0 | 0.996 |
| 0.5B-it +sys+prefix | 36.1 | 44/18/10 | 0.89 |
| 1.5B-it +sys | 30.6 | 17/48/7 | 1.000 |
| 1.5B-it +sys+prefix | 34.7 | 25/25/22 | 0.91 |

  **Format compliance is now fixed (letter mass ~1, top-1 always a letter),
  but accuracy stays at chance.** So the problem is not that the models "want
  to reason in the response"; without reasoning they cannot do the task in one
  step. (The earlier 12-item reasoning probe: B 4/6 vs 2/6 single-token.)
- Bigger models: added `qwen2.5-3b-instruct`, `qwen3-1.7b`, `qwen3-4b`,
  `qwen3-4b-instruct-2507` (non-thinking), `phi-4-mini-instruct` (3.8B). All fit
  a Colab T4 in fp16. Qwen3 runs with `enable_thinking=False` (template adds an
  empty think block). Smoke-tested Qwen3-1.7B, 2 items, template correct.
- **Local CPU can't run them:** bf16 on this CPU is extremely slow (Qwen3-1.7B
  >10 min for 4 items, still running when I stopped it), and fp32 needs more
  RAM than the 14 GB laptop has free. C: is also nearly full (3 GB), so HF
  cache for new models is `D:/hf_cache`. → Run the bigger models in
  `baseline_colab.ipynb` (updated with SYSTEM / prefix / new models), or
  install CUDA torch for the local RTX 3060 (6 GB: fits ≤1.7B fp16).
- For the discussion side: (1) the C instruction issue from the reasoning
  probe still stands; (2) model choice now has stronger options (Qwen3-4B
  Instruct-2507, Phi-4-mini); (3) attribution with gradients on a 4B model
  on a T4 is feasible at batch 1 but slow for IG (n_steps × items).

## 2026-11-02 (later): Colab dev results for 7 models; adding Llama 3.2 + Gemma
- Colab T4 dev (72) results from the user (tables pasted in chat; per-item
  files still on Drive). Above chance (z > 3): **Qwen2.5-3B-Instruct v1 56.9**
  (letter mass 1.00, predictions 30/24/18 vs gold 29/25/18), Phi-4-mini +sys
  54.2 (mass 1.00), Qwen3-4B-2507 v1 52.8 (but mass 0.002 → scored from the
  tail; with +sys 37.5). Qwen2.5 0.5B/1.5B/Math-1.5B stay at chance under every
  prompt. The system prompt helps Phi and hurts Qwen2.5-3B and Qwen3. Strong
  models avoid the z-trap (3B 25%, 4B 17%). nonhom_complex is 0–25% for nearly
  every model (n=4 each): check on test for a systematic distractor pull.
- **Proposal for the discussion side:** main attribution model
  Qwen2.5-3B-Instruct (same family as 0.5B/1.5B → scale series, clean letter
  target). Choose the prompt per model on dev, restricted to configurations
  with letter mass > 0.9.
- Script now also supports Llama-3.2-1B/3B-Instruct, gemma-2-2b-it,
  gemma-3-1b-it and gemma-3-4b-it (gated; HF_TOKEN). Details:
  - Llama's template date is pinned (`date_string='26 Jul 2024'`) for
    reproducibility.
  - Gemma 2 has no system role, so the system text goes at the top of the user
    turn. That gives the same input Gemma 3's template builds itself, recorded
    as `system_in_user_turn`.
  - Gemma uses eager attention. The 1–2B Gemmas run fp32 on GPU (fp16
    overflow). Non-finite logits now abort the run.
  - Templates, BOS and single-token letters checked locally with the unsloth
    tokenizer mirrors. Model forward passes not yet run.

## 2026-11-02 (later): model decision (user)
- **Thesis models: Qwen2.5-3B-Instruct (prompt v1) and Phi-4-mini-instruct
  (`--system letter-only`).** Recorded in CLAUDE.md → Current decisions.
- **Dropped as not up to the task:** Qwen2.5-0.5B/1.5B-Instruct,
  Qwen2.5-Math-1.5B, Qwen3-1.7B and Qwen3-4B-Instruct-2507. They were at
  chance on dev under every prompt, or scored above chance only from
  non-output letters (Qwen3, letter mass < 0.01). The entries above that
  discuss them are kept as the record of why.
- Llama-3.2-3B-Instruct: HF access requested, not yet granted. Considered
  later; already supported in the script.
- `baseline_colab.ipynb` now defaults to `SPLIT='test'` with those two models
  and their prompts (one click). The test baseline has not been run yet.

## 2026-11-02 (later): TEST baseline, Qwen2.5-3B-Instruct (v1) + Phi-4-mini (+sys)
Run on Colab T4 by the user. Table pasted in chat; per-item files are on Drive,
not yet copied here. Each model ran twice with identical numbers
(deterministic).

| | Qwen2.5-3B (v1) | Phi-4-mini (+sys) |
|---|---|---|
| all (n=792) | **43.4** [39.9, 46.9] | **45.7** [42.2, 49.2] |
| B / C | 44.4 / 42.4 | 48.0 / 43.4 |
| order 1 / 2 | 45.5 / 41.4 | 50.8 / 40.7 |
| second_order_direct | 35.6 | 35.6 |
| C:extra1_direct | 37.0 | 34.0 |
| B:unrelated trap-pick | 28.8 | 34.1 |
| pred A/B/C (gold 264 each) | 247/294/251 | 210/246/336 |
| letter mass | 0.998 | 0.999 |

- Both are well above chance (z = 6.0 and 7.4), but far below dev (56.9 and
  54.2). With 72 items and picking the best prompt/model, dev overestimated
  (winner's curse). The Qwen–Phi gap is within noise; a paired test needs the
  per-item files.
- Slice SEs: ±4.3 pp at n≈132, ±7.4 pp at n=44. The dev "nonhom_complex ≈ 0%"
  pattern does not replicate (38.6 / 40.9).
- Near chance: second_order_direct (both 35.6), C:extra1_direct (37 / 34),
  i.e. one contradicting condition at the same point and derivative order as a
  sufficient one. Trap-pick is at chance: the trap neither attracts nor repels.
- **Key issue for attribution:** under a guess-correction model
  (acc = k + (1−k)/3), only k ≈ 15% (Qwen) and 19% (Phi) of items are
  "known". Of ~344 / 362 correct items, roughly 120 / 147 are known and the
  rest are lucky guesses. Attribution averaged over "correct" items would be
  mostly guess attributions. **Needs a decision:** how do we select items
  where correctness is genuine? Options: a high-confidence filter
  (probs_over_letters), agreement between the two models, or consistency
  under option-order permutation. The last is a behavioural reliability
  filter, not an attribution method; options would be permuted at run time,
  not by editing the data.

## 2026-11-02 (later): Llama-3.2-3B dev + diagnostic notebook
- Llama-3.2-3B-Instruct dev (72): v1 31.9% (letter mass 0.48, pred 17/12/43),
  +sys 29.2% (mass 1.00, pred **0/12/60**). At chance with a strong C bias.
  The user suspected a bug, so I built an end-to-end diagnostic.
- `gold_check.py` (new, repo root): an independent SymPy check working from
  the item text only. For every option it checks the ODE residual and each
  condition, and whether the satisfied conditions determine the solution
  (rank of the conditions in dsolve's constants). It derives the answer and
  compares it with the label and the `satisfies_conditions` metadata.
  **864/864 OK.** A first version that only counted conditions flagged
  ivp-0683: there the distractor satisfies y'(1) and y'(-1), which for
  y'' = f(x) fix only one constant. The label is right; the "determining"
  rule matters.
- `diagnose_colab.ipynb` (new), 10 steps: raw items → gold check → letter
  token ids → exact model input token by token → one-by-one next-token
  scores (top-10) → production script vs. one-by-one → greedy free-text
  answers → grader input/output → easy controls → **option-rotation test**
  (same option text vs. same letter across 3 orders, done in memory only).
  Tested locally with Qwen2.5-0.5B on 12 items: all steps run. Batched vs.
  one-by-one differ by 5e-5. Qwen-0.5B picks the same letter under all
  rotations on 92% of items (pure letter bias).
- Llama diagnostic still to be run on Colab by the user.
- Fix: Colab now runs transformers 5 (Python 3.13). `config._commit_hash` no
  longer exists, so the diagnostic crashed. The baseline script failed
  silently there: **Colab result files so far likely have `revision: null`.**
  `model_revision()` now reads the commit hash from the HF cache snapshot
  folder; it gives the same hash as the old method on transformers 4.49. Model
  loading uses `dtype=` on transformers ≥ 4.56 (`torch_dtype` is deprecated).

## 2026-11-02 (later): Llama-3.2-3B diagnostic run (Colab, transformers 5.21, T4 fp16)
The user saved the run as `diagnose_llama_run.ipynb`. Llama-3.2-3B, prompt v1, dev 72.
**Verdict: no pipeline bug. Llama's 31.9% is the model's real behaviour under
single-letter scoring.**
- Data/labels: gold re-derived for 72/72 (864/864 locally). Grader
  `evaluate` = manual recount (23/72). Gold column = dataset answers.
- Input: 1 BOS, date pinned, template correct, letter ids 32/33/34 round-trip.
  Easy controls 8/9 with letter mass ~1.0. Revision now recorded (0cb88a4f…).
- Reproducible: 31.94% and preds 17/12/43, identical to the earlier Colab
  run.
- **Letter score = what the model writes**, whenever it answers directly:
  4/4 shown items give the same letter in greedy generation.
- **When it writes first (top-1 "To", letter mass ≈ 0.01), the letter score
  reads a meaningless tail.** In both such items shown, the free-text
  reasoning was correct and picked A, while the letter score said B. So on
  ~half the items (mean mass 0.48) the v1 score is not the model's answer.
- **Variant-dependent letter bias:** on C items Llama predicts C 34/36 times
  (gold C = 7/36). On B items predictions are spread out (17/10/9).
- Rotation test: same letter under all 3 option orders 61%, same option text
  4%, correct in all 3 orders 1.4% (below the 3.7% chance rate). Choices
  follow position, not content. Mean p(pred) is higher when wrong (0.73) than
  when right (0.66).
- Batched vs. one-by-one log-prob diff is 0.19 on Colab fp16 (5e-5 locally
  fp32). Predictions agree 72/72 and accuracy is identical. Probably fp16
  arithmetic on low-probability tokens; added step 6b (no-padding vs.
  padded batch) to confirm. For attribution use batch size 1 anyway.
- **For the discussion side:**
  (1) Llama is unusable for letter-score scoring: v1 reads a tail on half the
      items, and +sys gives pure C bias (0/12/60).
  (2) With reasoning it solved the 2 easy B items it reasoned on (anecdotal):
      the single-token format again looks like the bottleneck.
  (3) **Run the rotation test for Qwen2.5-3B and Phi-4-mini**: it measures how
      much of their 43–46% is content-driven. This is the item filter
      proposed earlier.
  (4) B:duplicate items: the two conditions are identical text, so the
      "sufficient" vs "distractor" label between them is arbitrary. The
      attribution analysis should treat them symmetrically (any difference
      is a position effect).

## 2026-11-02 (later): option-order consistency test (set up, not yet run on test)
- `option_permutations.py` (new) scores every item under all 6 option orders.
  The options are reordered in memory via `spans`; the data files are
  untouched. Same prompt format and letter-log-prob scoring as the baseline.
  Per item: n_correct (0–6), the original option picked per order,
  `content_consistent`, `robust_correct` (all 6 correct; chance 0.14%) and
  `letter_consistent`. The summary has acc_original_order, acc_mean_over_orders
  (letter bias averages out), a histogram of k-of-6 correct vs. the
  random-guess binomial, and `robust_correct_ids`, the candidate item set for
  attribution.
- Behavioural reliability filter only, not an attribution method. The
  attribution target and inputs stay the original prompt. **The discussion
  side should confirm it as the item-selection rule.**
- Refactor: `baseline_letter_logprobs.py` now exposes `load_model()` and
  `score_texts()`, which both scripts use. The refactored baseline reproduces
  the earlier dev run exactly (log-prob diff 0.0, all fields identical).
- Local check (Qwen2.5-0.5B, dev): the original order matches the baseline
  on 72/72; a hand-checked permuted prompt is recorded correctly. 0.5B is the
  textbook letter-bias picture: never picks C, 65/72 items correct in exactly
  2 of 6 orders, mean-over-orders 34.0%, robust 0%.
- Colab: `permutations_colab.ipynb` runs Qwen2.5-3B (v1) and Phi-4-mini
  (+sys) on test.

## 2026-11-02 (later): option-order test results, TEST (Colab T4 fp16)
| | Qwen2.5-3B (v1) | Phi-4-mini (+sys) |
|---|---|---|
| acc, original order | 43.3 | 45.5 |
| **acc, mean over 6 orders** | **43.3** | **44.8** |
| **robust-correct (6/6)** | **9.2% (73 items)** | **12.1% (96 items)** |
| correct in ≥5/6 | 146 | 155 |
| content-consistent (same option ×6) | 15.7% | 19.3% |
| letter-consistent (same letter ×6) | 5.2% | 8.1% |
| trap-pick over all orders | 31.6% | 33.2% |

- **The skill is real:** accuracy barely drops once letter bias is averaged
  out (43.3 → 43.3, 45.5 → 44.8). Position bias is small, unlike Llama's 61%
  letter-consistency. Phi leans to C (1971 of 4752 predictions; uniform
  would be 1584).
- **The k-of-6 histogram is bimodal, not binomial.** k=6: 73/96 items vs. 1.1
  expected under guessing; k=0: 138/130 vs. 69.5. Items are either solved
  from the content or consistently missed. Consistently picking the same
  WRONG option: ~6.5% (Qwen) and 7.2% (Phi) of items. These are a natural
  contrast set ("what made it choose that wrong option").
- Robust-correct is close to the earlier guess-correction estimate of "known"
  items (15% / 19%), a bit stricter.
- Slices: highest robust for Phi first_order_direct (25%) and separable
  (23.5%), and Qwen B:duplicate (18%). Near zero for **second_order_direct**
  (4.5% / 1.5%, i.e. not solved at all) and **C:extra1_direct** (4% / 4%,
  where letter-consistency rises to 16% / 24%: direct conflicts fall back on
  position).
- Small discrepancy: the original order here gives 343 (Qwen) and 360 (Phi)
  correct vs. 344 / 362 in the baseline run. 1–2 near-tie items flip because
  the batches are composed differently in fp16 (same effect as the 0.19
  log-prob diff in the Llama diagnostic). For item-level sets used in
  attribution, rescore at batch size 1 so membership doesn't depend on batch
  composition.
- **For the discussion side, attribution item set:** (a) robust 6/6 (73 / 96
  items; tiny per-subtype counts), or (b) ≥5/6 (146 / 155; at most ~14 items
  expected by chance, so ≤10% contamination). Plus the consistently-wrong set
  as a contrast. Per-item files are on Drive (`results/option_permutations/`),
  not yet copied here.

## 2026-11-02 (later): final baseline run prepared
- `final_baseline_colab.ipynb` (new): the canonical test results. Runs the
  baseline and the option-permutation test for Qwen2.5-3B (v1) and
  Phi-4-mini (+sys) at **batch size 1** (no batch-composition flips), into
  `results/final/{baseline,option_permutations}/`.
- Then it checks: the original order in the permutation run must equal the
  baseline exactly; the revision is recorded and identical in both runs; it
  lists which items flipped vs. the earlier batch-size-8 runs. It writes
  `results/final/MANIFEST.json` (run folders, key numbers, sha256 of code and
  data; never overwritten). Tested locally (Qwen-0.5B, 6 dev items): all
  checks pass, identity-vs-baseline log-prob diff exactly 0.0.
- The Drive results copied here (`results-20261102T180536Z-1-001/`): every
  Colab baseline summary has `revision: None` (confirms the transformers 5
  bug); the two permutation runs have it. The local `results/` folder
  (`baseline_devtune/`, 9 CPU dev runs incl. the +prefix runs) is in the
  Windows Recycle Bin (deleted 2026-11-02 23:37); restoring it is up to the
  user.
- **Update (user decision): no batch-size-1 rerun.** The existing
  batch-size-8 test runs are final; `final_baseline_colab.ipynb` was removed
  unused. The canonical runs are listed in `results/CANONICAL.json`, with
  numbers checked against the summaries. Duplicate baseline runs are
  bit-identical. The baseline revision is taken from the same-day permutation
  run, since the baseline files have null (stated as such). The 1–2 near-tie
  flips stay as a documented caveat.
- `results/` now holds the Drive copy (22 runs). The older local
  `baseline_devtune/` CPU runs were deleted on purpose by the user; their
  numbers survive in the entries above.

## 2026-11-03: item-by-item analysis + final baseline report
- `analyze_baseline.py` (new) runs on the runs listed in `results/CANONICAL.json`.
  Output: `results/analysis/20261102T184529Z_baseline_items/` (`analysis.json`
  and `item_sets_<model>.json` with robust_6of6 / ge5of6 / consistently_wrong).
- **Paired comparison:** Qwen vs. Phi is not significant (exact McNemar
  p = 0.32; sign test on n_correct-of-6, p = 0.60).
- **Overlap:** 35 items are robust for both models (8.8 expected if
  independent); 27 B, 8 C, mostly first-order direct and separable. When both
  are wrong, they pick the same wrong option 68% of the time (50% chance).
- **Wrong-option kinds:** trap, extra-match, near-miss and far-miss are all
  within a few points of a random wrong pick. No "checked one condition"
  signature. Phi's trap is +4 pp at most.
- **Difficulty:** harder `value_complexity` → HIGHER accuracy (32.5 → 48.7
  for Qwen, 36.8 → 48.7 for Phi). It holds within 5/6 (Qwen) and 4/6 (Phi)
  ODE types, and the numbers-in-conditions shortcut stays at chance there
  (34.9%). Hypothesis: complex values are term-by-term substitutions of the
  option (symbolic matching), plain values need arithmetic. x0 = 0-only
  conditions are harder for order 1 (37.7 vs 47.4 Qwen; 44.7 vs 50.4 Phi).
  More conditions → lower accuracy.
- **Confidence:** informative for Phi (p 0.7–0.9: 65% acc; p ≥ 0.9: 81%,
  n = 16) but not for Qwen (465 items at p ≥ 0.9 with 48.6% acc).
- Report for the discussion side: `REPORT_2026-11-03_baseline.md` (also
  published as an artifact page). §8 lists the 6 decisions needed.

## 2026-11-03: easy-arithmetic sanity test (set up; Colab run pending)
- User request: check that the pipeline gives very high accuracy on trivial
  maths in the same format before moving on.
- `gen_easy_mcq.py` → `easy_mcq/easy_mcq_test.jsonl` (seed 7): 144 items,
  6 operations × 24 (`+ - * / **`, `a + b * c`), 3 options with one correct.
  Letters AND the rank of the correct value are 8/8/8 per category; all
  no-maths heuristics (middle, largest, smallest, closest to mean, always A,
  length) score 33.3%. Verified with Python eval(). A first version had the
  answer as the middle value too often (37.5% for "pick middle") and 8
  duplicate powers; both fixed.
- Pipeline change, minimal: `--data PATH` on `baseline_letter_logprobs.py`
  and `option_permutations.py` (default unchanged; the data path is recorded
  in `meta.data`). IVP regression run unchanged.
- **Local result with Qwen2.5-0.5B (CPU):** letter score picks **A on all 144
  items** (33.3%); with letter-only, 142 A. In free text it often solves the
  item ("The value of the expression is: C) 57", 3/4 on a spot check), but
  also writes wrong values on others. The first-token letter is a reflex
  (p(A) ≈ 0.2–0.45 next to "The"/"To"), not the computed answer.
- `easy_control_colab.ipynb`: Qwen2.5-3B + Phi-4-mini; letter score under
  both prompts, 6-order test with the thesis prompt, and greedy free
  generation on all 144 (letter read from the text) vs. the letter score.
  Results go to `results/easy_control/`. Tested end to end locally with 0.5B
  on 12 items.
- Colab GPU quota ran out, so the sanity test moves to Kaggle (free T4/P100,
  16 GB, same fp16 setup). `easy_control_kaggle.ipynb` is the same notebook
  with a Kaggle setup cell (finds the files under /kaggle/input, copies them
  to /kaggle/working/Thesis, checks the GPU and internet) and a final cell
  zipping `results/easy_control/`. Upload bundle:
  `kaggle_easy_control_files.zip` (5 files, 20 KB). Tested locally with
  simulated Kaggle paths, nested dataset folder included.

## 2026-11-03: easy-arithmetic sanity test, RESULTS (Kaggle T4, fp16, transformers 5.20.2)
Same model revisions as IVP (aa8e7253…, cfbefacb…). Results in `results/easy_control/`.

| category (24 each) | `*` | `**` | `/` | `+` | `-` | `a+b*c` | all |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B v1, letter score | 100 | 100 | 91.7 | **37.5** | **45.8** | 45.8 | 70.1 |
| Phi-4-mini +sys, letter score | 91.7 | 91.7 | 66.7 | **33.3** | **37.5** | 50.0 | 61.8 |

- **Not "very high" overall: the result splits by operation.** `*`, `**`
  and `/` are near-perfect (mean over 6 orders: Qwen 99/99/96, Phi 94/89/78);
  `+`, `-` and `a+b*c` are at or near chance (Qwen 42/42/44, Phi 34/37/44).
  Examples (Qwen, original order): 17+40 → picks 58; 7+8 → 12; 24+25: 0/6
  orders correct.
- **Not a pipeline bug:** the written answer equals the letter score on
  99.3% (Qwen) and 100% (Phi) of items; both start with the letter
  immediately (100%). The other prompt changes nothing (Qwen 70.8, Phi 60.4).
  The same pipeline reads near-perfect answers on `*`/`**`.
- **Not distractor construction:** within `+`/`-`, accuracy does not depend on
  the nearest-distractor distance, a shared last digit or the correct-value
  rank. Phi only: strong A bias when unsure (510 of 864 picks A over the 6
  orders; letter-consistent 29.9% vs. 8% on IVP).
- `a+b*c`: wrong picks lean to the LARGER option (34% / 42% of picks), e.g.
  the `(a+b)*c` slip.
- **Link to IVP:** checking an IVP option against a plain condition value
  needs exactly this addition/subtraction of constants (C = -4 - e + e …).
  That fits §7.3 of the report: substitution-style values are easier,
  arithmetic-style values harder.
- **Not tested yet (the open question):** can the models add when allowed
  to write the number first? The free-text check here only saw immediate
  letters. A 5-minute Kaggle check would settle it: (a) "17 + 40 =" with
  no options, (b) "write the value, then the letter".

## 2026-11-03: open-answer arithmetic test (set up; Kaggle run pending)
- Question (user): can the models calculate at all, without options? The
  "tokenization" idea was discussed: the same numbers are tokenized the same
  way in `*` (aced) and `+` (chance) items, and the two models tokenize
  numbers differently but fail alike, so tokenization is an unlikely main
  cause (not verified by tokenizing; that check was stopped at the user's
  request).
- `gen_easy_open.py` → `easy_open/easy_open_test.jsonl`: the SAME 144
  expressions and ids as easy_mcq, no options. Two prompts per item:
  "Answer with the number only." and "Show your working briefly, then
  'Answer: <number>'".
- `open_arithmetic.py`: greedy generation (same model loading and template
  as the baseline, no system message). Number read from the reply: first
  integer, or the one after the last "="; working mode uses the last
  "Answer:". Parser unit-tested. Output: `results/easy_open/`.
- Kaggle: `easy_open_kaggle.ipynb` + `kaggle_easy_open_files.zip` (4 files).
  Tested locally with simulated Kaggle paths. Qwen2.5-0.5B on 6 addition
  items: number-only 4/6, working 6/6 (its MCQ letter score was at chance;
  6 items, anecdotal).

## 2026-11-03: change of direction (user decision), CLAUDE.md updated
- **Research focus is now representations: probing, RSA, CKA.** How the
  models solve the problems and where they go wrong. Answer correctness is
  context and a label, not the goal.
- **Answer format:** whichever gives higher dev accuracy. Open-ended solving
  (no options, SymPy-graded `Answer: y = ...`) is preferred when it is at least
  as good. MCQ letter scoring stays available; its test baseline is done.
- **Open issue to settle: variant C without options.** Different sufficient
  subsets can give different valid solutions. Accept any solution of a
  determining subset, or rephrase or drop C.
- The earlier attribution plan (G×I, IG, attention, letter target) is on hold.
  Its perturbation constraints are kept in case a perturbation analysis comes
  back.
- New next steps in CLAUDE.md: (1) open-answer arithmetic control (Kaggle
  pending), (2) open-ended IVP solving on dev with SymPy grading vs. MCQ,
  (3) span → token mapping, (4) hidden-state extraction harness, then probing
  / RSA / CKA once the details are decided.
- For the discussion side to decide: layers, pooling, probe types and labels,
  and which RSA/CKA comparisons.
- **Variant C is dropped for now (user decision 2026-11-03).** Without options
  it has no unique correct answer. Work continues on variant B only (432
  items: dev 36, test 396); an alternative to C will be designed later. C items
  stay in `ivp_mcq/` untouched and are filtered out by `variant`. CLAUDE.md
  updated.

## 2026-11-03: open-answer arithmetic, RESULTS (Kaggle T4, fp16, tf 5.20.2)
Same 144 expressions as the easy MCQ, no options. Same revisions as before.
Results in `results/easy_open/`. Every stored grade was re-checked with
Python eval() (all match).

| % correct | add | sub | mul | div | pow | a+b*c | all |
|---|---|---|---|---|---|---|---|
| Qwen, MCQ letter score | 37.5 | 45.8 | 100 | 91.7 | 100 | 45.8 | 70.1 |
| **Qwen, open, number only** | **100** | **100** | 100 | 100 | 100 | 54.2 | **92.4** |
| **Qwen, open, with working** | 100 | 100 | 100 | 100 | 100 | **100** | **100** |
| Phi, MCQ letter score | 33.3 | 37.5 | 91.7 | 66.7 | 91.7 | 50.0 | 61.8 |
| **Phi, open, number only** | **83.3** | **87.5** | 100 | 95.8 | 91.7 | 20.8 | **79.9** |
| **Phi, open, with working** | 100 | 100 | 100 | 100 | 100 | **100** | **100** |

- **Large improvement without options.** Qwen goes from 70 to 92 (number
  only) to 100 (working); Phi from 62 to 80 to 100.
- **The models CAN add in one step.** Qwen writes 57 for 17 + 40 directly,
  yet picks the wrong letter in the MCQ. This refutes my earlier hypothesis
  ("cannot compute sums in one step") and the tokenization idea. **The MCQ
  add/sub failure is in matching the computed value to the right option
  letter, not in the arithmetic.** Qwen has 37 items with the number right
  but the MCQ wrong (mostly add/sub, mostly picking A); Phi has 41.
- **What still fails in one step: two operations (a + b*c)**, at 54% / 21%.
  Errors include dropping a (b*c), off-by-10, and other carry-like slips.
  With working it is 100%.
- Format compliance: number-only replies are a bare number in 144/144 (Qwen)
  and 138/144 (Phi). With working, Qwen ended 16 replies with "Final answer:
  \(23\)" instead of "Answer:"; they were parsed via the last number and all
  verified correct. No reply hit the token limit.
- **Implication (supports the 2026-11-03 decision):** the open-ended format
  removes a selection step the models handle badly. Expect open-ended IVP
  accuracy to rise, most of all with working. For the representation
  analysis, prompt-side hidden states are the same in both formats; only the
  correctness label changes.

## 2026-11-03: FINAL format decision + plan for 2026-11-04 (user), CLAUDE.md updated
- **Format, final:** open-ended problems with working allowed, ending with
  `Answer: y = ...`, graded by SymPy equivalence. The reason is accuracy (the
  easy control: MCQ 70/62, open number-only 92/80, open with working
  100/100). MCQ letter scoring is retired; its results stay on record.
  Variant C stays dropped; variant B only (432 items).
- **Models are reopened:** every candidate (including those dropped under
  MCQ) gets re-evaluated open-ended and then re-shortlisted on that accuracy.
- **Analysis order: attribution first (G×I, IG, attention/rollout), then
  representation analysis (probing, RSA, CKA).** This reverses the earlier
  "attribution on hold".
- **For the discussion side, needed before attribution:** what is the
  attribution TARGET for an open-ended answer with working? There is no single
  answer token. Candidates (in CLAUDE.md): log p of the final answer tokens,
  teacher-forced on the model's own working; per-step attribution; or log p
  of the true solution.
- Plan for 2026-11-04: (1) open-ended B dataset via generator + SymPy
  answer grading + re-evaluation of all models with working (batched
  generation, dev first, then full B for the strongest); (2) shortlist;
  (3) attribution on the shortlist (span → token mapping needed); (4)
  representation analysis.

## 2026-11-03 (evening): IVP-OPEN dataset + model-evaluation script
- **`ivp_open/` (new working dataset), built by `gen_ivp_open_dataset.py`
  (seed 23, 95 s).** It reuses the IVP-MCQ construction (imported, not
  copied), so the labels stay exact.
  - 864 items = 18 categories × 4 subtypes × 12; dev 72 (1 per cell),
    test 792.
  - Subtypes: `none` (plain IVP), `duplicate`, `rescaled`, `unrelated`
    (trap = z read as y). Exactly 12 per cell, the distractor position
    balanced, no duplicate problems. No variant C.
  - **Additions to flag:** the user asked for a larger set (864 vs. the 432
    B items) covering everything the stronger models can solve. I added
    the **`none` subtype**: the easiest items, and a control for "does the
    extra condition matter?" in the attribution and representation work.
  - Prompt `open-v1`: "Solve the initial value problem below." + equation +
    conditions + "Show your working, then finish with a final line of the
    form "Answer: y = <expression>", writing the expression in plain text
    (for example: y = 3*exp(2*x) - x^2)."
- **`ivp_open/grade_open.py`:** finds the final answer (the last "Answer:"
  line, else `\boxed{}`, else the last "y =" line), parses plain text and
  LaTeX, rejects free constants (C1 …), and checks equivalence numerically at
  5 points. It also records whether the answer solves the ODE, which
  conditions it satisfies, and for `unrelated` whether it follows z read as
  y or equals the trap. 27 parser unit cases pass.
- **`ivp_open/check_ivp_open.py`:** independent, from the text (via
  gold_check helpers). **864/864 pass.** Checked: the answer solves the ODE
  and satisfies all y-conditions; the conditions determine y; distractor
  properties per subtype; the trap behaves as described; spans are exact;
  the grader accepts every true answer and rejects every trap.
- **`eval_open_ivp.py`:** one model per run, batched greedy generation
  (default 768 new tokens, batch 8), progress saved per batch, a `--thinking`
  option for Qwen3, and a raw format for base models. It records per item:
  correct, solves_ode, satisfies, z_as_y, truncated. `--table` ranks runs.
  - Local smoke test (Qwen-0.5B, CPU) ran end to end. At 300 tokens every
    reply was cut off mid-working, so the default was raised to 768, and
    truncation is reported in the table.
- **Kaggle:** `open_eval_kaggle.ipynb` + `kaggle_open_eval_files.zip`
  (95 KB). It runs 8 candidates on dev, deletes each model's download
  afterwards, reads `HF_TOKEN` from Kaggle Secrets (needed for Llama), and
  ranks them with 95% Wilson intervals (dev n=72 is small). Tested locally
  with simulated Kaggle paths. **Not yet run on Kaggle.**

## 2026-11-03 (evening): IVP-OPEN dev results, 8 models (Kaggle T4, fp16, greedy, max 768 new tokens)
- Runs: `results/open_eval/20261103T08*_dev/` (as run) and `*_regraded20261103T0934*` (same replies,
  grader fixed for `\[..\]` + trailing full stop, `|x|`, `\cosh`; +1 item for Qwen2.5-3B and Qwen2.5-Math only).
  The parser tests are now saved in `ivp_open/test_grade_open.py` (27/27 pass). `eval_open_ivp.py --regrade` added.
- Dev accuracy (n=72, 95% Wilson CI) | truncated at 768 | accuracy on replies that finished:
  qwen3-1.7b 69.4 [58,79] | 19 | 91% · phi-4-mini 66.7 [55,77] | 16 | 86% · qwen2.5-math-1.5b-instruct 63.9 [52,74] | 20 | 87% ·
  qwen2.5-3b 59.7 [48,70] | 23 | 86% · qwen3-4b-2507 58.3 [47,69] | 32 | 98% · qwen2.5-1.5b 37.5 | 22 | 54% ·
  llama-3.2-3b 33.3 | 18 | 44% · qwen2.5-0.5b 16.7 | 34 | 32%.
- **Main finding: the 768-token cap, not maths, is the largest error source for the top 5.** Truncated replies
  are long Markdown/LaTeX write-ups (no loops) cut off just before the Answer line, mostly on order-2 items
  (e.g. Qwen3-4B: 24/36 order-2 truncated). Almost no truncated reply is graded correct. The top-5 ranking is
  a tie within CIs and is confounded by verbosity. Recommend a rerun at 1536-2048 tokens before shortlisting.
- Clear drops (wrong even when finished): Llama-3.2-3B (44% of finished; sign errors, e.g. 5.6% on `none`),
  Qwen2.5-1.5B (54%), Qwen2.5-0.5B (32%).
- z-trap (unrelated, answer = z read as y): Qwen2.5-3B and Qwen2.5-1.5B 16.7%, others 0-5.6%.
- Odd: Phi and Qwen2.5-3B do worse on `none` (44%, 39%) than on items with a distractor; on dev each cell has
  only 18 items, and several `none` items are truncated, so don't read much into it yet.
- (detour, for the thesis guide) `dataset_overview_kaggle.ipynb`: a description of IVP-OPEN with distributions, figures and a live
  verification run (864/864). Built by `make_dataset_nb.py`; upload `kaggle_dataset_overview_files.zip`. The figures are also in
  `figures/dataset/`. No change to the data or methods.

## 2026-11-03: model shortlist (user decision)
- **Shortlist: Phi-4-mini-instruct, Qwen2.5-Math-1.5B-Instruct, Qwen3-4B-Instruct-2507.** Dev accuracies 66.7 / 63.9 / 58.3%;
  on replies that finished, 86 / 87 / 98%. Two families, one maths-specialised model, sizes 1.5–4B.
- Not chosen: Qwen3-1.7B (69.4%, statistically tied with the top), Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-3B, Qwen2.5-0.5B.
- Recorded in CLAUDE.md ("Models"). Open: the token limit for shortlist runs (≥1536 recommended, since Qwen3-4B was cut
  off on 32/72 dev replies at 768); the attribution target (a/b/c in CLAUDE.md).

## 2026-11-03: attribution target decided; groundwork for attribution
- **Target (user): (a) main, (c) control.** (a) = log p of the model's own `Answer:` expression, teacher-forced on its own
  greedy working. (c) = log p of `Answer: y = <true solution>` directly after the prompt. Attribution goes to the prompt
  fields; the working gets its own bucket. Recorded in CLAUDE.md.
- `field_tokens.py`: maps the dataset spans to token indices inside the chat template. A first version failed on every
  prompt because Qwen merges a field's last character with the following newline (`.\n\n`, `)\n`); fixed by counting only
  non-whitespace characters in the overlap. Self-check 864/864 (Qwen2.5 tokenizer). Mean tokens per prompt: instruction 8,
  equation 10, conditions 25 in total, answer instruction 46, labels/numbering 13, chat template 29.
- `eval_open_ivp.py`: now saves `gen_ids` (exact generated tokens) and has `--resume`; both tested locally.
- `open_eval_test_kaggle.ipynb`: the shortlist on test (792), 1536 new tokens, Qwen3-4B on GPU 0 and Phi + Qwen2.5-Math on
  GPU 1, ~3–4 h. Not run yet.

## 2026-11-04: attribution scope (user) and the attribution harness
- **Scope (user):** attribution is kept short: only Gradient×Input, IG (pad and mask baselines) and attention/rollout.
  No LIME, SHAP, occlusion or other perturbation/subset methods: a small change to an ODE gives a different problem,
  so "the model should still be right" is not a fair expectation. Most of the time goes to probing, RSA and CKA.
  Recorded in CLAUDE.md.
- `attribution.py`: targets (a) and (c). Gradients w.r.t. the input embeddings, fp16 with gradient checkpointing,
  eager attention (needed for the weights), and dropout switched off. The gradient matches finite differences
  (0.0276 vs 0.028–0.029).
  - **Defined here, please confirm:** the IG "mask" baseline is the all-zero embedding (these models have no mask
    token); "pad" is the pad-token embedding. The baseline replaces the prompt and working tokens; the chat template,
    "Answer: y =" and the target stay fixed.
  - **IG convergence:** Gauss–Legendre (Captum's default) did not converge: gaps of −11.6 to +6.3 nats even at 64 nodes,
    on paths where f(x) − f(baseline) ≈ 11–19 nats. The midpoint rule did: on one item, gap 28.3 → −0.8 → 0.01 at
    8 / 32 / 128 steps (pad), and 6.5 → −2.5 → 1.1 (zero). Defaults: 128 steps for (c) (short inputs, batched) and
    32 for (a) (long inputs). The gap is saved per item, so poorly converged items can be filtered out.
  - **Rollout:** about 99.9% of the mass ends up on the chat-template tokens (attention sink). The summary therefore
    renormalises over the prompt fields; values are stored to 6 significant digits.
- `attribution_summary.py`: per-field share and "density" (share / token share). Comparisons: distractor vs
  sufficient, 2nd vs 1st copy in `duplicate`, working share (a); by subtype, correct/wrong and trap.
- `attribution_kaggle.ipynb`: 2 per category × subtype of test (144 items) per model and target, on 2 GPUs. Tested
  locally with Qwen2.5-0.5B; not run on Kaggle yet.
- The test-generation Kaggle kernel stopped once mid-run. `open_eval_test_kaggle.ipynb` can now continue from an
  earlier run's output added as an Input.

## 2026-11-04: test-split results for the shortlist (792 problems, greedy, 1536 new tokens, 2 × T4)
- Runs: `results/open_eval/20261103T2*_test/` (as generated) and `*_test_regraded20261104*` (grader fixed, see below;
  these are the reported numbers). Revisions: Phi cfbefacb, Qwen3-4B cdbee75f, Qwen2.5-Math aafeb0fc; fp16,
  transformers 5.20.2. Every reply stores `gen_ids`. Logs are in `results/open_eval/_logs_test_20261103/`.
- **Accuracy [95% CI]: Qwen3-4B 89.5% [87.2, 91.5], Qwen2.5-Math-1.5B 87.4% [84.9, 89.5], Phi-4-mini 77.0% [74.0, 79.8].**
  With 1536 tokens instead of 768 everything rose a lot (dev at 768: 58 / 64 / 67%).
  | subtype | Qwen3-4B | Qwen2.5-Math | Phi |
  |---|---|---|---|
  | none | 95.5 | 90.9 | 70.2 |
  | duplicate | 95.5 | 84.9 | 83.8 |
  | rescaled | 97.0 | 90.9 | 81.3 |
  | unrelated | **70.2** | 82.8 | 72.7 |
  Order 1 / 2: 97.2 / 81.8, 93.2 / 81.6, 86.6 / 67.4. Hardest type everywhere: 2nd-order non-homogeneous
  (74 / 72 / 47%). Took 258 / 45 / 130 min.
- **Finding (Qwen3-4B, unrelated):** 71 of 198 `unrelated` replies run past 1536 tokens. In all 71 the model
  calls the z-condition "a typo" for a y-condition, often re-reads it as a different condition (e.g. `z(1) = -2` as
  `y'(1) = -2`), finds the system inconsistent and keeps deliberating; there's no looping, and none of these
  replies reached the true solution before the cut. On the other subtypes it is 95–97%. So the distractor doesn't
  pull its answer to the trap (0% trap answers); it derails the solving. This is a candidate case for the
  attribution and representation work.
- Trap answers (z read as y): Qwen3-4B 0%, Qwen2.5-Math 4.5%, Phi 4.5%.
- Phi is again weakest on the plain `none` problems (70.2% vs 81–84% with a duplicate/rescaled condition; the
  same was seen on dev). Of its 56 wrong finished `none` replies, 32 still solve the ODE, i.e. the constants are
  wrong. Open question: does a repeated condition help Phi?
- Wrong answers that still solve the ODE (constants wrong): Phi 111, Qwen2.5-Math 84, Qwen3-4B 5.
- **Grader fixes (regrade, originals kept):**
  1. "Answer: The solution ... is:" with the expression on a following line is now read from that line. This
     affected Qwen2.5-Math: +9 items, 86.2 → 87.4%.
  2. **Please confirm:** answers written in `t` instead of `x` (e.g. `y = -3 + 3e^{-2t}`) are now accepted and
     flagged (`status = ok_variable_t`): 3 Phi answers, 2 of them correct (76.8 → 77.0%).
  Parser tests 29/29, verifier 864/864.

## 2026-11-04: attribution results, test split (2 per category × subtype = 144 items per model and target)
- Runs: `results/attribution/20261104T*_{model}_{c,a}_test/` (Kaggle 2 × T4, fp16, eager attention, gradient
  checkpointing; IG midpoint 128 steps for (c), 32 for (a)). Figures and every plotted number:
  `figures/attribution/20261104_test/` (`attribution_plots.py`; `stats.json`). Took ~7.5 h in total.
- Usable: **Qwen3-4B and Phi-4-mini, both targets** ((c) 144/144; (a) 129 and 143, the rest truncated replies).
- **Qwen2.5-Math-1.5B: every value is NaN (both targets).** Eager attention overflows in fp16 for this model
  (generation used the sdpa kernel and was fine). Fixed: `attribution.py` now runs it in fp32 by default and stops
  after 3 non-finite examples. Rerun with `attribution_math_kaggle.ipynb` (c on GPU 0, a on GPU 1), not done yet.
- Statistics: median of per-item log2 ratios, bootstrap 95% CI. Means of ratios are heavy-tailed (one item with
  a near-zero denominator gives "66 ± 62"), so the means in `attribution_summary.py` should not be quoted.
  "Density" = a field's share of the attribution ÷ its share of the prompt tokens (1 = in proportion).
- **Distractor vs sufficient conditions (density ratio, all distractor items, median [95% CI]):**
  | | attention (c) | attention (a) | G×I (c) | G×I (a) |
  |---|---|---|---|---|
  | Qwen3-4B | 0.65 [0.59, 0.82] | 0.74 [0.63, 0.90] | 1.01 [0.62, 1.37] | 1.16 [0.76, 1.88] |
  | Phi-4-mini | 0.65 [0.61, 0.87] | 0.87 [0.78, 0.97] | 0.64 [0.52, 0.92] | 1.01 [0.71, 1.27] |
  Attention consistently looks less at the distractor than at the conditions that determine the answer, most
  clearly for `unrelated` (0.68 / 0.61 in (c)) and least for `rescaled` (0.75 / 0.86, CIs reach 1). The gradient
  methods mostly do **not** show a smaller effect of the distractor on the answer's log p (only Phi (c) G×I does).
  So "looked at less" and "matters less for the answer" disagree here.
- **Duplicate position effect:** the second (identical) copy gets about half the attention of the first:
  0.54 / 0.53 in (c), 0.55 / 0.75 in (a), rollout similar. G×I: Qwen3 ~0.9 (no clear effect), Phi 0.48 [0.32, 0.66]
  in (c). The models attend mainly to the first occurrence of a repeated condition.
- **Layers (attention):** all prompt fields are below proportion in the first ~5 layers. From the middle layers on,
  the equation gets the most, and in (c) sufficient conditions separate from the distractor from about layer 10
  (Qwen3) / layer 8 (Phi). This may be a useful pointer for which layers to probe.
- **Working share (target a):** methods disagree on size: G×I 18% (Qwen3) / 36% (Phi), IG (pad) 59 / 65%, attention
  75 / 79%, rollout 28 / 29%.
- **Correct vs wrong (Phi only; Qwen3 has 13 / 3 wrong):** wrong answers go with less attention on the equation
  and the sufficient conditions, e.g. (a) equation 0.63 → 0.40, sufficient 0.63 → 0.48; (c) equation 1.82 → 1.40,
  sufficient 1.27 → 1.02. This holds even in (c), where the target is the true solution, so it looks like a property
  of how Phi reads those problems, not of its wrong output.
- **IG did not converge in most runs (please decide):** median relative completeness gap
  pad 6% only for Phi (c); otherwise pad 19–31%, zero baseline 60% to 1160% (Qwen3 (c)). The all-zero ("mask")
  baseline is far from any real embedding and the path integral is rough there. Options: drop the zero baseline,
  or more steps / fp32 for IG. G×I and attention are unaffected. IG numbers are plotted hollow when the median gap is > 10%.
- **G×I saturation:** 13 of 129 Qwen3 (a) items have log p(answer) = 0.0 (the answer is already in the working),
  so the gradient is 0 and G×I gives nothing; they drop out of the G×I numbers. IG covers this case by design.
- **Rollout** gives the instruction (the first prompt tokens) 6–9× its length share in every run: the usual
  early-token bias of rollout. Read rollout as weak evidence.

## 2026-11-04: IG demoted (user decision); reading of the attribution results; next steps
- **IG: decision by the user.** IG is kept in the outputs for completeness only; no conclusions rest on it, and no
  extra GPU time is spent on it (no more steps, no fp32 rerun for IG). This replaces the open question in the
  previous entry. Conclusions use **Gradient×Input and attention** (rollout as weak evidence).
  - User's reasoning: as with LIME/SHAP/occlusion, in an ODE problem any change to the input is a different
    problem, so methods that evaluate the model away from the real input don't fit this task.
  - Suggested wording for the thesis (code side): IG was included as the path-based gradient method, but
    (1) its interpolation path passes through embeddings that are not valid problems (a weaker form of the
    perturbation objection); (2) there is no natural "empty" baseline for an ODE problem, and the two baselines
    disagree strongly (median gap pad ~20–30%, zero 60–1160%); (3) it did not converge within a practical step
    budget. Caution: (3) alone is a numerical issue (completeness holds in the limit of many steps), so the
    argument should rest on (1) and (2), with (3) as supporting evidence. G×I is a gradient at the real input
    (no path, no baseline) and does not share these problems.
- **Points worth taking forward (for the discussion side):**
  1. **Attention vs gradient disagree on the distractor** (attention 0.65×, G×I ~1.0×). Either the distractor is
     looked at less but still affects the output, or G×I is too noisy at field level. A probe that separates
     sufficient from distractor conditions in the hidden states (layers ≥ 8–10, where attention separates them)
     would decide this. A natural bridge from attribution to probing.
  2. **`rescaled` is the weakest distractor effect** (CIs reach 1): a rescaled condition looks like a valid
     condition. Hypothesis for probing: its representation is closer to a sufficient condition's than the
     `unrelated` z-condition's is.
  3. **Duplicate position effect differs by model:** both attend mainly to the first copy (~0.55×), but by G×I
     only Phi relies on it (0.48×); Qwen3 uses both copies about equally (~0.9×).
  4. **Phi's wrong answers** go with less attention on equation and sufficient conditions in both (a) and (c),
     i.e. already in how it reads the problem. Correct/wrong comparison rests on Phi only (Qwen3 has 13 / 3 wrong).
- **Next (code side):** Qwen2.5-Math rerun in fp32 (`attribution_math_kaggle.ipynb`, user runs it, ~3 h). After
  that: a single **attribution results notebook** (all three models; figures + statistics, for the thesis and the
  guide), then no more attribution compute; move to representation analysis (probing, RSA, CKA).

## 2026-11-06: Qwen2.5-Math attribution (fp32 rerun), test split; figures for all three models
- Run: `results/attribution/20261104T185912Z_qwen2.5-math-1.5b-instruct_{c,a}_test/` (Kaggle 2 × T4, **fp32**, eager
  attention, checkpointing on; IG midpoint 128 / 32 steps). (c) 144/144, (a) 143/144 (1 truncated reply). No
  non-finite values. Took 4.1 h (c) and 5.9 h (a) in parallel, longer than the ~3 h estimate (fp32). Logs in
  `results/attribution/_logs/20261104T185912Z_math/`. The first launch (c without checkpointing) ran out of
  memory on every item; fixed, and `attribution.py` now stops after 3 OOMs with no item done.
- Figures and numbers for **all three models**: `figures/attribution/20261106_test_all3/` (+ `stats.json`; keys
  fixed so `distr/suff <subtype>` and `dup 2nd/1st` no longer overwrite each other).
- **Distractor ÷ sufficient density (median [95% CI]), Qwen2.5-Math:** attention (c) 0.69 [0.60, 0.81], (a) 0.84
  [0.74, 0.93]; G×I (c) **0.78 [0.62, 0.95]**, (a) 0.93 [0.68, 1.20]. So the attention result holds for all three
  models (0.65–0.69 in (c)). By G×I in (c), Math and Phi do reduce the distractor's effect (CI below 1); Qwen3 does
  not (1.01). In (a) no model shows it by G×I. Strongest for `unrelated` (attention 0.70), weakest for `rescaled`
  (0.89, CI reaches 1), same as the other two.
- **Duplicate, second copy ÷ first:** attention 0.55 (c) / 0.62 (a); G×I (c) **0.46 [0.26, 0.72]**, (a) 0.77 (CI
  reaches 1). Same as Phi (G×I 0.48); Qwen3 is the odd one out (~0.9). All three attend mainly to the first copy.
- **Math reads the equation much more and the instruction much less** than the other two: (c) attention density
  equation 1.75, instruction 0.44 (Qwen3: 1.31 / 1.30); G×I equation 2.26 in (c). Plausibly its maths
  specialisation; a candidate contrast for the representation analysis.
- **Layers (attention, c):** sufficient ÷ distractor is above 1 from about layer 3–4 and grows in the upper half
  (1.5–2.7 at layers 15–24 of 28). Earlier than Qwen3 (~10) and Phi (~8), and larger.
- **Working share (a):** G×I 31%, attention 82%, rollout 49% (others: G×I 18 / 36%, attention 75 / 79%).
- **Correct vs wrong (18 wrong in c, 17 in a):** in (c) wrong answers go with less attention on the equation
  (1.78 → 1.22) and less G×I on the sufficient conditions (1.42 → 0.97), the same direction as Phi. In (a) G×I on
  the equation falls 0.92 → 0.38 (wide CI). Attention in (a) shows no difference. Small n, so suggestive only.
  Qwen3 has 13 / 3 wrong.
- **IG (completeness only, by the user's decision):** Math pad baseline converges in (c) (median gap 4%, 79% of items
  < 10%) and nearly in (a) (10.5%); zero baseline 120% / 440%. Not used for conclusions.
- **No G×I saturation** for Math (unlike 13 Qwen3 (a) items).
- **Attribution compute is now finished for all three models.** Next: the attribution results notebook (figures +
  statistics), then representation analysis.

## 2026-11-06: attribution results summary notebook; findings file
- **`ATTRIBUTION_FINDINGS.md`**: the consolidated attribution findings for all three models (setup, 7 findings
  with tables, IG appendix, caveats, open questions for the representation analysis). Reference for the thesis
  chapter.
- **`attribution_results_summary.ipynb`** (built by `make_attribution_summary_nb.py --execute`, runs locally on
  `results/attribution/`, no GPU; figures embedded): headline figure, then one section per finding with figure,
  table (median [95% CI]) and a plain-language reading; IG only in an appendix. Figures:
  `figures/attribution/summary_notebook/`.
- **User request: main figures show only Gradient×Input, attention and rollout.** `attribution_plots.py` now uses
  `MAIN = ('gxi', 'attn', 'rollout')` for figures 1–6; IG appears only in `7_ig_completeness.png` and the new
  `8_ig_baselines.png`.
- **New IG evidence (baseline dependence):** per item, the rank correlation between the pad- and zero-baseline
  field attributions has median 0.09–0.37 across runs, and the two baselines pick the same top field in 1–33% of
  items (chance ≈ 1 in 6). So the two baselines give different explanations of the same item.
- **Layer result, corrected:** with a per-item log ratio and a common relative-depth axis (new figure
  `4b_layer_separation_depth.png`), sufficient ÷ distractor attention in target (c) peaks at 74% (Qwen3, 2.3×),
  74% (Math, 3.3×) and 61% (Phi, 2.5×) depth, with separation at roughly 40–85% depth in all three. In target (a)
  it is flatter (~1.2–1.6×, no clear peak). The earlier claim that Math separates earlier was wrong.
- The attribution step is finished. Next: representation analysis (decisions on layers, pooling, probes, RSA/CKA
  comparisons are open; questions from attribution are listed at the end of `ATTRIBUTION_FINDINGS.md`).

## 2026-11-09: why G×I crosses 1× on the distractor while attention doesn't (diagnostic, no new runs)
Per-item checks on the saved attribution runs (distractor items only):
- **G×I is about twice as noisy per item:** IQR of log2(distractor ÷ sufficient) 2.0–2.8 for G×I vs 0.7–1.3 for
  attention, so its CIs are about twice as wide.
- **Cancellation inside a field:** |Σ G×I| / Σ|G×I| has median 0.30–0.37 for both sufficient and distractor
  conditions, so about two thirds of the token-level signal cancels. Using Σ|G×I| (already stored as `gxi[g][1]`)
  halves the spread (IQR 1.1–1.6) and narrows the CIs, but the medians barely move: Qwen3 (c) 1.02 [0.83, 1.21],
  Math (c) 0.71 [0.63, 0.85], Phi (c) 0.78 [0.70, 0.96].
- **G×I and attention disagree per item:** rank correlation between the two ratios is −0.02 (Qwen3 c) to 0.54
  (Math c).
- **Target (a) is saturated:** median log p of the model's own answer is −0.00 to −0.03 (it copies the answer from
  its working), vs −11 to −17 in (c). Prompt gradients are then tiny, and all three G×I ratios in (a) sit near 1.
- Reading: noise and saturation explain the (a) rows; Qwen3 (c) is a genuine null (still 1.0 with the less noisy
  variant). G×I measures local sensitivity of log p, attention measures where the answer tokens read; they need not
  agree. Possible robustness check for the discussion side: report Σ|G×I| next to |Σ G×I|.

## 2026-11-09: IMPORTANT correction: the "attention avoids the distractor" effect depends on its order
Diagnostic on the saved attribution runs (no new runs). Attention, `rescaled` + `unrelated` items, distractor ÷
sufficient density by the distractor's place among the conditions:
- **Distractor first:** ≈ 1.05–1.26 (it gets as much or more than the useful condition), all models, (c) and (a).
- **Distractor after a useful condition:** ≈ 0.45–0.75 (about half).
- **Control** (no distractor, two useful conditions, 2nd ÷ 1st): 1.02–1.10 in (c), so it is not a plain
  position effect.
- G×I agrees for Math and Phi in (c) (last 0.53 / 0.49 vs not last 1.02 / 0.93); Qwen3 G×I shows neither.
- Positions are roughly balanced in the test split (distractor last in about half of the items), so the
  headline 0.65× is an average of ≈1.1 and ≈0.5.
- **Reading:** the models discount a condition that is redundant or irrelevant *given what came before it*. With
  causal attention, a distractor written first is encoded before any useful condition is seen.
- **Consequences:** `ATTRIBUTION_FINDINGS.md` has a new section 1b. The summary notebook's headline ("attention
  avoids the distractor") needs an order split; not changed yet. **For the representation analysis:** use the
  distractor's position as a factor (probe accuracy for condition role, split by distractor first vs after).
- (same day) The order split is now in the pipeline: `attribution_plots.py` → `9_distractor_order.png`, and
  section 1b of `attribution_results_summary.ipynb` (re-executed; headline title reworded). Exact numbers in
  `ATTRIBUTION_FINDINGS.md` section 1b.

## 2026-11-09: possible follow-up for G×I reliability and wide intervals (proposal only, not run)
- Why the intervals are wide: per-item spread (σ of log2 ratio ≈ 2.0 for G×I, ≈ 0.9 for attention), small cells
  (36 per subtype), and mixtures (distractor first vs after; arbitrary distractor label in `duplicate`).
  SE(median) ≈ 1.25·σ/√n gives, for n = 36, a 95% CI of about ×/÷1.8 (G×I) and ×/÷1.3 (attention), matching the figures.
- Cheap fixes, no new runs: stratify by order (done, section 1b), Σ|G×I| alongside |Σ G×I|, and one regression on
  log2 ratio with factors (order, subtype, ode order, model) instead of separate medians per panel.
- Optional rerun (needs the user's OK; attribution compute was declared finished): **G×I + attention only, no IG, on
  all 792 test items**. IG was most of the cost (2 × 128 steps per item in (c), 2 × 32 in (a)); without it, an
  estimated 2–3 h on 2 × T4 for all three models and both targets. n per subtype 36 → 198, so CIs about 2.3× narrower.
  Variants that stay gradient-at-the-real-input: the logit of the answer tokens instead of log p for target (a)
  (avoids saturation at p ≈ 1), and fp32 for a subset to rule out fp16 gradient noise.

## 2026-11-09: layer-by-layer attention (figure 4) is reproducible, and shows a stage order
- Split-half reliability of the median layer profiles: r = 0.96–1.00 (all models, targets, fields). The zigzag
  is a property of individual layers, not item noise.
- Early layers (2–4): the instruction takes 4.3–8.3× its share in Qwen3 and Phi, in both targets; the instruction
  profile is the same in (c) and (a) (r = 0.85–0.98). Source of rollout's early-token bias.
- Target (c): the equation and the useful conditions peak in the middle-to-upper layers (equation 2.8–4.3×,
  sufficient 1.7–2.1×, distractor ≈ 0.7–0.9×, instruction ≈ 0.1× at the peak layer).
- Target (a): the equation peaks early (13–26% depth); upper layers read the answer instruction (≈ 2×).
- Figure 4 hides this because it doesn't draw instruction and answer instruction. Details in
  `ATTRIBUTION_FINDINGS.md` section 5. For probing: use all layers; expect role information in the middle-to-upper ones.

## 2026-11-09: clean attribution results notebook (user decisions: target (a) only, attention only)
- **Decisions (user):** G×I is demoted to the appendix like IG (gradient = response to input changes; a differential
  equation is extremely sensitive to any change). Target (c) is no longer reported: the thesis studies the models
  where they solve the problems, i.e. with their own working (target (a)). Attention is the method for every result;
  rollout only for total-flow questions, never for position comparisons. Recorded in `CLAUDE.md`.
- **Built:** `attribution_report.py` (all analyses + figures, CPU, ~10 s), `make_attribution_results_nb.py` →
  `attribution_results_kaggle.ipynb` (clean, for Kaggle) and `attribution_results_kaggle_executed.ipynb` (local run,
  figures embedded), Kaggle input `kaggle_attribution_results_files.zip` (code + the three target-(a) runs, 1.9 MB).
  Figures, CSV tables and stats: `figures/attribution/report_a/`.
- **New: position vs role, within-item regression** (log2 attention density of each condition ~ position + role,
  item fixed effects; items with ≥ 2 conditions: none second-order, rescaled, unrelated; 75 / 90 / 89 items; 1000
  bootstrap resamples of items). Qwen3 / Math / Phi:
  - position 2 vs 1: 1.06 [0.84, 1.33] / 1.14 [0.96, 1.37] / 1.09 [0.93, 1.28] (no effect);
  - distractor written first vs useful: 1.23 [0.94, 1.59] / 1.34 [1.07, 1.64] / 1.09 [0.90, 1.32];
  - distractor written after a useful one vs useful: **0.60 [0.47, 0.74] / 0.69 [0.57, 0.82] / 0.74 [0.63, 0.86]**.
- **Per layer:** the position effect is in the first ~5–15% of layers (later condition 0.51–0.62×); the
  distractor-after effect is below 1 from ~20–25% depth on (median past 25% depth 0.56 / 0.78 / 0.76; lowest 0.42 at
  54% depth, 0.33 at 74%, 0.62 at 42%). Layer profiles are reproducible (split-half r = 0.97–1.00).
- **Correct vs wrong (target a):** Phi (39 wrong): equation 0.63× [0.53, 0.78], useful conditions 0.76×
  [0.68, 0.89], instruction 1.37× [1.20, 1.47]. Math (17 wrong): no difference. Qwen3: 3 wrong, not analysed.

## 2026-11-09: folder cleanup (user request; nothing deleted)
- 33 unused items moved into `_archive/` (README inside): MCQ-era notebooks/scripts (`mcq_baseline/`), easy-control
  notebooks/scripts (`easy_controls/`), downloaded Kaggle result zips (`downloaded_result_zips/`, already unpacked
  into `results/`; kept because a few files differ), and the superseded attribution summary notebook and figures.
- Kaggle upload zips moved to `kaggle_inputs/` (README maps zip → notebook). `__pycache__` removed.
- `results/`, the datasets, the accuracy and attribution notebooks and their code are unchanged and checked: all
  kept `.py` compile, grader tests 29/29, `attribution_report.build()` produces all figures. Paths updated in
  `CLAUDE.md` (new "Folder layout" section) and `ATTRIBUTION_FINDINGS.md`.

## 2026-11-09: Kaggle run of attribution_results_kaggle.ipynb verified
- The user ran the notebook on Kaggle (CPU) and downloaded `attribution_report_a.zip` (9 figures, 11 CSV tables,
  stats.json; archived as `_archive/downloaded_result_zips/attribution_report_a_kaggle_20261109.zip`).
- Identical to the local run of the same code and data: all 1,089 numeric values in stats.json (largest relative
  difference 9e-16, floating-point rounding) and all 11 tables byte-identical. Figures render correctly on Kaggle.
- The reported attribution results are therefore those in `figures/attribution/report_a/` and the executed notebook.

## 2026-11-10: layer profile (figure 5) in absolute terms; probing questions from the user
- Median share of ALL attention from the answer tokens, per layer (target a): equation + conditions together get
  0–7% in every layer (mostly 1–3%). The first token ("attention sink", chat template) takes 10–83% depending on
  the layer and grows towards the top (Qwen3 55–71% in layers 25–35, Phi 58–83% in layers 21–28); the working
  10–67%; the problem text as a whole 0–31% (peaks: Qwen3 layers 2–6 and 17–20, Phi 1–3, Math 11 and 16).
- So figure 5 shows how a small slice is split within the problem text. Within-prompt densities average to 1
  (length-weighted), so a rise of the 46-token answer-format instruction forces the other fields down
  (e.g. Qwen3 layer 20: answer-format 2.0×, all others ≤ 1/16×).
- User questions for the representation analysis (need decisions on the discussion side):
  (1) what do the layers at the attention minima of equation/conditions encode; (2) do layers after the
  conditions' attention peak reach the same probe accuracy as the last/best layer. Proposed design: per-layer
  probes on (i) field states and (ii) the answer-position state, plotted with the layer-to-layer gain, compared
  with the attention curve; report the earliest layer within a set margin of the best layer; control tasks for
  selectivity. Extraction is one teacher-forced forward pass per item (all test items with a finished answer).

## 2026-11-10: user decision: link representation analysis to the attention results
- Every representation result (probing, RSA, CKA) is to be read against the attention results, especially the
  layer-by-layer attention figure (`figures/attribution/report_a/5_layers_fields.png`, `6_layers_position_role.png`;
  per-layer numbers in `figures/attribution/report_a/stats.json`). Same layer indexing; per-layer curves shown
  side by side; correlations tested and reported with CIs, including null results. Recorded in `CLAUDE.md`
  (representation analysis section). The discussion side should plan the probes/RSA/CKA with this in mind.
