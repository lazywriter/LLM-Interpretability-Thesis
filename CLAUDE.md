# LLM Interpretability Thesis — coding workspace

This folder is the **coding side** of the project. Methodology discussions and
decisions happen in a separate Claude chat (the "discussion side"). Follow the
decisions below; don't change methodology on your own. If a coding result
suggests a decision should change, write it up in `NOTES_FROM_CODE.md` and flag
it instead.

## Workflow between the two sides
- **Discussion side → here:** decisions land in this file (section "Current
  decisions") and in the task given to you.
- **Here → discussion side:** after each coding session, append a dated entry
  to `NOTES_FROM_CODE.md`: what was built or run, key numbers, anything
  surprising, open questions. Keep it short and factual. The discussion side
  reads this file.
- Don't edit `ivp_mcq/` data files by hand. Regenerate with the generator.

## Project in one paragraph
A thesis on **LLM interpretability** using ODE / initial value problems as the
task domain: symbolic, short inputs, with SymPy as a reliable correctness oracle.
Models are **small open-source LLMs (~1.5–4B)**: the shortlist is
**Phi-4-mini-instruct, Qwen2.5-Math-1.5B-Instruct and Qwen3-4B-Instruct-2507**,
chosen on open-ended accuracy (see "Models" below). They get **open-ended problems and
may show their working** (no options), and are evaluated **zero-shot only (no
fine-tuning)**. Model runs are on **Colab free tier or Kaggle (T4, 16 GB)**, so
keep code lightweight and notebook-friendly (notebook-ready scripts, small
batches, fp16). The research question is **how the models solve the problems
and where they go wrong**. It is studied first with a **short attribution
step** (gradient and attention methods only), then, **where most of the time
goes, with representation analysis** (probing, RSA, CKA). Answer accuracy is
context and a label, not the goal.

## The working dataset: `ivp_open/` (see `ivp_open/README.md`)
864 open-ended IVPs (no options), built 2026-11-03 by `gen_ivp_open_dataset.py`
(seed 23) with the IVP-MCQ construction. **Dev 72** (model shortlist and prompt
tuning), **test 792**.
- 18 categories × 4 subtypes × 12: `none` (plain IVP, control), `duplicate`,
  `rescaled`, `unrelated` (z-condition; `info.trap_solution`). Variant C is
  not included.
- Same per-condition roles and `spans` as IVP-MCQ (minus options).
- Tools: `ivp_open/grade_open.py` (answer extraction, plain-text and LaTeX
  parsing, SymPy grading, what the answer satisfies, per-slice report) and
  `ivp_open/check_ivp_open.py` (independent verifier, all 864 pass).
- Model evaluation: `eval_open_ivp.py` (batched greedy generation, writes
  `results/open_eval/`). Kaggle: `open_eval_kaggle.ipynb` +
  `kaggle_inputs/kaggle_open_eval_files.zip`.

## The original dataset: `ivp_mcq/` (see `ivp_mcq/README.md` for full detail)
864 over-specified IVPs as 3-option MCQs (chance 33.3%). Dev 72 is for prompt
tuning only; **test 792 is for reported results**.
- 6 ode_types × 3 categories, 144 per type.
- **Variant B (pure distractor):** a solvable IVP plus one uninformative
  condition (`duplicate`, `rescaled`, `unrelated` z-condition with a **trap**
  option, `info.trap_letter`).
- **Variant C (inconsistent):** sufficient conditions plus 1–2 contradicting
  ones. Exactly one option satisfies a sufficient subset.
- Each item has `prompt`, `answer`, per-condition `role`
  (sufficient / contradicting / distractor), per-option `role` and
  `satisfies_conditions`, and **`spans`** (character offsets of instruction,
  equation, each condition, each option, answer instruction).
- Tools: `grade_mcq.py` (`load`, `extract_letter`, `letter_from_scores`,
  `evaluate`), `check_ivp_mcq.py` (independent verifier, all 864 pass),
  `audit_shortcuts.py` (no-maths heuristics, all at chance),
  `gen_ivp_mcq_dataset.py` (seed 11, ~5 min).

## Folder layout (cleaned up 2026-11-09)
- **Datasets:** `ivp_open/` (working dataset), `ivp_mcq/` (original MCQ dataset; its generator is imported by
  `gen_ivp_open_dataset.py`), `easy_mcq/`, `easy_open/` (easy-arithmetic controls).
- **Accuracy:** `eval_open_ivp.py`, `open_eval_kaggle.ipynb` (dev), `open_eval_test_kaggle.ipynb` (test, built by
  `make_testrun_nb.py`). Model wiring: `baseline_letter_logprobs.py` (imported by everything; keep it).
- **Attribution:** `attribution.py`, `field_tokens.py`, `attribution_summary.py`, `attribution_plots.py`;
  `attribution_kaggle.ipynb`, `attribution_math_kaggle.ipynb` (built by `make_attribution_nb.py`); results notebook
  `attribution_results_kaggle.ipynb` (+ `_executed` copy; built by `make_attribution_results_nb.py`, analysis in
  `attribution_report.py`); findings `ATTRIBUTION_FINDINGS.md`.
- **Dataset overview:** `dataset_overview_kaggle.ipynb` (+ `_executed`), built by `make_dataset_nb.py`;
  `gold_check.py` is imported by `ivp_open/check_ivp_open.py`.
- **Kaggle uploads:** `kaggle_inputs/` (README maps each zip to its notebook).
- **Outputs:** `results/` (never overwrite), `figures/` (current: `figures/attribution/report_a/`, `figures/dataset/`).
- **`_archive/`:** MCQ-era and easy-control notebooks/scripts, downloaded result zips, superseded attribution
  notebook and figures (README inside). Nothing was deleted.

## Current decisions (from the discussion side and the user)
- **Answer format: FINAL (decided 2026-11-03, user).** Open-ended problems
  with **working allowed**, for higher accuracy:
  - no options; the model solves step by step and ends with a fixed final line
    (`Answer: y = ...`);
  - graded by SymPy equivalence with the true solution.
  - Evidence: on the easy-arithmetic control, MCQ letter scoring gave 70%
    (Qwen) / 62% (Phi), open number-only 92% / 80%, and open with working
    100% / 100%. The MCQ failure was in matching the computed value to a
    letter, not in the arithmetic.
  - **MCQ letter scoring is retired.** Its results stay on record
    (`results/CANONICAL.json`, IVP test: Qwen 43.4%, Phi 45.7%) and
    `baseline_letter_logprobs.py` stays in the repo, but don't build new work on
    the letter format.
  - **Variant C is dropped for now (decided 2026-11-03, user).** Without
    options it has no single correct answer, because different sufficient
    subsets can give different valid solutions. Work with variant B only
    (432 items: dev 36, test 396). An alternative to C will be designed later.
    Keep the C items in `ivp_mcq/` untouched; just filter `variant == 'B'`.
  - Open-format prompts must be produced by a generator from the dataset
    (no hand edits).
- **Models: FINAL shortlist (decided 2026-11-03, user), from the IVP-OPEN dev
  run** (`results/open_eval/*_regraded*`, 72 problems, greedy, 768 new tokens):
  | model key | HF name | dev acc | accuracy on replies that finished | truncated |
  |---|---|---|---|---|
  | `phi-4-mini-instruct` | microsoft/Phi-4-mini-instruct | 66.7% | 86% | 16/72 |
  | `qwen2.5-math-1.5b-instruct` | Qwen/Qwen2.5-Math-1.5B-Instruct | 63.9% | 87% | 20/72 |
  | `qwen3-4b-instruct-2507` | Qwen/Qwen3-4B-Instruct-2507 | 58.3% | 98% | 32/72 |
  - **Test results (2026-11-04, 792 problems, greedy, 1536 new tokens; reported numbers):**
    `results/open_eval/*_test_regraded20261104*`
    | model | test acc [95% CI] | none / dup / resc / unrel | order 1 / 2 | cut off | acc on finished |
    |---|---|---|---|---|---|
    | Qwen3-4B-Instruct-2507 | 89.5% [87.2, 91.5] | 95.5 / 95.5 / 97.0 / 70.2 | 97.2 / 81.8 | 93 | 98.4% |
    | Qwen2.5-Math-1.5B-Instruct | 87.4% [84.9, 89.5] | 90.9 / 84.9 / 90.9 / 82.8 | 93.2 / 81.6 | 4 | 87.8% |
    | Phi-4-mini-instruct | 77.0% [74.0, 79.8] | 70.2 / 83.8 / 81.3 / 72.7 | 86.6 / 67.4 | 9 | 77.9% |
    - The generations (with `gen_ids`) are the input for attribution target (a)
      and for the correct/wrong labels in the representation analysis.
  - Use the **-Instruct** version of Qwen2.5-Math-1.5B (that is what was evaluated).
  - Together they cover two families (Phi, Qwen), a maths-specialised model and
    a range of sizes (1.5B, 3.8B, 4B).
  - Not chosen: Qwen3-1.7B (69.4%, tied with the top models), Qwen2.5-3B-Instruct
    (59.7%), Qwen2.5-1.5B (37.5%), Llama-3.2-3B (33.3%) and Qwen2.5-0.5B (16.7%).
    The earlier MCQ-based shortlist (Qwen2.5-3B, Phi-4-mini) is superseded.
  - Caveat: with a limit of 768 new tokens, many replies are cut off before the
    `Answer:` line, mostly on second-order problems. This affects Qwen3-4B most.
    Runs with the shortlist should use a larger limit (≥1536) so that
    correct/wrong labels aren't caused by truncation.
  - All models are wired up in `baseline_letter_logprobs.py` (`MODELS`,
    `TEMPLATE_KWARGS`, `LOAD_KWARGS`); `eval_open_ivp.py` runs them.
- **Research focus (decided 2026-11-03, user):** how the models solve the
  problems and where they go wrong.
  - **Order: attribution-based techniques first, then representation analysis
    (probing, RSA, CKA).**
  - **Effort (decided 2026-11-04, user):** attribution is kept short and
    limited. **Most of the time goes to probing, RSA and CKA.**
  - Whether the final answer is correct is context and a label, not the goal.
    Don't over-invest in answer-format mechanics.
- **Attribution (first analysis step, kept short):** **only** Gradient×Input,
  Integrated Gradients (report both a pad and a mask baseline), and attention
  (raw attention plus attention rollout).
  - **No perturbation- or subset-based methods (decided 2026-11-04, user):**
    no LIME, SHAP, occlusion or similar. These methods remove or change parts
    of the input around a local point. In a differential equation even a small
    change gives a completely different problem, so it is unfair to expect the
    model to still give the "correct" answer, and the resulting scores are not
    meaningful here. The gradient and attention methods keep the input as it is.
  - **IG demoted to completeness only (decided 2026-11-04, user):** IG is still
    computed and reported, but no conclusions rest on it and no extra compute
    goes into it (no more steps, no fp32 for IG). Reasons: its path runs through
    embeddings that are not valid problems, there is no natural baseline for an
    ODE problem (pad and zero disagree strongly), and it did not converge
    (median completeness gap 19–31% pad, 60–1160% zero).
  - **Reporting: attention only, target (a) only (decided 2026-11-09, user).**
    - **Gradient×Input is demoted like IG** (appendix only): gradient methods
      measure the response to (infinitesimal) changes of the input, and a
      differential equation is extremely sensitive to any change, so the same
      reasoning that excludes LIME/SHAP/occlusion applies.
    - **Target (c) is not reported.** The models solve the problems better with
      working, and the thesis studies them in that setting, so all attribution
      results use target (a) (own answer after own working).
    - **Attention** is the method for every result. **Rollout** only where the
      question is total flow through the layers (where the attention goes, share
      on the working); never for position comparisons (early-token bias).
    - Results notebook: `attribution_results_kaggle.ipynb` (built by
      `make_attribution_results_nb.py`, analysis in `attribution_report.py`,
      Kaggle input `kaggle_inputs/kaggle_attribution_results_files.zip`; runs on CPU).
      Supersedes `attribution_results_summary.ipynb` (now in `_archive/`).
  - Implementation: `attribution.py` (own code; Captum is not needed).
  - IG baselines: **pad** = the pad-token embedding; **mask** = the all-zero
    embedding (these models have no mask token). The baseline replaces the
    prompt and working tokens; the chat template and the target tokens stay
    fixed.
  - **Unit:** field level. Map `spans` to tokens with the tokenizer's
    `offset_mapping` and aggregate token attributions per field.
  - **Target: DECIDED 2026-11-03 (user): (a) as the main target, (c) as a
    control.** Attributions go to the prompt fields only.
    - **(a) main:** log p of the model's own final answer, i.e. the expression
      tokens of its `Answer: y = ...` line. It is teacher-forced on the model's
      own greedy working: input = chat prompt + the model's reply up to the
      answer. This explains what the model actually did, wrong answers
      included, so correct vs wrong and trap vs non-trap can be compared.
      The working tokens are part of the input but are not a field; report how
      much attribution they take, separately.
    - **(c) control:** log p of the true solution's tokens
      (`Answer: y = <answer_text>`) right after the chat prompt, with no
      working. Clean and comparable across models and items.
    - (b), attribution per generation step: not now; maybe later as a case
      study on a few items.
  - **Constraints:** no coefficient-substitution or counterfactual
    perturbations, no correctness-on-perturbed-input importance scores, and no
    perturbation attribution at all (see above).
  - **Questions to answer:** does attribution go to the conditions that
    determine the solution? Does it stay away from the distractor condition
    (`duplicate`, `rescaled`, `unrelated` z-condition)? Note: in `duplicate`
    items the two conditions are identical text, so any difference between
    them is a position effect.
- **Representation analysis (second step; details to be decided):** hidden states per field
  (instruction, equation, each condition), pooled over the
  field's tokens via `spans` → token offsets (`offset_mapping`, shifted by the
  prompt's offset inside the chat template). Natural labels: condition role
  (sufficient / distractor; contradicting only if C comes back), ode_type, subtype,
  solved or not, and whether the behaviour is content-driven (the 6-order
  test). Still open: layers, pooling, probe types, and which RSA/CKA
  comparisons (layers, models, conditions).
  - **Link to the attention results (decided 2026-11-10, user):** every
    representation result is read against the attention results, above all the
    layer-by-layer attention figure (`figures/attribution/report_a/5_layers_fields.png`,
    `6_layers_position_role.png`; per-layer values in `figures/attribution/report_a/stats.json`,
    keys `layers | <model> | <field>` and `layers regression | <key> | <model>`).
    Use the same layer indexing, put the per-layer probe / RSA / CKA curves next to
    the attention curves, and test whether they correlate (e.g. layer-to-layer
    probe gain vs attention to the equation / conditions; separability of the
    distractor vs the "distractor written after a useful condition" attention
    effect). Report the correlations with CIs, including when there is none.
- **Reporting:** keep per-slice reports (ode_type, category, subtype, correct
  vs. wrong) for any accuracy run. For B:unrelated, also report how often the
  answer satisfies the z-condition read as y (the old "trap").

## Next coding tasks (plan for 2026-11-04, set by the user)
1. **Re-evaluate every model's accuracy on the new open-ended dataset, with
   working allowed.**
   - DONE 2026-11-03: `ivp_open/` (864 items, user asked for a larger set,
     with a `none` control subtype added), the grader (27 parser cases pass),
     the verifier (864/864) and `eval_open_ivp.py` + `open_eval_kaggle.ipynb`.
     DONE 2026-11-03: dev run for 8 models on Kaggle (see "Models").
   - Candidates: Qwen2.5-0.5B/1.5B/3B-Instruct, Qwen2.5-Math-1.5B(-Instruct),
     Qwen3-1.7B, Qwen3-4B-Instruct-2507, Phi-4-mini-instruct and
     Llama-3.2-3B-Instruct (optionally Gemma). Qwen3 may run with thinking
     mode on, since working is allowed.
   - Budget: free-text working is long (up to ~512 tokens). Use **batched
     generation** to fit Kaggle/Colab time. Do a first pass on dev B (36),
     then the full B set for the strongest models.
2. **Shortlist models on the basis of that accuracy.** DONE 2026-11-03:
   Phi-4-mini-instruct, Qwen2.5-Math-1.5B-Instruct, Qwen3-4B-Instruct-2507.
3. **Attribution, kept short** (only Gradient×Input, IG, attention / rollout;
   no LIME/SHAP/occlusion) on the shortlist. Target decided: (a) main,
   (c) control (see "Attribution").
   - DONE 2026-11-03: span → token utility `field_tokens.py` (each token goes
     to the field it overlaps most, counting non-whitespace characters only;
     self-check 864/864 with the Qwen2.5 tokenizer, re-checked per shortlisted
     tokenizer on Kaggle).
   - DONE 2026-11-04: test-split generations for the shortlist (`open_eval_test_kaggle.ipynb`
     + `kaggle_inputs/kaggle_open_eval_test_files.zip`, 1536 new tokens, 2 GPUs in parallel).
     `generations.jsonl` now stores `gen_ids` (the exact generated tokens) for
     teacher forcing in target (a).
   - DONE 2026-11-04: attribution harness `attribution.py` + `attribution_summary.py`
     + `attribution_kaggle.ipynb` (`kaggle_inputs/kaggle_attribution_files.zip`). Default: 2 items per
     category × subtype of test (144 per model and target). IG uses the midpoint rule
     (Gauss–Legendre did not converge); 128 steps for (c), 32 for (a). The completeness
     gap is saved per item. Needs the test generations as a second input.
   - DONE 2026-11-04: attribution run on test for Qwen3-4B and Phi (both targets). Results and figures:
     `results/attribution/20261104T*`, `_archive/superseded_attribution/figures/20261104_test/` (`attribution_plots.py`: medians +
     bootstrap CIs, figures for every future run). Summary in `NOTES_FROM_CODE.md`.
     DONE 2026-11-06: Qwen2.5-Math rerun in fp32 (`results/attribution/20261104T185912Z_*`); figures for all
     three models in `_archive/superseded_attribution/figures/20261106_test_all3/`. Attribution compute is finished.
     DONE 2026-11-06 (superseded, in `_archive/`): attribution results notebook `attribution_results_summary.ipynb`
     (`make_attribution_summary_nb.py --execute`, local, no GPU) and the findings file
     `ATTRIBUTION_FINDINGS.md`. Main figures show G×I, attention and rollout only; IG only in an appendix.
     The attribution step is finished.
     DONE 2026-11-09: clean results notebook `attribution_results_kaggle.ipynb` (target (a), attention only;
     G×I and IG in the appendix), with a within-item regression separating position from role.
4. **Then representation analysis, the main part of the thesis:** hidden-state
   extraction per layer and field, then probing / RSA / CKA once the details
   are decided.

Done on 2026-11-03: easy-arithmetic controls (MCQ and open-answer); the
results are in `NOTES_FROM_CODE.md` and `results/easy_control/`,
`results/easy_open/`.

## Conventions
- Python 3, SymPy for anything symbolic. Plain-text maths notation
  (`y'`, `y''`, `^` for powers, `exp(1)` not `E`).
- Fix seeds and record model name, revision, dtype and prompt version with
  every result file.
- Results go in `results/<experiment>/` as JSONL + a short summary JSON. Never
  overwrite earlier results; use new run folders.
