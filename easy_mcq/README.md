# easy_mcq: easy-arithmetic control set

A sanity check for the letter-score pipeline: the same prompt layout as IVP-MCQ
(instruction, the problem, `Options:` A/B/C, "Answer with the letter of the
correct option only."), but the maths is trivial, so accuracy should be very high.

- 144 items, one split (`test`), 3 options, exactly one correct
- 6 categories × 24: `add`, `sub`, `mul`, `div` (exact), `pow` (small powers), `mixed` (`a + b * c`)
- answer letter 8/8/8 and rank of the correct value (smallest / middle / largest)
  8/8/8 inside every category; every no-maths heuristic scores 33.3%
- distractors: near values (±1 … ±10) and typical slips (`b*e` for `b ** e`,
  `(a + b) * c` for `a + b * c`, …), all positive and distinct
- every item verified with Python's own arithmetic
- same fields the IVP pipeline reads (`variant` = `E`, `ode_type` = `category` = operation,
  `order` = number of operations, `spans` for the options)

Regenerate (do not edit by hand): `python gen_easy_mcq.py` (seed 7).
Run: `baseline_letter_logprobs.py --data easy_mcq/easy_mcq_test.jsonl`, the same flag on
`option_permutations.py`, or `easy_control_colab.ipynb`.
