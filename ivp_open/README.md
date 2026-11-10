# IVP-OPEN: open-ended initial value problems (no options)

The format decided on 2026-11-03: the model solves the problem, may show its
working, and ends with `Answer: y = <expression>`. SymPy grades the answer.
This set replaces the multiple-choice format for model selection and for the
attribution / representation work. Built with the same exact construction as
IVP-MCQ (`ivp_mcq/gen_ivp_mcq_dataset.py`), so every label is exact.

- **864 items**: 18 categories (the 6 ODE types × 3 of IVP-MCQ) × 4 subtypes × 12
- **Splits**: `dev` 72 (one per category × subtype; for prompt tuning and the
  model shortlist), `test` 792
- **Subtypes** (216 each):

| subtype | conditions | what is added |
|---|---|---|
| `none` | exactly `order` | nothing: a plain IVP (control, and the easiest problems) |
| `duplicate` | order + 1 | one condition restated verbatim |
| `rescaled` | order + 1 | one condition restated as `k*y(x0) = k*v` |
| `unrelated` | order + 1 | a condition on a different function `z`; `info.trap_solution` is what you get by reading `z` as `y` |

- The position of the extra condition is balanced inside every category × subtype.
- Variant C (contradicting conditions) is not included: without options it
  has no unique answer (decision of 2026-11-03).

## Example

```
Solve the initial value problem below.

Differential equation: y'' = -3*exp(x)
Conditions:
1. z(0) = 0
2. y(0) = -2
3. y'(0) = 2

Show your working, then finish with a final line of the form "Answer: y = <expression>",
writing the expression in plain text (for example: y = 3*exp(2*x) - x^2).
```

## Fields

`id`, `split`, `subtype`, `ode_type`, `category`, `order`, `n_conditions`,
`equation`, `conditions` (text, role sufficient/distractor, function y/z,
derivative order, point, value_complexity), `answer_text`, `constants`,
`prompt`, `prompt_version` (`open-v1`), `spans` (instruction, equation, each
condition, answer instruction), `info` (distractor_kind, distractor_position,
restates_condition / mirrors_condition, trap_solution), and SymPy `srepr`s of
the ODE, the solution and the general solution.

## Tools

- `grade_open.py`: `extract_answer`, `parse_answer` (plain text and LaTeX:
  `\frac`, `\boxed`, `e^{...}`, `\cos x`, ...; answers with free constants are
  rejected), `grade_reply` (correct? solves the ODE? which conditions does the
  answer satisfy? equals the z-as-y trap?), `evaluate` (per-slice report).
- `check_ivp_open.py`: independent verifier working from the text only. All
  864 pass, including "the grader accepts every true answer and rejects every
  trap".

## Reproduce

```bash
python gen_ivp_open_dataset.py                                   # seed 23, ~2 min
python ivp_open/check_ivp_open.py ivp_open/ivp_open_all.jsonl    # ~3 min
```
