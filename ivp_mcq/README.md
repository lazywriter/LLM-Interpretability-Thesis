# IVP-MCQ: over-specified initial value problems as multiple choice

A balanced, SymPy-verified multiple-choice dataset of **initial value problems
with more conditions than needed**. It tests, zero-shot, whether small
open-source LLMs **use the right conditions** and **ignore the ones that carry
no usable information**. Because the relevant and irrelevant inputs are known
by construction, the dataset also gives **ground truth for attribution
methods** (Gradient×Input, Integrated Gradients, attention).

- **864 items**: 432 variant B + 432 variant C, 3 options each (chance = 33.3%)
- **6 ODE types × 3 categories = 18 families**, 48 items per family
- **Splits:** `dev` 72 (2 per family × variant, for prompt tuning only),
  `test` 792 (report results here)
- Every item passes an **independent verifier** that works from the rendered
  text alone (864/864)
- **Shortcut-audited:** heuristics that never do the maths score at chance

## The two variants

### B: pure distractor (a solvable IVP + 1 condition with no new information)

| subtype | what is added | example |
|---|---|---|
| `duplicate` | one condition restated verbatim | `y(0) = 3` … `y(0) = 3` |
| `rescaled` | one condition restated as `k*y(x0) = k*v` | `-2*y(1) = 6 - 8*exp(2)` |
| `unrelated` | a condition on a different function `z` | `z'(1) = -7 + exp(1)` |

`unrelated` items contain a **trap option**: the solution you get if you wrongly
read `z` as `y`. The trap-pick rate is a direct behavioural measure of
"the model used the distractor". Its letter is in `info.trap_letter`.

### C: inconsistent conditions (the conditions contradict each other)

`n` sufficient conditions plus 1 or 2 extra conditions that contradict them.
Exactly one option satisfies a subset of the conditions large enough to
determine the solution. The model is told the conditions are contradictory.

| subtype | extra conditions | conflict |
|---|---|---|
| `extra1_direct` / `extra2_direct` | 1 / 2 | one extra sits at the same point and derivative order as a sufficient condition, with a different value (`y(2) = -2` vs `y(2) = -5/2`) |
| `extra1_indirect` / `extra2_indirect` | 1 / 2 | extras at other points / derivative orders |

Example (`variant C`, `extra2_indirect`, answer **B**):

```
Differential equation: y'' = 12*sin(2*x)
Conditions:
1. y'(pi) = -4
2. y'(0) = -9
3. y(pi/2) = 5 - pi/2
4. y'(pi) = -7

Options:
A) y = -5*x - 3*sin(2*x) + 3      <- satisfies no condition
B) y = -x - 3*sin(2*x) + 5        <- satisfies 3 and 4 (sufficient: correct)
C) y = 2*x - 3*sin(2*x) + 2       <- satisfies only 1 (near miss)
```

## How distractor options are built

All options are the **correct general solution with different small integer
constants**. So no option can be ruled out by its form (no template shortcut)
or picked because it "looks nicer".
- **near miss** (2nd order): satisfies one sufficient condition but not enough
  to be determined. A model that checks only one condition picks it.
- **extra match** (C, 2nd order): satisfies one of the contradicting conditions.
- **far miss**: constants often reuse numbers visible in the conditions, which
  produces natural slips like reading `y'(0)` as `y(0)` or ignoring the
  particular solution.

## ODE coverage (balanced: 144 items per type, 48 per category)

| ode_type | categories (example) |
|---|---|
| first_order_direct | `direct_power` `y' = 3*x^2` · `direct_trig` `y' = 6*cos(2*x)` · `direct_exp` `y' = 8*exp(2*x)` |
| first_order_separable | `separable_exponential` `y' = -3*y` · `separable_xy` `y' = 4*x*y` · `separable_y_over_x` `y' = -y/x` |
| first_order_linear | `linear_const_forcing` `y' + 2*y = 6` · `linear_poly_forcing` `y' - 3*y = 2*x` · `linear_exp_forcing` `y' + y = 3*exp(2*x)` |
| second_order_direct | `direct2_power` `y'' = 6*x` · `direct2_trig` `y'' = 12*sin(2*x)` · `direct2_exp` `y'' = exp(x)` |
| second_order_homogeneous | `hom_real_distinct` `y'' - y' - 2*y = 0` · `hom_repeated` `y'' - 4*y' + 4*y = 0` · `hom_complex` `y'' - 4*y' + 8*y = 0` |
| second_order_nonhomogeneous | same three root types with constant forcing, e.g. `y'' + 3*y' + 2*y = 4` |

Second-order roots are integers (or `a ± bi` with integer `a, b`), so solutions
stay readable. Families that small models are unlikely to manage (Bernoulli,
exact, variable-coefficient, resonant forcing) are deliberately excluded.

Condition points are `0, 1, -1` (default), `0, pi/2, pi` (trigonometric and
complex-root families) and `1, 2, 3` (`y' = k*y/x`, which is undefined at 0).
Both value `y(x0)` and derivative `y'(x0)` conditions are used.

## Balance

| | counts |
|---|---|
| variant | B 432 · C 432 |
| B subtype | duplicate 144 · rescaled 144 · unrelated 144 |
| C subtype | 108 each |
| answer letter | A 288 · B 288 · C 288, and **exactly 24/24/24 inside every (variant, ode_type) stratum** |
| B distractor position | uniform within order (1st order: 108/108 · 2nd order: 72/72/72) |
| conditions per item | 2: 324 · 3: 432 · 4: 108 |

Full counts are in `distribution.json`. The dev split is small and its letters
are slightly uneven (A 29 / B 25 / C 18); use it only for prompt tuning.

## Verification (`check_ivp_mcq.py`)

The verifier ignores the generator's internals. From each item's **text** it:
1. solves the ODE with `sympy.dsolve`, and checks every option with `checkodesol`
2. solves every size-`n` subset of the y-conditions for the constants; an
   option is *achievable* iff it equals the unique solution of some subset
3. requires **exactly one achievable option, equal to the labelled answer**
4. C: requires the full set of conditions to be **jointly unsatisfiable**
5. B: requires the answer to satisfy every y-condition, and the trap to become
   achievable when `z` is misread as `y`

Result: **864/864 verified**.

## Shortcut audit (`audit_shortcuts.py`)

Accuracy of heuristics that never look at the maths (ties split evenly):

| heuristic | all | B | C | order 1 | order 2 |
|---|---|---|---|---|---|
| option sharing most constants with the others | 34.5 | 35.0 | 34.0 | 33.3 | 35.7 |
| option whose numbers appear in the conditions | 33.1 | 32.5 | 33.7 | 34.0 | 32.2 |
| shortest option | 33.1 | 33.6 | 32.5 | 35.0 | 31.2 |
| longest option | 32.9 | 32.1 | 33.6 | 32.6 | 33.1 |
| always A | 33.3 | 33.3 | 33.3 | 33.3 | 33.3 |

Before it was fixed, the "numbers appear in the conditions" heuristic scored
42.7%. The generator now keeps both content heuristics at chance within every
(variant, ode_type) stratum.

## Schema (one JSON object per line)

```jsonc
{
  "id": "ivp-0123", "split": "test",
  "variant": "C", "subtype": "extra2_indirect",
  "ode_type": "second_order_direct", "category": "direct2_trig",
  "order": 2, "n_required": 2, "n_conditions": 4,
  "equation": "y'' = 12*sin(2*x)",
  "conditions": [ {"text": "y'(pi) = -4", "role": "contradicting",   // sufficient | contradicting | distractor
                   "function": "y", "derivative_order": 1, "point": "pi",
                   "value_complexity": 0}, ... ],
  "options": [ {"letter": "A", "text": "y = ...", "role": "distractor",  // correct | distractor | trap
                "constants": [-5, 3], "satisfies_conditions": []}, ... ],
  "answer": "B", "answer_text": "y = -x - 3*sin(2*x) + 5",
  "prompt": "...full zero-shot prompt...",
  "spans": {"instruction": [s, e], "equation": [s, e],
            "conditions": [[s, e], ...], "options": [[s, e], ...],
            "answer_instruction": [s, e]},
  "info": {"n_extra": 2, "conflict": "indirect",
           "sufficient_positions": [3, 4], "contradicting_positions": [1, 2]},
           // B items: distractor_kind, distractor_position, trap_letter, ...
  "sympy_ode": "...srepr...", "sympy_general_solution": "...srepr..."
}
```

`spans` are character offsets into `prompt` for every field. Map them to
token offsets with the tokenizer's `offset_mapping` to get **field-level
attribution** (equation vs. each condition vs. each option vs. boilerplate)
without hand-labelling.

## Usage

```python
from grade_mcq import load, extract_letter, letter_from_scores, evaluate

test = load("ivp_mcq_test.jsonl")
prompt = test[0]["prompt"]                      # feed to the model, zero-shot

# recommended for small models: compare next-token log-probs of "A","B","C"
pred = letter_from_scores({"A": -2.1, "B": -0.4, "C": -3.0})

# or parse a free-text generation
pred = extract_letter("B) because ...")         # -> "B"; ambiguous -> None

acc, report = evaluate(test, list_of_letters)    # per variant / ode_type /
                                                 # category / subtype, plus the
                                                 # B:unrelated trap-pick rate
```

## Known caveats

- **Value complexity.** Conditions at `x0 != 0` often have values like
  `-3 + 4*exp(2)`. A model may ignore them because they are harder to use, not
  because it judged them irrelevant. Each condition records `value_complexity`
  (SymPy operation count) so you can control for this.
- **Textbook bias.** In C, the sufficient subset uses only `x0 = 0` in 59 of
  432 items. Most correct subsets are not the textbook `y(0), y'(0)` pair, so
  "always use `x = 0`" is not a winning strategy.
- **Variant C tells the model** that the conditions conflict. Without that,
  the task is ill-posed.
- **The MCQ format turns solving into verification:** a model can substitute
  conditions into options. That suits the question "which conditions does the
  model check?", but results should be framed that way.

## Reproduce

```bash
pip install sympy
python gen_ivp_mcq_dataset.py --outdir ivp_mcq       # seed 11, ~5 min
python check_ivp_mcq.py ivp_mcq/ivp_mcq_all.jsonl --outdir ivp_mcq   # ~1.5 min
python audit_shortcuts.py ivp_mcq/ivp_mcq_all.jsonl
```
