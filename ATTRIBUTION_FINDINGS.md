# Attribution findings (test split, all three models)

Final summary of the attribution step, 2026-11-06. Attribution compute is finished; this file is the reference
for the thesis chapter. Figures of this full record (both targets, G×I): `_archive/superseded_attribution/figures/summary_notebook/`
and the archived notebook `_archive/superseded_attribution/attribution_results_summary.ipynb`.

> **Reporting decision (user, 2026-11-09):** results are reported for **target (a) only, with attention**
> (rollout only for total-flow questions); Gradient×Input joins IG in the appendix. The reported version is
> `attribution_results_kaggle.ipynb` / `figures/attribution/report_a/`. This file keeps the full record (both
> targets, G×I) for reference. The key new result there: a within-item regression (item fixed effects) shows that
> position alone has no effect (1.06–1.14×, CIs include 1), while a distractor written after a useful condition gets
> 0.60× / 0.69× / 0.74× (Qwen3 / Math / Phi, CIs below 1) and one written first does not get less (1.09–1.34×).

## Setup
- **Models:** Qwen3-4B-Instruct-2507, Qwen2.5-Math-1.5B-Instruct, Phi-4-mini-instruct.
- **Items:** 2 per category × subtype of the test split = 144 per model and target (target (a) uses only replies
  that finished: 129 / 143 / 143).
- **Targets:**
  - **(a), main:** log p of the model's own final answer, teacher-forced on its own working. It explains what the
    model actually did, wrong answers included.
  - **(c), control:** log p of the true solution right after the prompt, no working. Same target for every model.
- **Methods used for conclusions:** **Gradient×Input** and **attention** (mean over heads and layers, target tokens
  as queries). **Attention rollout** is shown but read as weak evidence (it gives the first prompt tokens 3.6–9× their
  share in every run, the usual early-token bias). **IG is completeness-only** (see the IG section).
- **Unit:** field level (instruction, equation, each condition, answer instruction), via `spans` → tokens.
- **Measure:** *density* = a field's share of the attribution ÷ its share of the prompt tokens (1 = in proportion to
  its length). Comparisons are per item and summarised as the **median of log2 ratios with a bootstrap 95% CI**
  (2000 resamples). Means of ratios are heavy-tailed and are not quoted.
- **Runs:** `results/attribution/20261104T*` (Qwen3, Phi: fp16; Qwen2.5-Math: fp32, because eager attention
  overflowed in fp16).

## Main findings

### 1. On average, attention gives less to the distractor (but see 1b: only when it comes second)
Distractor density ÷ sufficient-condition density, all distractor items (median [95% CI]):

| | attention (c) | attention (a) | G×I (c) | G×I (a) |
|---|---|---|---|---|
| Qwen3-4B | 0.65 [0.59, 0.87] | 0.74 [0.63, 0.91] | 1.01 [0.61, 1.38] | 1.16 [0.76, 1.88] |
| Qwen2.5-Math-1.5B | 0.69 [0.60, 0.81] | 0.84 [0.74, 0.93] | 0.78 [0.62, 0.95] | 0.93 [0.68, 1.20] |
| Phi-4-mini | 0.65 [0.61, 0.87] | 0.87 [0.78, 0.97] | 0.64 [0.52, 0.93] | 1.01 [0.71, 1.27] |

- **Attention:** every model looks at the distractor about a third less than at the useful conditions in (c),
  and less in (a) too. This is the most consistent result of the attribution step.
- **Gradient×Input:** in (c), Math and Phi also reduce the distractor's *effect on the answer* (CI below 1);
  Qwen3-4B does not. In (a), no model shows it.
- **So "looked at less" and "matters less for the answer" do not always agree.** The answer's log-probability
  stays sensitive to the distractor even when attention avoids it, especially after the working (a). This is a
  question for the representation analysis (see "Open questions").

### 1b. Correction (2026-11-09): the attention effect depends on the distractor's ORDER
Same ratio, split by where the distractor is written among the conditions (`rescaled` + `unrelated` items;
figure `9_distractor_order.png`, notebook section 1b). Attention, median [95% CI]:

| | target | control: 2 useful conditions, 2nd ÷ 1st | distractor FIRST | distractor AFTER a useful one |
|---|---|---|---|---|
| Qwen3-4B | (c) | 1.05 [0.95, 1.27] | 1.06 [0.99, 1.20] | **0.51 [0.47, 0.59]** |
| Qwen2.5-Math | (c) | 1.10 [0.94, 1.67] | 1.25 [1.06, 1.38] | **0.58 [0.51, 0.66]** |
| Phi-4-mini | (c) | 1.02 [0.90, 1.11] | 1.22 [1.12, 1.31] | **0.55 [0.47, 0.60]** |
| Qwen3-4B | (a) | 1.35 [0.99, 1.72] | 1.25 [1.06, 1.33] | **0.52 [0.48, 0.66]** |
| Qwen2.5-Math | (a) | 1.41 [0.94, 1.91] | 1.11 [1.03, 1.40] | **0.74 [0.65, 0.79]** |
| Phi-4-mini | (a) | 1.15 [0.93, 1.80] | 0.99 [0.90, 1.07] | **0.75 [0.68, 0.81]** |

(n in (c): control 18, first 30, after 42.)
- When the distractor comes **first**, it gets **as much or more** attention than the useful condition.
- When it comes **after** a useful condition, it gets about **half**.
- This is **not a plain position effect**: with two useful conditions and no distractor, the second gets about the
  same attention as the first.
- G×I shows the same split for Math and Phi in (c): distractor after 0.40 [0.23, 0.71] / 0.50 [0.33, 0.70],
  first 1.24 / 1.00. Qwen3 G×I shows neither.
- The overall 0.65× in finding 1 is a mix of these two cases.
- Reading: the models discount a condition that is **redundant or irrelevant given what came before it**, not an
  irrelevant condition as such (even a z-condition written first is not discounted). A plausible mechanism is
  causal attention: a distractor written first is encoded before any useful condition was seen, so its hidden state
  cannot mark it as redundant. Testable prediction for the probes: the distractor is linearly separable from useful
  conditions only when it comes after one.
- The `duplicate` result (finding 3: second copy ≈ 0.5×) fits the same pattern.

### 2. The effect depends on the kind of distractor
- **`unrelated`** (a condition on z, not y) gives the clearest effect: attention 0.61–0.70 in (c), CIs below 1 for all
  three models.
- **`rescaled`** gives the weakest: attention 0.75–0.89 in (c), CIs reach 1. A rescaled condition looks like a valid
  condition on y, so it is plausible the models don't treat it as irrelevant.
- **`duplicate`**: see 3.

### 3. Repeated condition: the models read mainly the first copy (a position effect)
In `duplicate` items the two conditions are identical text, so any difference is due to position only.
Second copy ÷ first copy:

| | attention (c) | attention (a) | G×I (c) | G×I (a) |
|---|---|---|---|---|
| Qwen3-4B | 0.54 [0.50, 0.57] | 0.55 [0.50, 0.58] | 0.92 [0.57, 1.31] | 0.84 [0.52, 1.40] |
| Qwen2.5-Math-1.5B | 0.55 [0.52, 0.58] | 0.62 [0.54, 0.64] | 0.46 [0.26, 0.72] | 0.77 [0.49, 1.18] |
| Phi-4-mini | 0.53 [0.48, 0.56] | 0.75 [0.69, 0.82] | 0.48 [0.29, 0.66] | 0.58 [0.45, 1.09] |

- Attention to the second copy is about **half** that of the first, in every model, with tight CIs.
- By G×I, Math and Phi also *rely* mainly on the first copy in (c); Qwen3-4B uses both copies about equally.

### 4. Where the attribution goes: the equation dominates, the models differ in how they read the prompt
Median density in target (c) (1 = in proportion to length):

| | method | instruction | equation | sufficient | distractor | answer instr. |
|---|---|---|---|---|---|---|
| Qwen3-4B | G×I | 1.18 | 1.38 | 0.83 | 0.76 | 0.85 |
| | attention | 1.30 | 1.31 | 1.20 | 0.85 | 0.92 |
| Qwen2.5-Math | G×I | 0.81 | **2.26** | 1.39 | 0.91 | 0.41 |
| | attention | **0.44** | **1.75** | 1.35 | 1.00 | 0.90 |
| Phi-4-mini | G×I | 2.26 | 1.80 | 1.07 | 0.68 | 0.54 |
| | attention | 1.79 | 1.78 | 1.17 | 0.82 | 0.76 |

- The **equation** is above proportion for every model and method.
- **Qwen2.5-Math** puts the most weight on the equation and the least on the instruction (attention 0.44); the
  general models (Qwen3-4B, Phi) give the instruction as much or more than its share. Plausibly an effect of the
  maths specialisation: a contrast to test in the representation analysis.

### 5. By layer: the useful conditions separate from the distractor in the middle-to-upper layers
Attention density per layer (figure 4): all prompt fields are below proportion in the first few layers. The ratio
sufficient ÷ distractor is small (~1.0–1.3) in the lower layers and **largest in the middle-to-upper layers**:

| model (layers) | ratio ≥ ~1.5 in target (c) (layer index from 0) | peak |
|---|---|---|
| Qwen3-4B (36) | layers ~18–33 | 2.2 (layer 23) |
| Qwen2.5-Math-1.5B (28) | layers ~14–24 | 2.7 (layer 20) |
| Phi-4-mini (32) | layers ~13–21 | 2.1 (layer 16) |

(Ratio of median densities.) Per item, the median of log2(sufficient ÷ distractor) peaks at 74% depth (Qwen3,
layer 26, 2.3×), 74% (Math, layer 20, 3.3×) and 61% (Phi, layer 19, 2.5×) in target (c); notebook figure
`4b_layer_separation_depth.png` puts the three models on a common relative-depth axis.

- **Target (c):** separation in roughly **40–85% of the network in all three models**, which suggests where to look
  first with probes.
- **Target (a):** smaller and flatter, about 1.2–1.6× across most layers, with no clear peak.
- Correction to the earlier 2026-11-06 note: Math does not separate clearly earlier than the others.

**The layer profile is not noise (checked 2026-11-09).** Split-half reliability of the median layer profile (two
random halves of the items) is r = 0.96–1.00 for every model, target and field, so the layer-to-layer zigzag is a
reproducible property of individual layers. The dips where equation, sufficient and distractor are all below 1 are
filled by the two fields not drawn in figure 4:
- **Early layers (2–4) read the instruction:** instruction density 4.3–8.3× (Qwen3, Phi), in both targets. The
  instruction profile is nearly the same in (c) and (a) (r = 0.85–0.98), so it does not depend on the target. This
  is also where rollout's early-token bias comes from.
- **Target (c): the maths is read in the middle-to-upper layers.** At the equation's peak layer: Qwen3 layer 23
  (equation 2.76, sufficient 2.07, distractor 0.93, instruction 0.08), Phi layer 17 (4.33 / 1.66 / 0.92), Math
  layer 20 (3.39 / 1.82 / 0.67).
- **Target (a): the order reverses.** The equation peaks early (layer 4–7, 13–26% depth), and several upper layers
  read the answer instruction instead (≈ 2×, e.g. Qwen3 layers 20–21, Math layer 19). The answer-instruction profile
  is anti-correlated between (c) and (a) (r = −0.19 to −0.28).
- Reading: a stable early "where is the task" stage, then the maths in the middle-to-upper layers when the model must
  derive the answer (c); when it copies the answer from its working (a), the upper layers check the output format.

### 6. Target (a): the model's own working takes a large share
Share of the attribution on the working tokens (rest = prompt fields): **G×I 18% / 31% / 36%** (Qwen3 / Math / Phi),
**attention 75% / 82% / 79%**, rollout 28% / 49% / 29%. The final answer depends heavily on the working, so
explanations that look only at the problem text miss a large part of the process. The size depends strongly on the
method; the direction does not.

### 7. Wrong answers come with less weight on the equation and the useful conditions
(Runs with at least 10 wrong answers: Phi (c, a), Math (c, a), Qwen3 (c, 13 wrong).)
- **Phi**, attention: (a) equation 0.63 → 0.40, sufficient 0.63 → 0.48; (c) equation 1.82 → 1.40, sufficient
  1.27 → 1.02. Non-overlapping CIs in most cases (40 / 39 wrong).
- **Math**, (c): attention on the equation 1.78 → 1.22, G×I on the sufficient conditions 1.42 → 0.97 (18 wrong).
- **Qwen3**, (c): same direction (attention on the equation 1.32 → 1.04), 13 wrong only.
- This shows up **in (c) too**, where the scored answer is the true one, so it looks like a property of how the
  model reads those problems, not a side effect of its wrong output. Strongest evidence for Phi; suggestive for the
  others (small n).

## Integrated Gradients: computed, not used (user decision, 2026-11-04)
- **Why it doesn't fit this task:**
  1. IG integrates gradients along a path from a baseline to the input. The points on that path are not valid
     problems, which is a weaker form of the objection that rules out LIME, SHAP and occlusion here.
  2. An ODE problem has no natural "empty" baseline. The two baselines we tried give **different explanations**:
     per item, the rank correlation between the pad- and zero-baseline field attributions has median 0.09–0.37,
     and they agree on the most-attributed field in only 1–33% of items (chance is about 1 in 6).
  3. It did not converge within a practical step budget: median completeness gap, pad baseline 4–31%,
     zero baseline 60–1160%. Only Math (c) and Phi (c) with the pad baseline are within 10%.
- Point 3 alone is a numerical issue (completeness holds with enough steps); the argument rests on 1 and 2.
- G×I is a gradient at the real input (no path, no baseline), so it does not share these problems.
- Figures: `7_ig_completeness.png`, `8_ig_baselines.png`.

## Caveats
- **Sample size:** 144 items per model and target (2 per cell). Subtype panels have 36 items each; correct vs wrong
  comparisons rest on 13–40 wrong answers.
- **G×I saturation:** 13 of 129 Qwen3 (a) items have log p(answer) = 0 (the answer is already in the working), so
  the gradient is zero and they drop out of the G×I numbers.
- **Rollout** has an early-token bias (instruction 3.6–9× its share); read it as weak evidence.
- **Attention is averaged** over heads and layers in the field tables; single heads can behave very differently.
- **Precision:** Qwen2.5-Math ran in fp32, the others in fp16. The field-level numbers are ratios within an item, so
  this should not bias the comparisons, but it is a difference in setup.

## Open questions passed to the representation analysis
1. **Attention vs gradient on the distractor:** is the distractor encoded differently from sufficient conditions in
   the hidden states? A linear probe for condition role per layer would answer it; the attention result predicts
   separation in the middle-to-upper layers (40–85% depth).
2. **`rescaled`:** is a rescaled condition's representation closer to a sufficient condition's than the
   `unrelated` z-condition's is?
3. **Position:** in `duplicate` items, are the two identical copies represented the same way, layer by layer?
4. **Math vs general models:** does Qwen2.5-Math's stronger weight on the equation show up as a different
   representational geometry (RSA / CKA across models)?
5. **Correct vs wrong:** can the field representations predict whether the answer will be correct?
