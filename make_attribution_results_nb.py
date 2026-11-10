"""Builds attribution_results_kaggle.ipynb: the clean attribution results notebook (user decisions 2026-11-09).

Target (a) only, attention as the method for every result, rollout only where it answers a different question,
Gradient×Input and Integrated Gradients only in the appendix. All analysis code is in attribution_report.py.

  python make_attribution_results_nb.py            # writes the notebook
  python make_attribution_results_nb.py --execute  # writes and runs it locally (figures embedded)

Runs on Kaggle (CPU, no internet needed) with the dataset made from kaggle_attribution_results_files.zip, or
locally from this folder.
"""
import json, sys

CELLS = []
def md(s): CELLS.append(('md', s.strip('\n')))
def code(s): CELLS.append(('code', s.strip('\n')))


md(r"""
# Attribution results: where the models look when they write their answer

**Models:** Qwen3-4B-Instruct-2507, Qwen2.5-Math-1.5B-Instruct, Phi-4-mini-instruct.
**Items:** test split, 2 problems per category × subtype (144 per model); only replies that finished with an
answer are used (129 / 143 / 143).

**What is explained.** The tokens of the model's **own final answer** (`Answer: y = ...`), with the model's **own
step-by-step working** in the input, exactly as it generated it. This is the setting in which the models solve the
problems; the answer-without-working control is not reported.

**Method.** **Attention** from the answer tokens to every input token, averaged over heads and layers. Attention
rollout (attention followed back through all layers) is used only in section 1, where the question is the total
flow through the network. It favours the first tokens of the input by construction, so it is not used to compare
positions. Gradient×Input and Integrated Gradients are not used for results; Appendix A shows why.

**Measure.** The problem text is split into fields: instruction, equation, each condition, answer-format
instruction. *Density* = a field's share of the attention ÷ its share of the problem's tokens (1× = in proportion to
its length). Every comparison is made **per item** and summarised by the **median with a bootstrap 95% CI**
(2000 resamples, seed 0). An interval that does not contain 1× is a clear effect.

All numbers below are computed by `attribution_report.py` in the next cells; figures and CSV tables are saved
next to the notebook and zipped at the end.
""")

code(r"""
import glob, os, shutil, sys
INPUT = '/kaggle/input'
if os.path.isdir(INPUT):                                    # on Kaggle: copy the uploaded files to a writable folder
    found = glob.glob(os.path.join(INPUT, '**', 'attribution_report.py'), recursive=True)
    assert found, 'add the dataset made from kaggle_attribution_results_files.zip as an Input of this notebook'
    WORK = '/kaggle/working/Thesis'
    shutil.copytree(os.path.dirname(found[0]), WORK, dirs_exist_ok=True)
    os.chdir(WORK)
sys.path.insert(0, os.getcwd())
OUT = os.path.join('figures', 'attribution', 'report_a')

import pandas as pd
from IPython.display import Image, display
import attribution_report as AR
pd.set_option('display.max_colwidth', None)

R = AR.build('results/attribution', OUT)                    # all analyses, ~10-60 s on CPU
T = lambda name: pd.DataFrame(R['tables'][name])
show = lambda key, width=None: display(Image(filename=R['figures'][key], width=width))
T('runs')
""")

md("## Key results")
code("T('summary').set_index('result')")
md(r"""
- **The answer is read mostly from the model's own working:** 75–82% of the answer tokens' attention goes to the
  working, the rest to the problem text.
- **The distractor condition is attended less only when it is written after a useful condition** (0.60–0.74× a
  useful condition at the same position). Written first, it is not attended less (1.09–1.34×). Position on its own
  has no effect (1.06–1.14×, all intervals include 1).
- **A condition written twice:** the second copy gets 0.55–0.75× the attention of the first.
- **Phi-4-mini's wrong answers** come with less attention on the equation (0.63×) and the useful conditions (0.76×).
  Qwen2.5-Math shows no difference; Qwen3-4B has only 3 wrong answers.
""")

md(r"""
## 1. Where the answer tokens look
Left: direct attention from the answer tokens to each field of the problem. Right: attention rollout, which also
counts indirect paths (answer → working → problem).
""")
code("show('where')\nT('where').set_index(['method', 'model'])")
md(r"""
**Results**
- Within the problem text, **Qwen3-4B and Phi-4-mini attend most to the two instructions**: instruction 1.58× and
  2.67×, answer-format instruction 1.50× and 1.09×. The equation and the conditions get less than their share
  (0.40–0.58×).
- **Qwen2.5-Math reads the problem differently:** instruction 0.39×, equation 0.90×, useful conditions 0.97×,
  answer-format instruction 1.31×.
- **The distractor is the least-attended field in every model** (0.29×, 0.80×, 0.48×).
- Rollout puts 3.6–8.0× on the instruction, the first tokens of the problem. That is rollout's known bias towards
  early tokens, not a property of the instruction.
""")
code("show('working', 750)\nT('working').set_index('method')")
md(r"""
**Results**
- **Direct attention:** 75% (Qwen3-4B), 82% (Qwen2.5-Math), 79% (Phi-4-mini) of the answer tokens' attention goes
  to the model's own working.
- **Rollout:** 28%, 49%, 29%. Once indirect paths are counted, more of the flow ends in the problem text, because the
  working tokens themselves attend back to the problem.
""")

md(r"""
## 2. The distractor condition: order matters
Each item has the useful conditions (needed for the solution) and, in three subtypes, one **distractor**:
`rescaled` (a useful condition multiplied through), `unrelated` (a condition on another variable z), or `duplicate`
(a useful condition written twice). Ratios are attention **per token**, so condition length does not matter.
The control row uses items with two useful conditions and no distractor.
""")
code("show('order', 850)\nT('order').set_index('case')")
md(r"""
**Results**
- **Two useful conditions:** the second gets 1.15–1.41× the first; all intervals include 1.
- **Distractor written first:** 0.99–1.25× a useful condition, so it is **not** attended less.
- **Distractor written after a useful condition:** 0.52× (Qwen3-4B), 0.74× (Qwen2.5-Math), 0.75× (Phi-4-mini); all
  intervals below 1.
- **Same condition written twice:** the second copy gets 0.55×, 0.62×, 0.75× of the first; all intervals below 1.
""")
md("The same split for each kind of distractor:")
code("T('order_subtype').set_index(['subtype', 'distractor'])")
md(r"""
**Results:** `rescaled` and `unrelated` show the same pattern. Written first: 0.97–1.27×. Written after a useful
condition: 0.50–0.79×. (The `unrelated`-first cells have only 7–11 items.)
""")

md(r"""
## 3. Position and role, separated
Section 2 compares conditions at different positions, so position and role are mixed. Here they are separated
with one regression per model: for each condition of an item, log2 attention density is modelled as

`log2 density = item effect + position (2nd, 3rd vs 1st) + role (distractor written first / distractor written after a useful one, vs useful)`

The item effect removes everything that is shared within a problem. Data: items with at least two conditions
(`none` second-order, `rescaled`, `unrelated`; `duplicate` is left out because its two copies are identical text).
CIs come from 1000 bootstrap resamples of items.
""")
code("show('regression', 850)\nT('regression').set_index('effect')")
md(r"""
**Results**
- **Position on its own has no clear effect:** position 2 vs 1 is 1.06–1.14×, position 3 vs 1 is 0.98–1.17×; all
  intervals include 1.
- **Distractor written first, same position:** 1.23× (Qwen3-4B), 1.34× (Qwen2.5-Math, interval above 1), 1.09×
  (Phi-4-mini).
- **Distractor written after a useful one, same position:** 0.60× [0.47, 0.74], 0.69× [0.57, 0.82], 0.74×
  [0.63, 0.86]; all intervals below 1.
- So the lower attention to the distractor is **not a position effect**. It appears only when a useful condition
  comes before the distractor.
""")

md(r"""
## 4. Layer by layer
Figure 5: median density of each field in each layer. Figure 6: the regression of section 3, fitted separately in
every layer, on a common relative-depth axis.
""")
code("show('layers')\nprint('Split-half reliability of each layer profile (two random halves of the items, median of 200 splits):')\nT('layers_reliability').set_index('model')")
md(r"""
**Results**
- **The profiles are reproducible:** two random halves of the items give the same layer profile (r = 0.97–1.00), so
  the jumps between neighbouring layers are properties of those layers, not noise.
- **Instruction in the early layers:** Qwen3-4B layers 2–6 (up to 5.9×), Phi-4-mini layers 1–3 (up to 8.3×),
  Qwen2.5-Math up to 2.4× (layer 3).
- **Equation in the early-middle layers:** Qwen3-4B layers 7–9 (up to 1.6×), Qwen2.5-Math layers 2–9 (up to 1.9×),
  Phi-4-mini layers 4–6 (up to 1.5×).
- **Answer-format instruction in the upper layers:** above 1.2× in Qwen3-4B layers 13–34 (up to 2.0×), most of
  Qwen2.5-Math layers 10–26 (up to 1.9×), and Phi-4-mini layers 7–25 (up to 1.5×).
""")
code("show('layers_effects')")
md(r"""
**Results**
- **Position:** in the first ~5–15% of the layers, a later condition gets 0.51–0.62× the attention of an earlier one
  (the early layers favour earlier text). From about 25% depth on, the median effect is 1.12–1.19×.
- **Distractor written after a useful one:** below 1× from about 20–25% depth on. Median over the layers past 25%
  depth: 0.56× (Qwen3-4B), 0.78× (Qwen2.5-Math), 0.76× (Phi-4-mini). The lowest values are 0.42× (Qwen3-4B, layer 19,
  54% depth), 0.33× (Qwen2.5-Math, layer 20, 74%) and 0.62× (Phi-4-mini, layer 13, 42%).
- **Distractor written first:** at or above 1× in most layers. Median past 25% depth: 1.38×, 1.44×, 1.14×.
- The position effect sits in the early layers; the role effect sits in the middle and upper layers.
""")

md(r"""
## 5. Correct vs wrong answers
Attention density for wrong answers ÷ correct answers, per field (median ratio, bootstrap 95% CI). Only models with
at least 10 wrong answers among these items.
""")
code("print('Not analysed (fewer than 10 wrong answers):', ', '.join(R['skipped_wrong']) or 'none')\nshow('wrong', 800)\nT('wrong').set_index(['model', 'field'])")
md(r"""
**Results**
- **Phi-4-mini (39 wrong of 143):** wrong answers come with less attention on the equation (0.63× [0.53, 0.78]), the
  useful conditions (0.76× [0.68, 0.89]) and the distractor (0.81× [0.73, 0.95]), and more on the instruction
  (1.37× [1.20, 1.47]). The answer-format instruction doesn't change (1.04×, interval includes 1).
- **Qwen2.5-Math (17 wrong of 143):** no difference in any field (0.94–1.07×, all intervals include 1).
- **Qwen3-4B:** 3 wrong answers among these items, too few to compare.
""")

md(r"""
## Appendix A. Gradient-based methods (Gradient×Input, Integrated Gradients): not used
Both methods measure how the answer's log-probability **responds to changes in the input embeddings**:
Gradient×Input responds to an infinitesimal change at the input, and Integrated Gradients along a path of changed
inputs from a baseline. In a differential equation every symbol matters, and a small change gives a different
problem. The response to such changes therefore doesn't show what the model used to solve *this* problem. This is
the same reason LIME, SHAP and occlusion are excluded. The two methods were computed on the same items; the
figures below show how they behave here.
""")
code("show('gxi')\nT('gxi').set_index('model')")
md(r"""
**Gradient×Input**
- Per item, its distractor ratio barely agrees with attention (rank correlation −0.08 to 0.25), and it is 2–5× more
  spread out (IQR 2.0–3.1 vs 0.6–1.1 in log2).
- The model's own answer is almost certain after its working (p > 0.99 for 32–76% of items). The gradient of log p
  shrinks towards zero as p approaches 1.
- Within a field, positive and negative token contributions cancel: only 32–35% of the summed magnitude remains.
""")
code("show('ig')\nT('ig').set_index('model')")
md(r"""
**Integrated Gradients**
- The attributions don't add up to the change in the output: the median completeness gap is 10–31% with the pad
  baseline and 60–441% with the all-zero baseline (32 steps).
- The two baselines explain the same item differently: the rank correlation of their field attributions is
  0.14–0.37, and they agree on the top field in 26–31% of items (chance 16%).
""")

md(r"""
## Appendix B. Notes on the data
- **Sample size:** 144 items per model (2 per category × subtype); 129 / 143 / 143 with a finished answer. Order
  groups have 16–42 items; correct-vs-wrong rests on 17 and 39 wrong answers.
- **Attention is averaged over heads** (and over layers, except in section 4); single heads can behave differently.
- **Precision:** Qwen2.5-Math ran in fp32 (fp16 overflowed in eager attention), the other two in fp16. All measures
  are ratios within an item.
- **`duplicate` items** are used only for the second-copy ÷ first-copy comparison. Which of the two identical copies
  the dataset labels "distractor" is arbitrary, so they are left out of the distractor ÷ useful comparisons.
""")

code(r"""
# everything this notebook produced, in one zip (Kaggle: Output tab)
path = shutil.make_archive('attribution_report_a', 'zip', root_dir=OUT)
print('wrote', os.path.abspath(path), '(figures, CSV tables, stats.json)')
""")


def to_nb():
    cells = []
    for kind, s in CELLS:
        c = {'cell_type': 'markdown' if kind == 'md' else 'code', 'metadata': {}, 'source': s.splitlines(True)}
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                         'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 4}


if __name__ == '__main__':
    out = 'attribution_results_kaggle.ipynb'
    json.dump(to_nb(), open(out, 'w', encoding='utf-8'), indent=1)
    print('wrote', out)
    if '--execute' in sys.argv:
        import nbformat
        from nbclient import NotebookClient
        nb = nbformat.read(out, as_version=4)
        NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
        nbformat.write(nb, out.replace('.ipynb', '_executed.ipynb'))
        print('executed ->', out.replace('.ipynb', '_executed.ipynb'))
