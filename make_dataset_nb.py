"""Builds dataset_overview_kaggle.ipynb (IVP-OPEN description with figures)."""
import json, sys

CELLS = []
def md(s): CELLS.append(('md', s.strip('\n')))
def code(s): CELLS.append(('code', s.strip('\n')))

md(r"""
# IVP-OPEN: the dataset for the thesis
**How do small language models solve initial value problems, and where do they go wrong?**

This notebook describes the dataset used to answer that question: what it contains, how it was built, how it is
distributed, and why each answer can be trusted. Every number and figure below is computed live from the data files.

**At a glance**
- **864 initial value problems (IVPs)**: ODE + conditions; the model must find the unique solution `y(x)`.
- **Open-ended**: no answer options. The model shows its working and ends with `Answer: y = <expression>`;
  SymPy checks the answer against the exact solution.
- **18 problem categories** (6 ODE types × 3), **4 condition settings** (a plain control plus 3 kinds of
  irrelevant extra condition), **12 problems per category × setting**: fully balanced.
- **Splits:** `dev` = 72 problems (model selection, prompt tuning), `test` = 792 problems (reported results).
- **Exact labels:** every problem is built backwards from a known solution and checked again by an
  independent verifier (section 8).
- **Annotated for interpretability:** every condition carries its role (*sufficient* or *distractor*), and
  every prompt carries character spans for the instruction, the equation and each condition. These spans let
  attributions and hidden states be pooled per field.

*Run on Kaggle:* add the dataset made from `kaggle_dataset_overview_files.zip` as an Input, then **Run All**
(CPU is enough: no GPU or internet needed; about 2 minutes, most of it the full verification in section 8). The figures are
also saved as PNGs in `/kaggle/working/figures/`.
""")

code(r"""
import os, sys, glob, json, re, collections
import numpy as np, pandas as pd, sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown

INPUT = os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input')
found = glob.glob(os.path.join(INPUT, '**', 'ivp_open_all.jsonl'), recursive=True)
assert found, f'ivp_open_all.jsonl not found under {INPUT}: add the dataset as an Input of this notebook'
DATA_DIR = os.path.dirname(found[0])                      # .../ivp_open
sys.path[:0] = [DATA_DIR, os.path.dirname(DATA_DIR)]      # grade_open.py, gold_check.py
FIG = os.environ.get('FIG_DIR', '/kaggle/working/figures'); os.makedirs(FIG, exist_ok=True)

items = [json.loads(l) for l in open(found[0], encoding='utf-8')]
df = pd.DataFrame(items)
x = sp.Symbol('x')
TYPES = ['first_order_direct', 'first_order_separable', 'first_order_linear',
         'second_order_direct', 'second_order_homogeneous', 'second_order_nonhomogeneous']
TYPE_LABEL = {t: t.replace('first_order_', '1st: ').replace('second_order_', '2nd: ') for t in TYPES}
SUBTYPES = ['none', 'duplicate', 'rescaled', 'unrelated']
CATS = [c for t in TYPES for c in sorted(df[df.ode_type == t].category.unique())]
df['ode_type'] = pd.Categorical(df.ode_type, TYPES, ordered=True)
df['subtype'] = pd.Categorical(df.subtype, SUBTYPES, ordered=True)
df['category'] = pd.Categorical(df.category, CATS, ordered=True)

plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'font.size': 10})
SUB_COL = {'none': '#7f7f7f', 'duplicate': '#1f77b4', 'rescaled': '#2ca02c', 'unrelated': '#d62728'}
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), bbox_inches='tight', dpi=150)
print(f'{len(items)} problems loaded from {found[0]}')
""")

md(r"""
## 1. Overview
""")
code(r"""
conds = pd.DataFrame([dict(c, id=it['id'], subtype=it['subtype'], ode_type=it['ode_type'], order=it['order'],
                           position=k + 1) for it in items for k, c in enumerate(it['conditions'])])
overview = pd.Series({
    'problems': len(df),
    'dev / test': f"{(df.split == 'dev').sum()} / {(df.split == 'test').sum()}",
    'ODE types / categories': f"{df.ode_type.nunique()} / {df.category.nunique()}",
    'condition settings (subtypes)': ', '.join(SUBTYPES),
    'problems per category × subtype': f"{df.groupby(['category', 'subtype'], observed=True).size().min()}"
                                       f"–{df.groupby(['category', 'subtype'], observed=True).size().max()}",
    '1st-order / 2nd-order problems': f"{(df.order == 1).sum()} / {(df.order == 2).sum()}",
    'conditions in total': len(conds),
    'of which distractors': int((conds.role == 'distractor').sum()),
    'unique prompts': df.prompt.nunique(),
    'prompt version': ', '.join(df.prompt_version.unique()),
}, name='value').to_frame()
display(overview)
""")

md(r"""
## 2. What a problem looks like

Every prompt has the same four fields: an **instruction**, the **differential equation**, a numbered list of
**conditions**, and an **answer instruction** that fixes the format of the final line. One problem from each
condition setting (all four in the same category, so only the conditions differ in kind):
""")
code(r"""
def show_item(it, title=''):
    roles = ' / '.join(f"{k}: {c['role']}" for k, c in enumerate(it['conditions'], 1))
    extra = ''
    if it['subtype'] == 'unrelated':
        extra = f"\n# trap (z-condition read as y):  {it['info']['trap_solution']}"
    print(f"# {title}{it['id']}  [{it['split']}]  {it['ode_type']} / {it['category']} / subtype = {it['subtype']}")
    print(it['prompt'])
    print(f"# condition roles: {roles}\n# true solution:   {it['answer_text']}{extra}\n")

cat = 'direct2_trig'
for s in SUBTYPES:
    show_item(next(it for it in items if it['category'] == cat and it['subtype'] == s and it['split'] == 'dev'))
""")

md(r"""
## 3. Problem types: 6 ODE types × 3 categories

The categories cover the standard textbook IVPs: direct integration, separable, first-order linear, and
second-order constant-coefficient equations (homogeneous and non-homogeneous, with distinct real, repeated and
complex characteristic roots). Each general solution is linear in its free constants. Each problem's
coefficients and true constants are small random integers, so problems in one category share their
structure but differ in every number.
""")
code(r"""
rows = []
for c in CATS:
    it = next(i for i in items if i['category'] == c)
    g = sp.sympify(it['sympy_general_solution'])
    rows.append({'ode_type': it['ode_type'], 'category': c, 'order': it['order'],
                 'example equation': it['equation'], 'its general solution': f'y = {g}',
                 'example true solution': it['answer_text'], 'problems': int((df.category == c).sum())})
display(pd.DataFrame(rows).set_index(['ode_type', 'category']))
""")

md(r"""
**The 18 categories drawn.** One test problem per category: the true solution (solid line) and its conditions
(dots: a value condition `y(x0)`; triangles: a slope condition `y'(x0)`).
""")
code(r"""
def f_np(it, key='sympy_solution'):
    return sp.lambdify(x, sp.sympify(it[key]), 'numpy')

def cond_points(it, fn='y'):
    out = []
    for c in it['conditions']:
        if c['function'] != fn: continue
        m = re.match(r"^\s*(-?\d+)?\*?[yz]('*)\(([^)]*)\)\s*=\s*(.+)$", c['text'])
        coef = int(m.group(1)) if m.group(1) else 1
        out.append((float(sp.sympify(m.group(3))), float(sp.sympify(m.group(4))) / coef, len(m.group(2))))
    return out

fig, axes = plt.subplots(6, 3, figsize=(12, 17))
for ax, c in zip(axes.flat, CATS):
    it = next(i for i in items if i['category'] == c and i['split'] == 'test' and i['subtype'] == 'none')
    pts = cond_points(it)
    lo, hi = min(p[0] for p in pts) - 1.5, max(p[0] for p in pts) + 1.5
    if c == 'separable_y_over_x': lo = max(lo, 0.35)
    xs = np.linspace(lo, hi, 300); f = f_np(it); ys = f(xs) * np.ones_like(xs)
    ax.plot(xs, ys, color='#1f77b4', lw=2)
    for x0, v, d in pts:
        if d == 0: ax.plot(x0, v, 'o', color='k', ms=7, zorder=5)
        else:
            ax.plot(x0, f(x0), '^', color='#ff7f0e', ms=8, zorder=5)
            ax.plot([x0 - .4, x0 + .4], [f(x0) - .4 * v, f(x0) + .4 * v], color='#ff7f0e', lw=2)
    q = np.nanpercentile(ys, [3, 97]); pad = (q[1] - q[0]) * .25 + 1
    ax.set_ylim(q[0] - pad, q[1] + pad)
    ax.set_title(f"{c}\n{it['equation']}", fontsize=9)
    ax.text(.02, .04, it['answer_text'], transform=ax.transAxes, fontsize=8, color='#1f77b4')
fig.suptitle('One problem per category: true solution and its conditions', y=1.0, fontsize=13)
fig.tight_layout(); save(fig, '01_categories'); plt.show()
""")

md(r"""
## 4. Condition settings (subtypes)

Every problem has exactly enough information to fix `y`. Three of the four settings add **one extra
condition that carries no new information**. This is the experimental handle for the research question: does
the model use the conditions that determine the answer, and does it leave the irrelevant one alone?

| subtype | conditions | the extra condition | what it tests |
|---|---|---|---|
| `none` | exactly as many as the ODE order | none | control: the plain IVP |
| `duplicate` | order + 1 | an exact copy of one condition | the same text twice; any difference in how the model treats the two copies is a position effect |
| `rescaled` | order + 1 | one condition multiplied through, e.g. `-2*y(1) = 6` next to `y(1) = -3` | recognising an equivalent but differently written condition |
| `unrelated` | order + 1 | a condition on a *different* function, `z(x0) = v` | ignoring irrelevant information; reading `z` as `y` gives a specific wrong answer (the **trap**) |

For every condition the data records its **role** (`sufficient` or `distractor`), the function it constrains
(`y` or `z`), its derivative order, its point `x0` and the complexity of its value.
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
ct = pd.crosstab(df.subtype, df.n_conditions)
ct.plot.bar(ax=axes[0], color=['#c7dcef', '#6baed6', '#2171b5'], rot=0, width=.75)
axes[0].set_title('Conditions per problem'); axes[0].set_xlabel(''); axes[0].set_ylabel('problems')
axes[0].legend(title='conditions', frameon=False, ncol=3, loc='upper center'); axes[0].set_ylim(0, 140)

d = df[df.subtype != 'none'].copy(); d['distractor_position'] = d['info'].map(lambda i: i['distractor_position'])
ct = pd.crosstab([d.subtype.astype(str), d.order], d.distractor_position)
ct.index = [f'{s}\norder {o}' for s, o in ct.index]
ct.plot.bar(ax=axes[1], stacked=True, color=['#fdd0a2', '#fd8d3c', '#a63603'], rot=0, width=.7)
axes[1].set_title('Position of the distractor in the list'); axes[1].set_ylabel('problems')
axes[1].legend(title='position', frameon=False, fontsize=8, ncol=3, loc='upper center'); axes[1].set_ylim(0, 135)
axes[1].tick_params(axis='x', labelsize=8)

dist = conds[conds.role == 'distractor']
kinds = pd.crosstab(dist.subtype, dist.derivative_order).reindex(SUBTYPES[1:])
kinds.columns = [["value y(x0)", "slope y'(x0)"][k] for k in kinds.columns]
kinds.plot.bar(ax=axes[2], color=['#9e9ac8', '#54278f'], rot=0, width=.7)
axes[2].set_title('What the distractor constrains'); axes[2].set_xlabel('')
axes[2].legend(frameon=False, ncol=2, loc='upper center'); axes[2].set_ylim(0, 190)
fig.tight_layout(); save(fig, '02_subtypes'); plt.show()
""")

md(r"""
The distractor's position is balanced within every category × subtype. A model can't find the distractor
by its place in the list, and position effects can be separated from content effects.
""")

md(r"""
## 5. Distribution and balance
""")
code(r"""
fig, ax = plt.subplots(figsize=(9, 7))
ct = pd.crosstab(df.category, df.subtype).reindex(index=CATS, columns=SUBTYPES)
dv = pd.crosstab(df[df.split == 'dev'].category, df[df.split == 'dev'].subtype).reindex(index=CATS, columns=SUBTYPES).fillna(0).astype(int)
ax.imshow(ct.values, cmap='Blues', vmin=0, vmax=ct.values.max() * 2.2, aspect='auto')
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, f'{ct.values[i, j]}  ({dv.values[i, j]} dev / {ct.values[i, j] - dv.values[i, j]} test)',
                ha='center', va='center', fontsize=8.5)
ax.set_xticks(range(4), SUBTYPES); ax.set_yticks(range(len(CATS)), CATS); ax.grid(False)
ax.xaxis.tick_top()
for b, t in zip((2.5, 5.5, 8.5, 11.5, 14.5, 17.5), TYPES):
    ax.axhline(b, color='k', lw=.6)
    short = TYPE_LABEL[t].replace('nonhomogeneous', 'non-hom.').replace('homogeneous', 'hom.')
    ax.text(3.62, b - 1.5, short, rotation=-90, va='center', fontsize=8.5, color='#444')
ax.set_title('Problems per category × subtype (all 864)', pad=28)
fig.tight_layout(); save(fig, '03_balance_heatmap'); plt.show()

split_tab = pd.crosstab([df.ode_type], [df.split, df.subtype]).reindex(TYPES)
split_tab.loc['total'] = split_tab.sum()
display(split_tab)
""")

md(r"""
Every cell holds exactly 12 problems: 1 in dev and 11 in test. All results can be broken down by ODE type,
category or subtype without unequal group sizes.
""")

md(r"""
## 6. The conditions in detail
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
conds['x0'] = conds.point.astype(str)
order_pts = ['0', '1', '-1', '2', '3', 'pi/2', 'pi']
pt = pd.crosstab(conds.x0, conds.role).reindex([p for p in order_pts if p in set(conds.x0)])
pt.plot.bar(ax=axes[0], stacked=True, color={'sufficient': '#4292c6', 'distractor': '#ef6548'}, rot=0)
axes[0].set_title('Condition point x0'); axes[0].set_xlabel('x0'); axes[0].set_ylabel('conditions')
axes[0].legend(frameon=False)

do = pd.crosstab(conds.order.map({1: '1st-order ODEs', 2: '2nd-order ODEs'}), conds.derivative_order)
do.columns = ["y(x0)", "y'(x0)"]
do.plot.bar(ax=axes[1], color=['#9e9ac8', '#54278f'], rot=0)
axes[1].set_title('Value vs slope conditions'); axes[1].set_xlabel(''); axes[1].legend(frameon=False)

bins = np.arange(-.5, conds.value_complexity.max() + 1.5)
for r, col in [('sufficient', '#4292c6'), ('distractor', '#ef6548')]:
    v = conds[conds.role == r].value_complexity
    axes[2].hist(v, bins=bins, alpha=.65, color=col, label=f'{r} (mean {v.mean():.1f})', density=True)
axes[2].set_title('Complexity of the condition value\n(SymPy operation count; 0 = a plain integer)')
axes[2].set_xlabel('operations in the value'); axes[2].set_ylabel('share of conditions'); axes[2].legend(frameon=False)
fig.tight_layout(); save(fig, '04_conditions'); plt.show()

ex = conds.sort_values('value_complexity').groupby('value_complexity').first()[['text', 'ode_type']]
display(ex.rename(columns={'text': 'example condition'}))
""")

md(r"""
The points are chosen per category so that the values remain exact. Away from `x0 = 0`, values contain
terms like `exp(2)` or `cos(1)`, which makes the substitution step harder. Distractor values follow the
same distribution as the real conditions, so the distractor can't be spotted by how its value looks.
""")

md(r"""
## 7. How hard are the problems? Size of prompts and solutions
""")
code(r"""
df['prompt_chars'] = df.prompt.str.len()
df['solution_ops'] = [int(sp.count_ops(sp.sympify(s))) for s in df.sympy_solution]
df['solution_terms'] = [len(sp.Add.make_args(sp.expand(sp.sympify(s)))) for s in df.sympy_solution]
df['max_abs_constant'] = [max(abs(c) for c in cs) for cs in df.constants]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, col, title in [(axes[0], 'prompt_chars', 'Prompt length (characters)'),
                       (axes[1], 'solution_ops', 'Size of the true solution\n(SymPy operation count)')]:
    data = [df[df.ode_type == t][col] for t in TYPES]
    bp = ax.boxplot(data, patch_artist=True, widths=.6)
    for b, t in zip(bp['boxes'], TYPES): b.set_facecolor('#c6dbef' if t.startswith('first') else '#fdd0a2')
    ax.set_xticks(range(1, 7), [TYPE_LABEL[t] for t in TYPES], rotation=30, ha='right', fontsize=8)
    ax.set_title(title)
cs = pd.Series([c for cs in df.constants for c in cs])
axes[2].bar(*np.unique(cs, return_counts=True), color='#6a51a3')
axes[2].set_title('True constants (C1, C2) of the solutions'); axes[2].set_xlabel('value'); axes[2].set_ylabel('count')
fig.tight_layout(); save(fig, '05_size'); plt.show()

display(df.groupby('ode_type', observed=True)[['prompt_chars', 'solution_ops', 'solution_terms']]
          .agg(['mean', 'min', 'max']).round(1))
""")

md(r"""
The prompts are short (a few hundred characters), so the whole input fits easily into attribution and
probing analyses. Solution size grows with the ODE type: second-order non-homogeneous problems have the
largest solutions, and they are also where the models struggle most.
""")

md(r"""
## 8. Why the labels can be trusted

**Construction (backwards from the answer).** For each problem the generator:
1. picks an ODE from the category, with random small-integer coefficients;
2. picks the true constants, which give the true solution `y(x)`;
3. derives the conditions from that solution;
4. keeps the problem only if the `y`-conditions determine the constants uniquely (the rank of the condition
   system in the general solution's constants equals the ODE order).

For the `unrelated` subtype it also stores the **trap solution**: what you get by treating the `z`-condition
as a condition on `y`.

**Independent verification.** `check_ivp_open.py` re-reads every problem *from its text only*, without the
generator's internals, and checks with SymPy:
1. the answer solves the ODE and satisfies every `y`-condition;
2. those conditions determine the solution;
3. each subtype's distractor has its property:
   - `duplicate`: a verbatim copy;
   - `rescaled`: still determined without it;
   - `unrelated`: the only `z`-condition, and its trap differs from the answer;
4. the character spans point at the right text;
5. the grader accepts the true answer and rejects the trap.
""")
code(r"""
from check_ivp_open import check
import time
t0 = time.time()
failures = {it['id']: e for it in items if (e := check(it))}
print(f'VERIFIED {len(items) - len(failures)}/{len(items)} problems in {time.time() - t0:.0f}s')
for k, v in list(failures.items())[:10]: print(k, v)
""")

md(r"""
**Grading a model's reply.** The grader (`grade_open.py`) takes the expression on the final `Answer:` line,
accepting plain text or LaTeX (`\frac`, `e^{2x}`, `\boxed{}`, ...). It then checks the expression against
the true solution numerically at five points. Answers that still contain free constants (`C_1`) count as
wrong. Besides right/wrong, it records what a wrong answer does: does it still solve the ODE? Which
conditions does it satisfy? Did it fall into the `z`-trap? Some example replies:
""")
code(r"""
from grade_open import grade_reply
it = next(i for i in items if i['subtype'] == 'unrelated' and i['category'] == 'direct2_trig' and i['split'] == 'dev')
true = it['answer_text'].split('=', 1)[1].strip()
trap = it['info']['trap_solution'].split('=', 1)[1].strip()
replies = {
    'correct, plain text':           f'... working ...\nAnswer: y = {true}',
    'correct, LaTeX':                '... \\[ y = ' + sp.latex(sp.sympify(it['sympy_solution'])) + ' \\]\nAnswer: $y = ' + sp.latex(sp.sympify(it['sympy_solution'])) + '$',
    'fell into the z-trap':          f'Answer: y = {trap}',
    'left a free constant':          'Answer: y = ' + str(sp.sympify(it['sympy_general_solution'])),
}
print(it['prompt'].split('\n\nShow')[0], '\n')
rows = []
for name, rep in replies.items():
    g = grade_reply(it, rep)
    rows.append({'reply': name, 'parsed answer': g['answer_parsed'], 'status': g['status'], 'correct': g['correct'],
                 'solves ODE': g['solves_ode'], 'satisfies conditions': g['satisfies'], 'equals trap': g['equals_trap']})
display(pd.DataFrame(rows).set_index('reply'))
""")

md(r"""
**The `unrelated` trap, drawn.** Each panel shows one problem: the true solution, which passes through
its `y`-conditions, and the trap, which passes through the `z`-condition read as `y`. Both solve the same ODE.
""")
code(r"""
un = [i for i in items if i['subtype'] == 'unrelated' and i['split'] == 'dev']
pick = [next(i for i in un if i['ode_type'] == t) for t in TYPES]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, it in zip(axes.flat, pick):
    pts, zpts = cond_points(it, 'y'), cond_points(it, 'z')
    allx = [p[0] for p in pts + zpts]
    lo, hi = min(allx) - 1.5, max(allx) + 1.5
    if it['category'] == 'separable_y_over_x': lo = max(lo, .35)
    xs = np.linspace(lo, hi, 300)
    f = f_np(it); g = sp.lambdify(x, sp.sympify(it['info']['trap_solution'].split('=', 1)[1].replace('^', '**')), 'numpy')
    yt, yg = f(xs) * np.ones_like(xs), g(xs) * np.ones_like(xs)
    ax.plot(xs, yt, color='#1f77b4', lw=2, label='true solution')
    ax.plot(xs, yg, color='#d62728', lw=2, ls='--', label='trap (z read as y)')
    for (pp, fn, col, mk) in [(pts, f, '#1f77b4', 'o'), (zpts, g, '#d62728', 's')]:
        for x0, v, d in pp:   # value condition: marker at (x0, v); slope condition: triangle + tangent
            if d == 0: ax.plot(x0, v, mk, color=col, ms=8, zorder=5)
            else:
                ax.plot(x0, fn(x0), '^', color=col, ms=8, zorder=5)
                ax.plot([x0 - .4, x0 + .4], [fn(x0) - .4 * v, fn(x0) + .4 * v], color=col, lw=1.5)
    q = np.nanpercentile(np.r_[yt, yg], [3, 97]); pad = (q[1] - q[0]) * .2 + 1
    ax.set_ylim(q[0] - pad, q[1] + pad)
    ax.set_title(f"{it['category']}: {it['equation']}\n" + ', '.join(c['text'] for c in it['conditions']), fontsize=8.5)
axes.flat[0].legend(frameon=False, fontsize=8)
fig.suptitle('unrelated subtype: true solution vs the z-trap (circle = y-condition, square = z-condition, triangle = slope condition)', fontsize=12)
fig.tight_layout(); save(fig, '06_trap'); plt.show()
""")

md(r"""
## 9. Field annotations for the interpretability analyses

Every problem stores character **spans** for its fields. Tokenizer offsets map them to tokens, so
token-level attributions or hidden states can be summed or pooled per field. For example: how much
attribution goes to the distractor condition, compared with the conditions that determine the answer?
Below, one prompt is coloured by field.
""")
code(r"""
FIELD_COL = {'instruction': '#e0e0e0', 'equation': '#c6dbef', 'sufficient': '#c7e9c0', 'distractor': '#fcbba1',
             'answer_instruction': '#fdf2c4'}
def spans_html(it):
    marks = [(*it['spans']['instruction'], 'instruction'), (*it['spans']['equation'], 'equation'),
             (*it['spans']['answer_instruction'], 'answer_instruction')]
    marks += [(s, e, c['role']) for (s, e), c in zip(it['spans']['conditions'], it['conditions'])]
    marks.sort(); p = it['prompt']; out, pos = [], 0
    esc = lambda t: t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    for s, e, lab in marks:
        out.append(esc(p[pos:s]))
        out.append(f'<span style="background:{FIELD_COL[lab]};padding:1px 2px;border-radius:3px" title="{lab}">{esc(p[s:e])}</span>')
        pos = e
    out.append(esc(p[pos:]))
    legend = ' '.join(f'<span style="background:{c};padding:2px 6px;border-radius:3px;margin-right:4px">{k}</span>'
                      for k, c in FIELD_COL.items())
    return (f'<div style="font-family:monospace;white-space:pre-wrap;color:#111;background:#fff;'
            f'border:1px solid #ccc;padding:10px;max-width:900px">{"".join(out)}</div><div style="margin-top:6px">{legend}</div>')
it = next(i for i in items if i['subtype'] == 'rescaled' and i['order'] == 2 and i['split'] == 'dev')
display(HTML(spans_html(it)))
print('spans:', json.dumps(it['spans']))
""")

md(r"""
## 10. Summary

| property | IVP-OPEN |
|---|---|
| size | 864 problems (dev 72, test 792) |
| problem types | 6 ODE types × 3 categories = 18, first and second order |
| condition settings | `none` (control), `duplicate`, `rescaled`, `unrelated` (with a trap answer) |
| balance | exactly 12 problems per category × subtype; distractor position balanced |
| answer format | open-ended, working allowed, final line `Answer: y = ...` |
| correctness oracle | exact construction + SymPy; independently verified (section 8) |
| annotations | condition roles, trap solutions, character spans per field |
| use in the thesis | dev: model shortlist and prompt; test: reported accuracy, labels (correct or wrong) for the attribution and representation analyses |

The dataset is a controlled test bed: the inputs are short and fully annotated, the labels are exact, and
the distractor conditions let us ask whether a model attends to what actually determines the answer.
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
    json.dump(to_nb(), open(sys.argv[1], 'w', encoding='utf-8'), indent=1)
