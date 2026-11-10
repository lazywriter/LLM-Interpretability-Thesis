"""Builds attribution_kaggle.ipynb: Gradient×Input, IG and attention for the shortlist.

  python make_attribution_nb.py attribution_kaggle.ipynb
  python make_attribution_nb.py attribution_math_kaggle.ipynb --only qwen2.5-math-1.5b-instruct
      (one model: target c on GPU 0, target a on GPU 1)
"""
import json, sys

ONLY = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None

CELLS = []
def md(s): CELLS.append(('md', s.strip('\n')))
def code(s): CELLS.append(('code', s.strip('\n')))

md(r"""
# Attribution on IVP-OPEN: Gradient×Input, Integrated Gradients, attention (Kaggle, 2 × T4)

Field-level attribution for the shortlisted models: how much of the target's log-probability goes to the
equation, to each condition (sufficient vs distractor), and to the instructions. Only gradient and attention
methods are used; perturbation methods (LIME, SHAP, occlusion) are excluded on purpose (see `CLAUDE.md`).

- **Target (a)** (main): the model's own final answer, teacher-forced on its own working, taken from the test-run
  generations. Only replies that finished and contain an answer are used.
- **Target (c)** (control): the true solution, right after the prompt.

**Before running** (right-hand panel):
- *Accelerator*: **GPU T4 x2**; *Internet*: **on**
- *Input* 1: the dataset made from `kaggle_attribution_files.zip`
- *Input* 2: a dataset made from **`open_eval_test_regraded.zip`** (the test generations, regraded with the current
  grader). It is needed for target (a) and gives the correct/wrong labels.

Run with **Save Version → Save & Run All (Commit)** and download `attribution_results.zip` from the version's
Output tab. Re-running the run cell skips the runs that already finished in this session.
""")

code(r"""
import os, sys, glob, json, shutil, subprocess, time
INPUT = os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input')
WORK = os.environ.get('KAGGLE_WORK_DIR', '/kaggle/working/Thesis')
found = glob.glob(os.path.join(INPUT, '**', 'attribution.py'), recursive=True)
assert found, f'project files not found under {INPUT}: add the dataset as an Input of this notebook'
shutil.copytree(os.path.dirname(found[0]), WORK, dirs_exist_ok=True)
os.chdir(WORK)
for f in ['attribution.py', 'attribution_summary.py', 'field_tokens.py', 'baseline_letter_logprobs.py',
          'ivp_open/grade_open.py', 'ivp_open/ivp_open_dev.jsonl', 'ivp_open/ivp_open_test.jsonl', 'ivp_mcq/grade_mcq.py']:
    assert os.path.exists(f), f'missing {f} in the uploaded dataset (keep the folder structure of the zip)'
ON_KAGGLE = INPUT == '/kaggle/input'
if ON_KAGGLE:
    r = subprocess.run('curl -s -o /dev/null -w "%{http_code}" https://huggingface.co', shell=True, capture_output=True, text=True)
    assert r.stdout.strip() in ('200', '301', '302'), 'no internet: turn Internet on in Session options'
    subprocess.run('pip -q install "transformers>=4.51" accelerate', shell=True)
import torch, transformers
N_GPU = torch.cuda.device_count()
if ON_KAGGLE:
    assert N_GPU, 'no GPU: set Accelerator to GPU T4 x2'
print('torch', torch.__version__, '| transformers', transformers.__version__, '| GPUs:',
      [torch.cuda.get_device_name(i) for i in range(N_GPU)] or 'none')

# test-run generations (target a): the newest finished run per model
GENS = {}
for s in sorted(glob.glob(os.path.join(INPUT, '**', 'open_eval', '*_test*', 'summary.json'), recursive=True)):  # a regraded run sorts last and wins
    GENS[json.load(open(s))['meta']['model_key']] = os.path.dirname(s)
print('generations found for:', {k: os.path.basename(v) for k, v in GENS.items()} or 'none (target a will be skipped)')
""")

md(r"""
## Settings
`PER_CELL` items per category × subtype (72 cells). The test split has 11 per cell, so `PER_CELL = 2` gives 144
problems per model and target. Target (a) only uses replies that finished with an answer, so it has fewer.
""")
code(r"""
SPLIT = 'test'
PER_CELL = {'a': 2, 'c': 2}
IG_STEPS = {'a': 32, 'c': 128}      # midpoint steps; (a) inputs are long (~1000 tokens), so fewer steps
IG_BATCH = {'a': 1, 'c': 16}        # steps per forward/backward pass
QUEUES = [[('qwen3-4b-instruct-2507', 'c'), ('qwen3-4b-instruct-2507', 'a')],
          [('qwen2.5-math-1.5b-instruct', 'c'), ('phi-4-mini-instruct', 'c'),
           ('qwen2.5-math-1.5b-instruct', 'a'), ('phi-4-mini-instruct', 'a')]]
EXTRA = {'c': '', 'a': ''}          # extra attribution.py flags per target
LIMIT = None                        # e.g. 3 for a quick trial
OUT = 'results/attribution'
if N_GPU < 2:
    QUEUES = [[r for q in QUEUES for r in q]]
""")

md("## Run (two GPUs in parallel)")
code(r"""
def run_dirs(model, target):
    return sorted(glob.glob(os.path.join(OUT, f'*_{model}_{target}_{SPLIT}')))

def command(model, target):
    if any(os.path.exists(os.path.join(d, 'meta.json')) for d in run_dirs(model, target)):
        return None                                   # finished earlier in this session
    if target == 'a' and model not in GENS:
        print(f'  skip {model} target a: no test generations in the inputs')
        return None
    c = (f'{sys.executable} attribution.py --model {model} --target {target} --split {SPLIT} '
         f'--per-cell {PER_CELL[target]} --ig-steps {IG_STEPS[target]} --ig-batch {IG_BATCH[target]} --outdir {OUT}')
    if model in GENS:
        c += f' --gens "{GENS[model]}"'
    return c + (f' --limit {LIMIT}' if LIMIT else '') + (' ' + EXTRA[target] if EXTRA[target] else '')

os.makedirs('logs', exist_ok=True)
procs = []
for g, queue in enumerate(QUEUES):
    cmds = [(f'{m} {t}', c) for m, t in queue if (c := command(m, t))]
    if not cmds:
        continue
    driver = ('import subprocess\n'
              f'for name, c in {cmds!r}:\n'
              '    print("=== " + name, flush=True)\n'
              '    r = subprocess.run(c, shell=True)\n'
              '    print(f"=== exit {r.returncode} {name}", flush=True)\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g) if N_GPU else '')
    log = open(f'logs/attr_gpu{g}.log', 'a')
    procs.append((g, subprocess.Popen([sys.executable, '-u', '-c', driver], env=env, stdout=log,
                                      stderr=subprocess.STDOUT), log))
    print(f'GPU {g}:', [n for n, _ in cmds])

def last_progress(g):
    t = open(f'logs/attr_gpu{g}.log', errors='replace').read().replace('\r', '\n').strip().splitlines()
    heads = [i for i, l in enumerate(t) if l.startswith('=== ') and 'exit' not in l]
    if not heads:
        return 'starting'
    cur = t[heads[-1]:]                               # lines of the run in progress (or the last one)
    prog = [l for l in cur if '/' in l and l.split('/')[0].strip().isdigit()]
    oom = sum('out of memory' in l for l in cur)
    ended = [l for l in cur if l.startswith('=== exit')]
    s = f"{t[heads[-1]][4:]} {prog[-1] if prog else '(no item finished yet)'}"
    s += f' | OOM x{oom}' if oom else ''
    if ended:                                         # finished or crashed: show the last message
        msg = [l for l in cur if l.strip() and not l.startswith('===')]
        s += f" | {ended[-1][4:]}: {msg[-1][:150] if msg else ''}"
    return s

t0 = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(300 if ON_KAGGLE else 5)
    print(f'[{(time.time() - t0) / 60:5.0f} min] ' + ' | '.join(f'GPU {g}: {last_progress(g)}' for g, _, _ in procs), flush=True)
for g, p, log in procs:
    log.close()
    t = open(f'logs/attr_gpu{g}.log', errors='replace').read().replace('\r', '\n').splitlines()
    print(f'\n----- GPU {g} -----')
    print('\n'.join(l for l in t if l.startswith('===') or ' done ' in l or 'Error' in l or 'out of memory' in l)[-3000:])
""")

md(r"""
## Summary per model and target
**Density**: a field's share of the attribution divided by its share of the prompt tokens (1 = in proportion
to its length). **distr/suff** < 1 means the distractor condition gets less than the conditions that determine
the answer. **dup 2nd/1st** compares the two identical copies in `duplicate` items, so any difference is a
position effect. **working**: the part of the total that goes to the model's own working (target a).
""")
code(r"""
done = [d for d in sorted(glob.glob(os.path.join(OUT, '*'))) if os.path.exists(os.path.join(d, 'meta.json'))]
r = subprocess.run(f'{sys.executable} attribution_summary.py ' + ' '.join(f'"{d}"' for d in done),
                   shell=True, capture_output=True, text=True)
print(r.stdout[-20000:] or r.stderr[-3000:])
""")

md(r"""
## Figures
Medians of per-item log ratios with bootstrap 95% CIs (`attribution_plots.py`). IG points are hollow when the
run's median completeness gap is above 10%.
""")
code(r"""
subprocess.run('pip -q install matplotlib', shell=True)
from IPython.display import Image, display
FIG = os.path.join(OUT, '_figures')
r = subprocess.run(f'{sys.executable} attribution_plots.py ' + ' '.join(f'"{d}"' for d in done) + f' --out "{FIG}"',
                   shell=True, capture_output=True, text=True)
print(r.stdout[-3000:] or r.stderr[-3000:])
for f in sorted(glob.glob(os.path.join(FIG, '*.png'))):
    print(os.path.basename(f))
    display(Image(filename=f))
""")

md("## Download")
code(r"""
shutil.copytree('logs', os.path.join(OUT, '_logs'), dirs_exist_ok=True)
path = shutil.make_archive(os.path.join(os.path.dirname(WORK), 'attribution_results'), 'zip', root_dir=WORK, base_dir=OUT)
print('wrote', path, f'({os.path.getsize(path) / 1e6:.1f} MB): download it from the Output tab')
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
    if ONLY:
        for i, (kind, src) in enumerate(CELLS):
            if kind == 'code' and src.startswith('SPLIT ='):
                src = src.replace(src[src.index('QUEUES = '):src.index('LIMIT = ')],
                                  f"QUEUES = [[({ONLY!r}, 'c')], [({ONLY!r}, 'a')]]\n"
                                  "EXTRA = {'c': '', 'a': ''}   # keep checkpointing: fp32 without it runs out of memory\n")
                CELLS[i] = (kind, src)
    json.dump(to_nb(), open(sys.argv[1], 'w', encoding='utf-8'), indent=1)
