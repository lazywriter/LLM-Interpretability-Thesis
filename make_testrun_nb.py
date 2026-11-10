"""Builds open_eval_test_kaggle.ipynb: the shortlisted models on the IVP-OPEN test split.

  python make_testrun_nb.py open_eval_test_kaggle.ipynb
"""
import json, sys

CELLS = []
def md(s): CELLS.append(('md', s.strip('\n')))
def code(s): CELLS.append(('code', s.strip('\n')))

md(r"""
# Shortlisted models on the IVP-OPEN test split (Kaggle, 2 × T4)

Phi-4-mini-instruct, Qwen2.5-Math-1.5B-Instruct and Qwen3-4B-Instruct-2507 solve all 792 test problems with working
allowed (greedy, **up to 1536 new tokens**). Every reply is graded and saved together with its exact generated
token ids. These generations are the input for attribution target (a): the model's own answer, teacher-forced on
its own working.

**Before running** (right-hand panel):
- *Session options → Accelerator*: **GPU T4 x2**. The two GPUs run two models at the same time. With one GPU
  everything still runs, one model after another.
- *Session options → Internet*: **on**
- *Input*: the dataset made from `kaggle_open_eval_test_files.zip`

**How to run.** The run takes about 3–4 hours on 2 × T4, so use **Save Version → Save & Run All (Commit)**.
That runs in the background, so you can close the browser, and keeps `/kaggle/working` as the version's output.
When it finishes, download `open_eval_test_results.zip` from the version's *Output* tab.

If a model fails (for example out of memory), the run cell prints it. Re-running that cell continues every
unfinished model where it stopped (`--resume`) and skips finished ones. If the whole session was lost, add the
earlier run's output as an Input: see "Continue from an earlier run" below.
""")

code(r"""
import os, sys, glob, json, math, shutil, subprocess, time
INPUT = os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input')
WORK = os.environ.get('KAGGLE_WORK_DIR', '/kaggle/working/Thesis')
found = glob.glob(os.path.join(INPUT, '**', 'eval_open_ivp.py'), recursive=True)
assert found, f'project files not found under {INPUT}: add the dataset as an Input of this notebook'
shutil.copytree(os.path.dirname(found[0]), WORK, dirs_exist_ok=True)
os.chdir(WORK)
for f in ['eval_open_ivp.py', 'baseline_letter_logprobs.py', 'field_tokens.py', 'ivp_open/grade_open.py',
          'ivp_open/ivp_open_dev.jsonl', 'ivp_open/ivp_open_test.jsonl', 'ivp_open/ivp_open_all.jsonl', 'ivp_mcq/grade_mcq.py']:
    assert os.path.exists(f), f'missing {f} in the uploaded dataset (keep the folder structure of the zip)'
ON_KAGGLE = INPUT == '/kaggle/input'
if ON_KAGGLE:
    r = subprocess.run('curl -s -o /dev/null -w "%{http_code}" https://huggingface.co', shell=True, capture_output=True, text=True)
    assert r.stdout.strip() in ('200', '301', '302'), 'no internet: turn Internet on in Session options'
    subprocess.run('pip -q install "transformers>=4.51" accelerate', shell=True)
import torch, transformers
N_GPU = torch.cuda.device_count()
if ON_KAGGLE:
    assert N_GPU, 'no GPU: set Accelerator to GPU T4 x2 in Session options'
print('torch', torch.__version__, '| transformers', transformers.__version__, '| GPUs:',
      [torch.cuda.get_device_name(i) for i in range(N_GPU)] or 'none')
""")

md("## Settings")
code(r"""
SPLIT = 'test'
# one queue per GPU, balanced by the dev run times (Qwen3-4B alone ~ Phi + Qwen2.5-Math)
QUEUES = [['qwen3-4b-instruct-2507'],
          ['phi-4-mini-instruct', 'qwen2.5-math-1.5b-instruct']]
BATCH = {'qwen3-4b-instruct-2507': 8, 'phi-4-mini-instruct': 8, 'qwen2.5-math-1.5b-instruct': 16}
MAX_NEW_TOKENS = 1536
LIMIT = None        # e.g. 16 for a quick trial; None for the real run
OUT = 'results/open_eval'
if N_GPU < 2:       # one GPU (or a CPU trial): a single queue
    QUEUES = [[m for q in QUEUES for m in q]]
""")

md(r"""
## Check the field → token mapping for each tokenizer
Every prompt field (instruction, equation, each condition, answer instruction) must map to its own tokens inside
the chat template. The attribution step relies on this. It runs on the CPU in about a minute.
""")
code(r"""
for m in [m for q in QUEUES for m in q]:
    r = subprocess.run(f'{sys.executable} field_tokens.py --model {m}', shell=True, capture_output=True, text=True)
    print((r.stdout.strip() or r.stderr.strip()[-800:]), '\n')
    assert r.returncode == 0, f'field mapping failed for {m}'
""")

md(r"""
## Continue from an earlier, interrupted run (optional)
If an earlier session stopped part-way, add its output, `open_eval_test_results.zip` or the failed version's output,
as an extra Input. Its runs are copied back here, so finished models are skipped and unfinished ones continue
(`--resume`) instead of starting again.
""")
code(r"""
prev = [os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, '**', 'open_eval', f'*_{SPLIT}', 'generations.jsonl'),
                                              recursive=True)]
for d in prev:
    dst = os.path.join(OUT, os.path.basename(d))
    if not os.path.exists(dst):
        shutil.copytree(d, dst)
    n = sum(1 for _ in open(os.path.join(dst, 'generations.jsonl'), encoding='utf-8'))
    print(f"{os.path.basename(d)}: {n} replies, {'finished' if os.path.exists(os.path.join(dst, 'summary.json')) else 'unfinished: will resume'}")
print('no earlier runs in the inputs' if not prev else '')
""")

md("## Run (two GPUs in parallel)")
code(r"""
def run_dirs(model):
    return sorted(glob.glob(os.path.join(OUT, f'*_{model}_{SPLIT}')))

def command(model):
    base = (f'{sys.executable} eval_open_ivp.py --model {model} --split {SPLIT} --max-new-tokens {MAX_NEW_TOKENS} '
            f'--batch-size {BATCH[model]} --outdir {OUT}' + (f' --limit {LIMIT}' if LIMIT else ''))
    dirs = run_dirs(model)
    if dirs and os.path.exists(os.path.join(dirs[-1], 'summary.json')):
        return None                                  # finished earlier in this session
    return base + (f' --resume {dirs[-1]}' if dirs else '')

os.makedirs('logs', exist_ok=True)
procs = []
for g, queue in enumerate(QUEUES):
    cmds = [(m, c) for m in queue if (c := command(m))]
    if not cmds:
        continue
    driver = ('import subprocess, sys\n'   # runs this GPU's models one after another
              f'for m, c in {cmds!r}:\n'
              '    print("=== " + m, flush=True)\n'
              '    r = subprocess.run(c, shell=True)\n'
              '    print(f"=== exit {r.returncode} {m}", flush=True)\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g) if N_GPU else '')
    log = open(f'logs/gpu{g}.log', 'a')
    procs.append((g, subprocess.Popen([sys.executable, '-u', '-c', driver], env=env, stdout=log,
                                      stderr=subprocess.STDOUT), log))
    print(f'GPU {g}:', [m for m, c in cmds if c])

def last_progress(g):
    t = open(f'logs/gpu{g}.log', errors='replace').read().replace('\r', '\n').strip().splitlines()
    heads = [l for l in t if l.startswith('=== ') and 'exit' not in l]
    prog = [l for l in t if '/' in l and l.split('/')[0].strip().isdigit()]
    return f"{heads[-1][4:] if heads else ''} {prog[-1] if prog else ''}"

t0 = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(300 if ON_KAGGLE else 5)
    print(f'[{(time.time() - t0) / 60:5.0f} min] ' + ' | '.join(f'GPU {g}: {last_progress(g)}' for g, _, _ in procs), flush=True)
for g, p, log in procs:
    log.close()
    print(f'\n----- GPU {g} log (end) -----')
    t = open(f'logs/gpu{g}.log', errors='replace').read().replace('\r', '\n').splitlines()
    print('\n'.join(l for l in t if 'acc=' in l or l.startswith('===') or 'Error' in l or 'error' in l.lower()[:40])[-3000:])
""")

md("## Results")
code(r"""
models = [m for q in QUEUES for m in q]
done = [d for m in models for d in run_dirs(m)[-1:] if os.path.exists(os.path.join(d, 'summary.json'))]
missing = [m for m in models if not any(f'_{m}_' in d for d in done)]
print('finished:', [os.path.basename(d) for d in done], '| not finished:', missing or 'none')
if done:
    r = subprocess.run(f'{sys.executable} eval_open_ivp.py --table ' + ' '.join(f'"{d}/summary.json"' for d in done),
                       shell=True, capture_output=True, text=True)
    print(r.stdout)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)
print('\naccuracy with 95% confidence intervals')
for d in done:
    s = json.load(open(f'{d}/summary.json')); n = s['meta']['n_items']; k = round(s['accuracy'] * n)
    lo, hi = wilson(k, n)
    print(f"  {s['meta']['model_key']:30s} {100 * s['accuracy']:5.1f}%  [{lo:4.1f}, {hi:4.1f}]  ({k}/{n})  "
          f"truncated {s['truncated']}  mean tokens {s['mean_new_tokens']:.0f}  {s['meta']['seconds'] / 60:.0f} min")
""")

md("## Download")
code(r"""
zip_base = os.path.join(os.path.dirname(WORK), 'open_eval_test_results')
shutil.copytree('logs', os.path.join(OUT, '_logs'), dirs_exist_ok=True)
path = shutil.make_archive(zip_base, 'zip', root_dir=WORK, base_dir=OUT)
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
    json.dump(to_nb(), open(sys.argv[1], 'w', encoding='utf-8'), indent=1)
