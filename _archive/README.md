# Archive (moved here 2026-11-09, nothing deleted)

Files that are no longer used by the current work. Their results stay in `results/` (unchanged). To restore a file,
move it back to the Thesis root (the paths below are where each file came from).

| folder | what | why archived |
|---|---|---|
| `mcq_baseline/` | MCQ-era notebooks and scripts: `baseline_colab.ipynb`, `diagnose_colab.ipynb`, `diagnose_llama_run.ipynb`, `permutations_colab.ipynb`, `option_permutations.py`, `analyze_baseline.py`, `REPORT_2026-11-03_baseline.md` | MCQ letter scoring is retired (decision 2026-11-03). Results stay in `results/baseline/`, `results/option_permutations/`, `results/analysis/`, `results/CANONICAL.json`. `baseline_letter_logprobs.py` stays in the root (other code imports it). |
| `easy_controls/` | easy-arithmetic control experiments: `easy_control_colab.ipynb`, `easy_control_kaggle.ipynb`, `easy_open_kaggle.ipynb`, `open_arithmetic.py`, `gen_easy_mcq.py`, `gen_easy_open.py`, their Kaggle upload zips | Done on 2026-11-03 (evidence for the answer-format decision). Datasets stay in `easy_mcq/` and `easy_open/`; results in `results/easy_control/` and `results/easy_open/`. |
| `downloaded_result_zips/` | the zips downloaded from Kaggle: `attribution_results.zip`, `attribution_results_math.zip`, `easy_control_results.zip`, `easy_open_results.zip`, `open_eval_results.zip`, `open_eval_test_results.zip` | Already unpacked into `results/`. Kept because a few files differ from or are missing in `results/` (Kaggle figure outputs, GPU logs, the original `summary.json` of four attribution runs). |
| `superseded_attribution/` | `attribution_results_summary.ipynb`, `make_attribution_summary_nb.py`, and the earlier attribution figure folders (`figures/20261104_test`, `20261106_test_all3`, `20261106_test_all3_main`, `summary_notebook`) | Superseded by `attribution_results_kaggle.ipynb` (target (a), attention only; decision 2026-11-09). The earlier figures include target (c) and G×I. |

Note: scripts moved here import modules from the Thesis root (e.g. `baseline_letter_logprobs`), so run them from
the root after moving them back.
