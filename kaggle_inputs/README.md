# Kaggle inputs

Upload a zip as a Kaggle dataset (Kaggle unpacks it) and add it as an Input of the matching notebook.
The notebooks themselves are in the Thesis root.

| zip | notebook | what it contains |
|---|---|---|
| `kaggle_open_eval_files.zip` | `open_eval_kaggle.ipynb` | accuracy run on the dev split: `eval_open_ivp.py`, model wiring, grader, IVP-OPEN dev/test |
| `kaggle_open_eval_test_files.zip` | `open_eval_test_kaggle.ipynb` | accuracy run on the test split (1536 new tokens, 2 GPUs) |
| `kaggle_attribution_files.zip` | `attribution_kaggle.ipynb`, `attribution_math_kaggle.ipynb` | attribution code + data (GPU) |
| `open_eval_test_regraded.zip` | `attribution_kaggle.ipynb`, `attribution_math_kaggle.ipynb` (second input) | the regraded test generations, needed for target (a) |
| `kaggle_attribution_results_files.zip` | `attribution_results_kaggle.ipynb` | report code + the three target-(a) attribution runs (CPU) |
| `kaggle_dataset_overview_files.zip` | `dataset_overview_kaggle.ipynb` | IVP-OPEN dataset overview (CPU) |
