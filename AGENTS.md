# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a **Data Science and MLOps Learning Resources** repository — a collection of educational Jupyter notebooks, CSV datasets, and reference documents. There is no deployable application, API, or service. The primary workflow is running Jupyter notebooks interactively.

### Running the environment

- **Jupyter Notebook**: `jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''` from the repo root.
- Notebooks can also be executed in batch via `jupyter nbconvert --to notebook --execute <notebook.ipynb>`.

### Key directories

| Directory | Contents |
|---|---|
| `Python_Introduction/` | Python basics notebooks and a sample package |
| `DataWrangling/` | NumPy, Pandas, PySpark notebooks + exercises |
| `Visualization/` | Matplotlib and Seaborn notebooks |
| `ConventionalMachineLearning/` | Association rules, K-means notebooks |
| `DSLC/` | Data Science Life Cycle notebooks (end-to-end ML, AutoKeras, templates) |
| `NLP/` | Transformer-related notebooks |
| `datasets/` | CSV datasets used by notebooks |

### Linting

Run `flake8` on the Python source files (there are only 3):
```
flake8 Python_Introduction/my_package/ Python_Introduction/exercise_solution.py --max-line-length=120
```

### Gotchas

- Some notebooks use **deprecated API calls** (e.g. `fillna(method='ffill')` in pandas, `seaborn-whitegrid` style name in matplotlib). These will error with current library versions. This is pre-existing in the repo, not a setup issue.
- The `Python_Basics.ipynb` notebook calls `input()`, so it cannot be executed headlessly via `nbconvert`.
- Heavy/optional notebooks (PySpark, TensorFlow/AutoKeras, Transformers) require additional dependencies not installed by default (`pyspark`, `tensorflow`, `autokeras`, `torch`, `transformers`). Install them only if those specific notebooks need to run.
- `$HOME/.local/bin` must be on `PATH` for pip-installed CLI tools like `jupyter`, `flake8`, etc.
