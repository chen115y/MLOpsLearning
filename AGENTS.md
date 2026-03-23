# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a **static educational repository** of Jupyter notebooks and datasets for learning data science, machine learning, and MLOps. There are no running services, APIs, or build systems. The "application" is JupyterLab serving the notebooks.

### Running the application

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token='' --allow-root
```

JupyterLab will be available at `http://localhost:8888/lab`.

### Running notebooks headlessly

Use `jupyter nbconvert` to execute notebooks without a browser:

```bash
jupyter nbconvert --to notebook --execute <notebook_path> --output /tmp/output.ipynb
```

### Known compatibility issues

Some notebooks contain calls to APIs that were deprecated or removed in newer versions of pandas/matplotlib/seaborn. For example:
- `DataFrame.fillna(method='ffill')` was removed in pandas 3.x (use `DataFrame.ffill()` instead)
- `plt.style.use('seaborn-whitegrid')` was renamed in matplotlib 3.x (use `seaborn-v0_8-whitegrid`)

These are not environment issues — the notebooks were written for older library versions.

### Notebooks confirmed to run cleanly

- `DataWrangling/Numpy.ipynb`
- `DataWrangling/Exercise.ipynb`
- `ConventionalMachineLearning/plot_kmeans_digits.ipynb`

### Key dependencies

All installed via pip: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `jupyter`, `jupyterlab`, `ipykernel`, `nbformat`, `nbconvert`.

### No lint/test/build system

This repository has no linter configuration, no automated test suite, and no build system. Quality checks are limited to verifying notebooks execute without errors.
