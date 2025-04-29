[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![code style: Ruff](
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](
    https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

# EngiOpt

This repository contains the code for optimization and machine learning algorithms for engineering design problems. Our goal here is to provide clean example usage of EngiBench and provide strong baselines for future comparisons.

## Coding Philosophy
As much as we can, we follow the [CleanRL](https://github.com/vwxyzjn/cleanrl) philosophy: single-file, high-quality implementations with research-friendly features:
* Single-file implementation: every detail is in one file, so you can easily understand and modify the code.
* High-quality: we use type hints, docstrings, and comments to make the code easy to understand.
* Logging: we use experiment tracking tools like [Weights & Biases](https://wandb.ai/site) to log the results of our experiments.
* Reproducibility: we pin all the versions used in the experiment to ensure reproducibility. We also seed all the random number generators.

## Install
1. Install EngiBench from source:
```
git clone git@github.com:IDEALLab/EngiBench.git
cd engibench
pip install -e ".[all]"
```
2. Install EngiOpt dependencies:
```
cd ../engiopt
pip install -e .
```

## Dashboards
The integration with WandB allows us to access live dashboards of our runs (on the cluster or not). We also upload the trained models there. You can access some of our runs at https://wandb.ai/engibench/engiopt.
<img src="imgs/wandb_dasboard.png" alt="WandB dashboards"/>




## Wishlist
see https://github.com/IDEALLab/EngiBench/issues/4
