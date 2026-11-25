# Experiments: One-Step RL and Critic Regularization Coincide in Stochastic Environments

This folder contains all the Python code behind the experiments and figures in the paper
*One-Step RL and Critic Regularization Coincide in Stochastic Environments: A Target-Linear Analysis of Classifier Actor–Critic* (Akhmitzyanov R.).

## Files

| File | What it does |
|---|---|
| `common.py` | Shared code: exact value and fixed-point computations, the random MDPs and the slippery gridworld, and offline dataset collection |
| `e1_sweep.py` | Experiment E1: random MDPs, sweeping transition stochasticity and reward scale (Fig. 1, Table 1) |
| `e2_policy.py` | Experiment E2: maximises the critic-regularised objective in the gridworld and compares the result with one-step RL (Fig. 2, Table 2) |
| `e3_sgd.py` | Experiment E3: single-sample SGD training of the critic (Fig. 3) |
| `e3_ablation.py` | E3 step-size ablation (table in Fig. 3) |
| `make_figs.py` | Builds all three figures from the saved results |

Saved results from the paper's run are included: `e1.json`, `e2.json`, `e3.json` and `abl.json`.

## Requirements

Python 3 with NumPy, SciPy and Matplotlib:

```bash
pip install numpy scipy matplotlib
```

## Reproducing the results

Run from inside this folder:

```bash
python3 e1_sweep.py .
python3 e2_policy.py .
python3 e3_sgd.py .
python3 e3_ablation.py .
python3 make_figs.py .
```

E3 takes about 2 minutes; E1 and E2 take a few minutes each.

The experiment scripts write `e1.json`, `e2.json` and `e3.json` to the current folder. `e3_ablation.py` prints its results; the paper's values are saved in `abl.json`. `make_figs.py` writes `fig_e1.pdf`, `fig_e2.pdf` and `fig_e3.pdf` to the parent folder, next to `main.tex`. To rebuild the figures from the saved results without rerunning the experiments, run only `make_figs.py`.
