# Counterfactual Residual Data Augmentation for Regression (CRDA)

Official code for the paper **"Counterfactual Residual Data Augmentation for Regression"**, published at the **43rd International Conference on Machine Learning (ICML 2026)**.

🌐 **Project page:** <https://crda-project.github.io>

📦 **Python package:** [`pip install crda`](https://pypi.org/project/crda/) — a reference implementation for easy use of CRDA as a tool.

CRDA is a **model-agnostic** data augmentation technique for **tabular regression** under data scarcity. Its key insight: once a regressor has captured the systematic component of the data, the remaining noise can be treated as an **invariant residual** that stays stable under small perturbations of carefully selected features. CRDA reuses that residual to synthesize new, realistic training samples — expanding the dataset without collecting more real data.

On average, CRDA reduces an **MLP Regressor's MSE by 22.9%** and an **XGBoost Regressor's MSE by 6.4%** across nine benchmark datasets, and consistently outperforms state-of-the-art tabular data generators and augmentation methods.

---

## 🔍 How CRDA Works

Given a dataset and a base regressor `g(·)` (e.g. MLP or XGBoost), CRDA:

1. **Trains the baseline** `g` and computes the **residual** `zᵢ = yᵢ − g(xᵢ)` for each training point.
2. **Selects perturbable features** via a two-stage independence screen against the residual `z`:
   - **PC algorithm** (Peter–Clark) — drops features with a direct causal edge to `z`.
   - **Pearson correlation check** — drops features strongly correlated with `z`.
   - The surviving features form the perturbable set `X_P`; the rest are held fixed (`X_R`).
3. **Generates counterfactual samples** by multiplicatively perturbing the selected features, `x′_P = x_P · (1 + δ)` with `δ ~ Unif[−p, p]`, and **reusing the original residual**: `y′ = g(x′) + z`. Preserving `z` keeps the instance-specific noise structure intact.
4. **Validates with a safety gate** — a Wilcoxon signed-rank test over K-fold CV compares augmented vs. unaugmented models. CRDA retrains on the augmented data **only if** the improvement is statistically significant (p < 0.05); otherwise it falls back to the untouched baseline.

This combination of residual reuse + independence screening + a statistical safety gate is what distinguishes CRDA from naive feature perturbation or generative augmentation.

---

## 📁 Repository Structure

| Path | Description | Paper artifact |
|------|-------------|----------------|
| `src/` | CRDA implementation (baseline, residual filter, PC/causal screen, experiment harness) | — |
| `data/` | The 9 benchmark datasets + synthetic ANM data (`anm_data.csv`, generator `anm_gen.py`) | Table 3 |
| `demo/` | `demonstration.ipynb` — ~1-minute interactive walkthrough | — |
| `experiments/` | Main MLP & XGBoost results; `full_reproduction.ipynb` | **Tables 1 & 12, Figure 2** |
| `experiments_all_baselines/` | CRDA vs. C-Mixup, ADA, TabDDPM, TVAE, CTGAN | **Table 2** |
| `experiments_synthetic/` | Synthetic sample-size-scaling study | **Figure 3** |
| `experiments_divergence/` | Mutual-information validation of residual independence | **Table 8** |
| `experiments_catboost/` | CatBoost base-regressor study | **Table 10** |
| `experiments_linreg/` | Linear-regression (weak base learner) study | **Table 11** |
| `scripts/` | Result-aggregation and figure-generation scripts | Figures 5–8 |

Each experiment run is saved to a timestamped directory:

```
experiments/<dataset>/<baseline>_<timestamp>/
├── config.json              # full experiment configuration
├── results.csv              # aggregated metrics (mean ± SEM across seeds)
├── interim_results/         # per-seed raw results
├── models/                  # trained model artifacts (only if save_models=True)
└── params/                  # tuned hyperparameters (only if save_params=True)
```

---

## ⚙️ Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
```

**System dependency** — Graphviz (for causal-graph visualization):

```bash
# macOS
brew install graphviz
# Ubuntu/Debian
sudo apt-get install graphviz
```

**Environment**
- **Python**: 3.11 (tested)
- **OS**: macOS, Linux, Windows
- Core packages: `scikit-learn`, `torch`, `xgboost`, `catboost`, `causal-learn` (PC algorithm), `optuna` (hyperparameter tuning), `sdv` (CTGAN/TVAE baselines), `pandas`, `numpy`, `matplotlib`. Full pinned list in [`requirements.txt`](requirements.txt).

---

## 🚀 Quick Start

```bash
cd demo/
jupyter notebook demonstration.ipynb
```

The demo runs the full CRDA pipeline on the Energy Efficiency dataset with visualizations in ~1 minute — a good way to verify your install and understand the method end-to-end.

---

## 🛠️ Configuration

Every run is driven by a `Config` object (see **[`src/utils/config.py`](src/utils/config.py)** for the full list, defaults, and inline docs). The main CRDA knobs:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `baseline` | `"mlp"` | Base regressor: `"mlp"` or `"xgboost"` |
| `aug_data_size_factor` | `1.0` | How many counterfactual samples to generate, relative to the training set (`1.0` = doubles it) |
| `max_n_features_to_perturb` | `5` | Max number of selected features perturbed per sample |
| `max_perturb_percent` / `min_perturb_percent` | `0.1` / `-0.1` | Perturbation range `p`: each feature is scaled by `1 + δ`, `δ ~ Unif[min, max]` |
| `alpha` | `0.05` | Significance level for the PC algorithm (causal screen) |
| `r_corr_threshold` / `p_corr_threshold` | `0.1` / `0.05` | Pearson correlation/p-value thresholds for the residual-correlation screen |
| `p_wilcoxon_threshold` | `0.05` | Significance threshold for the Wilcoxon safety gate |
| `ignore_filter` | `False` | If `True`, skip the independence screen and always augment |
| `hyperparam_tune` | `False` | If `True`, tune base-model + CRDA knobs (RandomizedSearchCV + Optuna) |
| `num_seeds` | `15` | Number of seeds averaged per run |
| `sample_sizes` | `None` | List of training-subset sizes to sweep (`None` = full dataset) |

Example:

```python
from src.utils.config import Config
from src.utils.logger import Logger
from src.experiment import Experiment

config = Config(
    dataset_path="./data/HousePrice.csv",
    baseline="xgboost",
    aug_data_size_factor=1.25,
    max_n_features_to_perturb=2,
    max_perturb_percent=0.7,
    min_perturb_percent=-0.7,
    results_dir="./runs",
)
Experiment(config, Logger()).run()
```

See `src/utils/config.py` for the remaining options (output toggles such as `save_models`, `save_plots`, `test_size`, etc.).

---

## 📊 Datasets

All nine datasets are preprocessed and ready to use in `./data/`. Each is a CSV where the **last column is the continuous target** and all other columns are **numerical features** (no missing values, no categorical columns). Provenance follows the paper (Appendix C, Table 3).

| Dataset | Samples | Features | Source | File |
|---------|--------:|---------:|--------|------|
| CPU Performance | 8192 | 12 | PMLB | `227_cpu_small.csv` |
| Satellite Image | 6435 | 36 | PMLB | `294_satellite_image.csv` |
| Wind Power | 6574 | 14 | UCI | `503_wind.csv` |
| Synthetic Regression | 1000 | 10 | PMLB | `623_fri_c4_1000_10.csv` |
| Concrete Strength | 1005 | 8 | UCI | `ConcreteCompressiveStrength.csv` |
| Energy Efficiency | 768 | 9 | UCI | `EnergyEfficiency.csv` |
| House Price | 1000 | 7 | Kaggle | `HousePrice.csv` |
| Parkinson's Monitoring | 5875 | 20 | UCI | `ParkinsonsTelemonitoring.csv` |
| Wine Quality | 5318 | 11 | UCI | `WineQuality.csv` |

The synthetic study (Figure 3) uses an additive-noise DGP `Y = X₁² + X₂X₃ + Z`, with `Z ⊥ (X₁, X₂, X₃)`, generated by `data/anm_gen.py` (cached as `data/anm_data.csv`).

---

## 📈 Reproducing the Paper

### Main results
Run the full reproduction notebook for the headline MLP/XGBoost results (Tables 1 & 12, Figure 2):

```bash
jupyter notebook experiments/full_reproduction.ipynb
```

Other experiments each have a notebook with an intro cell describing what it produces:

| Notebook | Produces |
|----------|----------|
| `experiments/full_reproduction.ipynb` | Tables 1 & 12, Figure 2 |
| `experiments_all_baselines/aug_data_all_baselines.ipynb` | Table 2 |
| `experiments_synthetic/synthetic.ipynb` | Figure 3 |
| `experiments_divergence/divergence_experiment.ipynb` | Table 8 |
| `experiments_catboost/experiment_catboost.ipynb` | Table 10 |
| `experiments_linreg/main_experiments_linreg.ipynb` | Table 11 |

### Aggregation & figure scripts
After running the experiments, aggregate the saved per-seed results and regenerate the figures:

```bash
# Aggregate main MLP/XGBoost results (Tables 1 & 12) -> scripts/all_results.csv
python scripts/collect_main_experiment_results.py

# Aggregate linear-regression results (Table 11) -> scripts/all_results_linreg.csv
python scripts/collect_linreg_results.py

# Aggregate CatBoost results (Table 10)
python scripts/collect_catboost_results.py

# Aggregate augmentation-baseline results (Table 2) -> experiments_all_baselines/aggregated_results.csv
python scripts/collect_baseline_results.py

# Knob-sensitivity plots (Figures 5-6)
python scripts/knob_sensitivity.py

# Wilcoxon significance heatmaps (Figures 7-8)
python scripts/p_vals.py
```

**Reported metrics**
- `mse` — baseline test MSE; `aug_mse` — CRDA-augmented test MSE
- `delta_mse` — percent change `100·(aug_mse − mse)/mse` (**negative = improvement**)
- `p_wilcoxon` — Wilcoxon signed-rank p-value for the safety gate

---

## 🧪 Headline Results (Δ MSE %, lower is better)

Percent change in test MSE after applying CRDA, averaged over 15 seeds (paper Table 1). Negative = improvement; **bold** = >5% reduction.

| Dataset | Sample Size | XGB Δ% | MLP Δ% |
|---------|------------:|-------:|-------:|
| **CPU Performance** | 1638 | -6.99 | **-20.24** |
| | 3276 | -9.47 | **-14.03** |
| | 4914 | -6.20 | **-11.31** |
| | 6552 | -4.13 | **-10.48** |
| | 8190 | -5.19 | **-10.23** |
| **Satellite Image** | 1287 | -4.54 | **-18.36** |
| | 2574 | -3.73 | **-16.69** |
| | 3861 | -4.79 | **-23.14** |
| | 5148 | -4.73 | **-23.72** |
| | 6435 | -5.31 | **-19.66** |
| **Wind Power** | 1314 | -2.82 | **-7.22** |
| | 2628 | 0.20 | **-9.17** |
| | 3942 | -1.33 | **-9.03** |
| | 5256 | -1.40 | **-6.15** |
| | 6570 | -1.08 | **-5.56** |
| **Synthetic Regression** | 200 | -12.00 | **-28.80** |
| | 400 | -3.16 | **-36.93** |
| | 600 | -7.94 | **-27.91** |
| | 800 | -2.23 | **-34.12** |
| | 1000 | -4.59 | **-42.33** |
| **Concrete Strength** | 201 | -8.01 | **-17.80** |
| | 402 | -8.43 | **-19.83** |
| | 603 | -9.75 | **-17.64** |
| | 804 | -15.72 | **-24.77** |
| | 1005 | -12.19 | **-26.90** |
| **Energy Efficiency** | 153 | -13.33 | **-25.10** |
| | 306 | -12.20 | **-28.13** |
| | 459 | -10.55 | **-42.98** |
| | 612 | -19.35 | **-40.71** |
| | 765 | -20.96 | **-28.31** |
| **House Price** | 200 | -14.23 | **-40.57** |
| | 400 | -5.39 | **-37.02** |
| | 600 | -4.87 | **-30.14** |
| | 800 | -9.86 | **-30.32** |
| | 1000 | -6.50 | **-26.97** |
| **Parkinson's Monitoring** | 1175 | -8.40 | **-36.17** |
| | 2350 | -6.60 | **-31.82** |
| | 3525 | -2.79 | **-36.60** |
| | 4700 | -6.26 | **-46.40** |
| | 5875 | 1.65 | **-47.23** |
| **Wine Quality** | 1063 | 0.31 | -0.34 |
| | 2126 | 1.01 | **-5.24** |
| | 3189 | -0.33 | **-3.63** |
| | 4252 | -0.61 | **-4.44** |
| | 5315 | -1.08 | **-4.99** |

---

## 🔬 Reproducibility Notes

- **Seeds.** Main MLP/XGBoost experiments and the CatBoost/linear-regression studies use **15 seeds**; the augmentation-baseline comparison (Table 2) uses **10 fixed seeds**. All RNGs (Python, NumPy, PyTorch) are seeded.
- **Significance.** A two-sided Wilcoxon signed-rank test (10-fold CV) gates whether augmentation is applied.
- **Standard error.** Aggregation reports the **standard error of the mean** (`sample_std / √n`, computed with `ddof=1`) over seeds.
- **Determinism.** CRDA, C-Mixup, ADA, and the cached TabDDPM column are deterministic given a seed. **CTGAN and TVAE are trained live**, so their numbers may drift slightly across library/hardware versions even with seeding.

### External baselines (C-Mixup & ADA)
C-Mixup and ADA are run from the original authors' implementation, which lives in a **separate repository and conda environment** (its dependencies conflict with ours):

- Upstream: <https://github.com/NoraSchneider/anchordataaugmentation>
- The `experiments_all_baselines/aug_data_all_baselines.ipynb` notebook shells out to that repo via a conda env named `ada`. The path to it is currently **hard-coded** in the C-Mixup/ADA cells — clone the upstream repo, create its `ada` env, and edit that path to run them yourself. If unset, only those two cells are skipped; the other methods still run.
- Their precomputed per-seed results are already included in `experiments_all_baselines/comparison.csv`, so re-running them is **not** required to inspect Table 2.

### TabDDPM
TabDDPM is **not** trained in this repo. We generated its synthetic samples using the official implementation (<https://github.com/yandex-research/tab-ddpm>) in a **separate environment**, then cached them in `experiments_all_baselines/diffusion_data/`. The notebook loads those cached samples at run time — keep that folder to reproduce the TabDDPM column.

---

## 📄 License

This code is released under the MIT License — see the [`LICENSE`](LICENSE) file.

---

## 📚 Citation

If you use this code or method, please cite:

```bibtex
@inproceedings{mohebbi2026crda,
  title     = {Counterfactual Residual Data Augmentation for Regression},
  author    = {Mohebbi, Hossein and Schulte, Oliver and Li, Ke and Poupart, Pascal},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
