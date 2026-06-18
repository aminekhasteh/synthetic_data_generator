# Synthetic Data Generator

Generate differentially private synthetic tabular data with [OpenDP SmartNoise](https://docs.smartnoise.org/synth/index.html) and evaluate quality across five dimensions: **statistical fidelity**, **temporal structure**, **privacy**, **Azure compatibility**, and **downstream utility**.

Includes a CLI for arbitrary seed datasets and a Jupyter notebook walkthrough using the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Features

- **SmartNoise synthesizers** — MWEM (default), DP-CTGAN, MST, AIM
- **Arbitrary seed datasets** — CSV or Parquet with optional data dictionary
- **Five metric dimensions** — fidelity, temporal, privacy, Azure compat, utility (TSTR/TRTR)
- **Bootstrap uncertainty** — configurable replicates with confidence intervals
- **Structured outputs** — timestamped run folders with `seed_data/` artifacts

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CLI (any dataset)

```bash
python run_synthetic_pipeline.py \
  --seed-dataset /path/to/data.csv \
  --columns Time,Amount,V1,V14,Class \
  --data-dictionary examples/creditcard_dictionary.yaml \
  --sampling true
```

### Notebook (credit card fraud EDA)

```bash
jupyter notebook notebooks/smartnoise_synthetic_eda.ipynb
```

Requires Kaggle credentials for `kagglehub` (`~/.kaggle/kaggle.json`).

## CLI reference

### Required

| Flag | Description |
|------|-------------|
| `--seed-dataset` | Path to seed CSV or Parquet |
| `--columns` | Comma-separated column list |

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dictionary` | — | Optional YAML/JSON with column metadata |
| `--sampling` | `true` | Fast (`true`) or full (`false`) run |
| `--epsilon` | `1.0` | Differential privacy budget |
| `--synthesizer` | `mwem` | `mwem`, `dpctgan`, `mst`, `aim` |
| `--output-dir` | `outputs` | Base output directory |
| `--seed-size` | 2000 / 10000 | Seed sample size (fast / slow) |
| `--holdout-size` | 5000 / 20000 | Holdout for utility metrics |
| `--pool-size` | 10000 / 50000 | Bootstrap pool size |
| `--random-state` | `42` | Random seed |
| `--verbose` | off | Debug logging |

### Bootstrap options

| Flag | Default | Description |
|------|---------|-------------|
| `--bootstrap` | `true` | Enable/disable bootstrap |
| `--bootstrap-n` | 5 / 30 | Number of replicates (fast / slow) |
| `--bootstrap-sample-size` | `--seed-size` | Rows per bootstrap resample |
| `--bootstrap-seed` | `--random-state` | Bootstrap RNG seed |
| `--bootstrap-ci-low` | `0.025` | Lower CI quantile |
| `--bootstrap-ci-high` | `0.975` | Upper CI quantile |
| `--bootstrap-min-class-count` | `1` | Min positive-class rows (stratified) |
| `--bootstrap-save-replicates` | `true` | Save `bootstrap_*/` subfolders |

## Data dictionary (optional)

```yaml
table_name: creditcard
target_column: Class
temporal_column: Time
categorical_columns:
  - Class
continuous_columns:
  - Time
  - Amount
  - V1
  # ...
```

See [`examples/creditcard_dictionary.yaml`](examples/creditcard_dictionary.yaml).

## Output layout

Each run creates a folder like:

```
outputs/creditcard_20260618_123006_sampling-true_boot-10_eps-1.0_synth-mwem/
├── seed_data/
│   ├── seed.csv              # seed used for synthesis
│   ├── source.csv            # copy of input file
│   ├── dictionary.yaml       # copied if provided
│   └── run_config.json       # all run parameters
├── synthetic.csv
├── metrics.json
├── bootstrap_metrics.csv     # if bootstrap enabled
├── bootstrap_replicates.json
├── dataset_config.json
└── azure_compat_report.json
```

## Metrics

| Dimension | Examples |
|-----------|----------|
| **Statistical fidelity** | KS statistic, Wasserstein distance, correlation L2 diff, target prevalence error |
| **Temporal** | Time distribution KS, inter-arrival gaps, binned transaction rates |
| **Privacy** | DP epsilon spent, DCR, NN overlap, membership inference AUC |
| **Azure compatibility** | CSV/Parquet export, smartnoise-sql metadata, DP SQL smoke test |
| **Utility** | TRTR vs TSTR AUROC, PR-AUC, F1 for downstream classification |

## Project structure

```
├── run_synthetic_pipeline.py   # CLI entry point
├── requirements.txt
├── notebooks/
│   └── smartnoise_synthetic_eda.ipynb
├── examples/
│   └── creditcard_dictionary.yaml
└── src/
    ├── config.py               # dataset config & data dictionary
    ├── data.py                 # sampling & bootstrap helpers
    ├── pipeline.py             # end-to-end pipeline
    ├── synthesis.py            # SmartNoise wrapper
    └── metrics/                # fidelity, temporal, privacy, utility, azure
```

## Notes

- **Highly imbalanced targets** (e.g. fraud at 0.17%) may produce zero positive rows in synthetic data at low epsilon — raise `--epsilon` or try `dpctgan` for better continuous fidelity.
- **MWEM** requires `split_factor=1` on high-dimensional data (handled automatically).
- Output artifacts and local CSVs are gitignored; download seed data separately.

## References

- [SmartNoise Synthesizers docs](https://docs.smartnoise.org/synth/index.html)
- [SmartNoise SQL docs](https://docs.smartnoise.org/sql/)
- [OpenDP SmartNoise SDK](https://github.com/opendp/smartnoise-sdk)

## License

MIT (SmartNoise components use MIT — see upstream packages).
