---
description: Conventions for contributing to blind-ml
alwaysApply: true
---

# blind-ml contributor rules

This repo demonstrates training sklearn-style models on **encrypted aggregate queries** via [Blind Insight](https://blindinsight.com). Demo notebooks live at the repo root; shared library code is in `blind_ml/`.

## Before you start

1. Read [APPROACH.md](APPROACH.md) — how training uses counts, not decrypted rows.
2. Read [CONTRIBUTING.md](CONTRIBUTING.md) — setup, lint, PR flow.
3. For BI platform questions, use [docs.blindinsight.io](https://docs.blindinsight.io) (not assumptions about internal deployments).

## Adding a notebook or model

- Open an issue or comment on a PR before large new demos.
- Put notebooks at repo root (`fraud.ipynb` pattern) or discuss a subfolder in the issue.
- Add a `get_*_demo_config()` helper (see `get_fraud_demo_config()` / `get_bc_demo_config()`) for dataset/schema slugs and paths — **not** `.env`.
- Reuse `BlindInsightClient` (proxy HTTP API only); credentials come from `.env` via `load_env()`.
- Plaintext comparison SQLite lives under `demo_data/plaintext/` (gitignored) — generate with `scripts/generate_*.py`.
- Upload JSON batches are **not** in this repo; generate locally or copy from [demo-datasets](https://github.com/blind-insight/demo-datasets/tree/main/datasets/blind-ml).
- Use `pd.get_dummies()` for one-hot encoding in plaintext baselines, not `LabelEncoder`, so splits align with BI categorical handling.
- Keep paths relative to repo root (`demo_data/...`).

## Notebooks in git

- **Do not commit cell outputs** — clear outputs before opening a PR (`jupyter nbconvert --clear-output` or Jupyter “Clear All Outputs”).
- Do not commit `.env`, `demo_data/plaintext/*.db`, or `demo_data/upload_batches/*.json`.

## Lint

```bash
ruff check --fix .
ruff format .
nbqa ruff --fix breast_cancer.ipynb fraud.ipynb   # same ignores as CI; see pyproject.toml
python3 scripts/smoke_test.py
```

## Secrets

Never commit credentials, real customer data, or internal-only URLs. Synthetic generators only.
