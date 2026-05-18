---
name: blind-ml-demo
description: Build or extend Blind Insight encrypted-ML demo notebooks in blind-ml. Use when adding a notebook, a model trained on aggregate queries, demo_helpers/healthcare helpers, BI client usage, or debugging parity between encrypted and sklearn baselines.
---

# blind-ml demo contributor

## Architecture (short)

- **Training:** `BlindInsightClient.query(..., count_only=True)` and aggregate filters — no record-level decryption during training.
- **Plaintext baseline:** Local SQLite under `demo_data/plaintext/` (from `scripts/generate_*.py`).
- **Config:** `get_fraud_demo_config()` in `blind_ml/demo_helpers.py`, `get_bc_demo_config()` in `blind_ml/healthcare.py`.
- **Auth:** `BI_EMAIL`, `BI_PASSWORD`, `BI_ORG` in `.env` only.

## Checklist for a new demo

1. **Schema** — add `schemas/<name>.json`; document field types and min/max (integers often need `maximum = actual_max + 2` in schema).
2. **Generator** — `scripts/generate_<domain>_data.py` writes matching SQLite + `demo_data/upload_batches/*.json`.
3. **Helpers** — training/query/HTML tables in `blind_ml/demo_helpers.py` or `blind_ml/healthcare.py`; keep notebooks thin.
4. **Notebook** — setup cell: `load_env()`, config dict, `BlindInsightClient(proxy_url=...)`, `warm_up()`.
5. **Register symbols** — add imports to `scripts/smoke_test.py` notebook symbol lists.
6. **Docs** — README row + short section; link upload batches in [demo-datasets](https://github.com/blind-insight/demo-datasets/tree/main/datasets/blind-ml).

## Algorithms that fit today

See [APPROACH.md](../../APPROACH.md#what-algorithms-work-on-encrypted-data). Prefer reusing existing patterns:

| Pattern | Example in repo |
|---------|-----------------|
| NB from counts | `run_bi_training`, `build_bc_model` |
| DT from marginals + local cross-tabs | `run_encrypted_dt_fraud`, `run_encrypted_dt` |
| LR from X'X / X'y + IRLS | `build_fraud_linear_model`, `train_evaluate_bc_lr_models` |

## Common pitfalls

- **Query syntax:** `field:count(50~100)` not `count(50, 100)`.
- **Warm-up:** call `client.warm_up(org, dataset, schema)` before heavy query loops.
- **Scale:** fraud batches are 50K records each; BC batches 20K — upload as many as needed; SQLite generator can build full scale locally.
- **Parity:** encrypted vs sklearn F1 gaps often mean schema bounds, indexing not finished (wait ~30s after upload), or plaintext/BI data mismatch — regenerate with the same script for both.

## Files to touch

| Change | Files |
|--------|--------|
| New fraud-style demo | `schemas/`, `scripts/generate_*.py`, `blind_ml/demo_helpers.py`, `*.ipynb`, `smoke_test.py`, README |
| New healthcare-style demo | same + `blind_ml/healthcare.py` |
| Client behavior | `blind_ml/client.py` (proxy-only) |

Do not add alternate BI backends or new dependencies without maintainer approval.
