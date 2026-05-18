# Demo data

Nothing in this directory is committed except `.gitkeep` placeholders. Generate data locally or copy pre-built batches from the public [demo-datasets](https://github.com/blind-insight/demo-datasets) repo.

## Option A: Generate locally (recommended for full-scale runs)

From the **blind-ml repo root**:

| Command | Output |
|---------|--------|
| `python3 scripts/generate_fraud_data.py` | `plaintext/fraud_*.db` + `upload_batches/fraud_*.json` |
| `python3 scripts/generate_fraud_data.py --append-noise` | Extra noise rows/batches |
| `python3 scripts/generate_healthcare_data.py` | `plaintext/bc_*.db` + `upload_batches/bc_*.json` |

## Option B: Download sample batches

Pre-built JSON for quick starts lives in [demo-datasets `datasets/blind-ml/`](https://github.com/blind-insight/demo-datasets/tree/main/datasets/blind-ml):

```bash
git clone --depth 1 https://github.com/blind-insight/demo-datasets.git /tmp/demo-datasets
cp /tmp/demo-datasets/datasets/blind-ml/*.json demo_data/upload_batches/
```

Expected files include `fraud_train_batch_01.json`, `fraud_test_batch_01.json`, `bc_train_batch_01.json`, and `bc_test_batch_01.json` (synthetic data only).

## Layout

| Path | Purpose |
|------|---------|
| `plaintext/` | SQLite mirrors for sklearn comparison (gitignored) |
| `upload_batches/` | JSON upload files for Blind Insight (gitignored) |

## Upload

See the main [README](../README.md#upload-data-to-blind-insight). Replace `REPLACE_WITH_YOUR_SCHEMA_URL` in batch JSON before bulk upload unless using the Web UI (which can infer schema).

Breast cancer serial upload helper:

```bash
python3 scripts/upload_bc_batches.py
```
