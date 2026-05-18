# Demo Datasets

This folder contains demo data for both Blind Insight ML notebooks. **All data is generated** — run the generators from the repo root to create them.

## Generators

| Script | Dataset | Output |
|--------|---------|--------|
| `python3 generate_demo_data.py` | Fraud detection | 500K train + 50K test |
| `python3 generate_demo_data.py --append-noise` | Fraud noise | +100K train + 4K test (flipped fraud_type) |
| `python3 generate_healthcare_data.py` | Breast cancer screening | 50K train + 10K test |

## `plaintext/`

Local SQLite databases used by the notebooks for plaintext model comparison:

- **`fraud_train.db`** — 600K training records (table: `train`): 500K base + 100K noise
- **`fraud_test.db`** — 54K test records (table: `test`): 50K base + 4K noise
- **`bc_train.db`** — 50K breast cancer training records (table: `train`)
- **`bc_test.db`** — 10K breast cancer test records (table: `test`)

Each 50K slice of the fraud training data (rows 1–50K, 1–100K, etc.) has the same feature distributions, so the plaintext comparison stays valid regardless of how many batches you uploaded to Blind Insight. The same applies to each 10K breast cancer batch.

## `upload-to-bi/`

JSON batch files formatted for uploading to Blind Insight:

**Fraud detection:**
- **Training:** 10 files × 50K records (`train_batch_01.json` – `train_batch_10.json`) = 500K
- **Test:** 1 file × 50K records (`test_batch_01.json`)
- **Noise training:** 2 files × 50K records (`train_noise_01.json` – `train_noise_02.json`) = 100K
- **Noise test:** 1 file × 4K records (`test_noise_01.json`)

**Breast cancer screening:**
- **Training:** 5 files × 10K records (`bc_train_batch_01.json` – `bc_train_batch_05.json`) = 50K
- **Test:** 1 file × 10K records (`bc_test_batch_01.json`)

### How to upload to Blind Insight

1. **Create your dataset and schemas** using the Blind CLI (see the main [README](../README.md) for full instructions and schema files).

2. **Replace the schema URL placeholder** in each batch file with your schema ID (from `blind schema list`). Use the full URI format per [official docs](https://docs.blindinsight.io/getting-started/uploading-data/#json-support):
   ```bash
   # Fraud train batches:
   sed -i '' 's|REPLACE_WITH_YOUR_SCHEMA_URL|https://api.app.blindinsight.io/api/schemas/YOUR_TRAIN_SCHEMA_ID/|g' upload-to-bi/train_batch_*.json upload-to-bi/train_noise_*.json

   # Fraud test batches:
   sed -i '' 's|REPLACE_WITH_YOUR_SCHEMA_URL|https://api.app.blindinsight.io/api/schemas/YOUR_TEST_SCHEMA_ID/|g' upload-to-bi/test_batch_01.json upload-to-bi/test_noise_01.json

   # Breast cancer (use your BC schema IDs):
   sed -i '' 's|REPLACE_WITH_YOUR_SCHEMA_URL|https://api.app.blindinsight.io/api/schemas/YOUR_BC_TRAIN_SCHEMA_ID/|g' upload-to-bi/bc_train_batch_*.json
   sed -i '' 's|REPLACE_WITH_YOUR_SCHEMA_URL|https://api.app.blindinsight.io/api/schemas/YOUR_BC_TEST_SCHEMA_ID/|g' upload-to-bi/bc_test_batch_01.json
   ```

3. **Upload via Web UI, Proxy API, or CLI** (see [full upload docs](https://docs.blindinsight.io/getting-started/uploading-data/)):

   **Web UI (drag and drop — easiest):**
   - Open the Blind Insight web app and navigate to your schema's record list
   - Drag and drop a batch file onto the upload area
   - Wait for the progress bar to finish before uploading the next file
   - Web UI auto-detects the schema, so you can skip the `sed` step above
   - See [Web UI docs](https://docs.blindinsight.io/getting-started/uploading-data/#web-ui)

   **Upload script (recommended for bulk):**
   ```bash
   # Set credentials
   export BI_EMAIL=your-email@example.com
   export BI_PASSWORD=yourpassword

   # Upload fraud data only
   bash demo-datasets/upload-to-bi/upload_batches.sh

   # Upload fraud + noise
   bash demo-datasets/upload-to-bi/upload_batches.sh --with-noise

   # Upload breast cancer data only
   bash demo-datasets/upload-to-bi/upload_batches.sh --bc

   # Upload everything
   bash demo-datasets/upload-to-bi/upload_batches.sh --all
   ```

   **Proxy REST API — curl (manual):**
   ```bash
   curl -X POST 'https://local.blindinsight.io/api/jobs/upload/' \
     -u "$BI_EMAIL:$BI_PASSWORD" \
     -H 'Content-Type: application/json' \
     --data-binary "@upload-to-bi/train_batch_01.json"
   ```
   Each request returns a `{"job_id": "..."}` you can poll: `curl -s https://local.blindinsight.io/api/jobs/JOB_ID/ -u "$BI_EMAIL:$BI_PASSWORD"`

   **CLI** (`blind jobs upload`):
   ```bash
   blind jobs upload --data upload-to-bi/train_batch_01.json
   ```

   Wait for indexing to complete between batches. You can upload as few as 1 training batch for a quick test.

See [Uploading Data docs](https://docs.blindinsight.io/getting-started/uploading-data/) for details on JSON/CSV formats, batch sizes, and job monitoring.
