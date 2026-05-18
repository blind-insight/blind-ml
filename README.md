# blind-ml — Train ML models on encrypted data

[![Watch Demo Recording](https://i9.ytimg.com/vi/N9VNa7xC_48/mqdefault.jpg?sqp=CIDA5s0G-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGD8gEyh_MA8=&rs=AOn4CLDpMxVDSvv8_M38H_ROE8uBJrjTAg)](https://www.youtube.com/watch?v=N9VNa7xC_48 "Encrypted Fraud Training")

**Train sklearn-style models on sensitive data that never gets decrypted** — using [Blind Insight](https://blindinsight.com) searchable encryption.

| Notebook | Domain | Models | Scale |
|----------|--------|--------|-------|
| [`fraud.ipynb`](fraud.ipynb) | Cross-border fraud (IBANs, jurisdictions, reports) | Naive Bayes, Decision Tree, Logistic Regression | 50K per batch — upload as many batches as you want |
| [`breast_cancer.ipynb`](breast_cancer.ipynb) | Breast cancer screening risk (HIPAA k=11 binning) | Naive Bayes, Decision Tree, Logistic Regression + Gail/BCSC benchmarks | 20K per batch — upload as many as you want |

Both demos match their sklearn plaintext counterparts while training **only on encrypted aggregate queries** (no record-level decryption during training).

---

## What you'll do

1. Sign up for [Blind Insight](https://app.blindinsight.io) and install the [Blind Proxy](https://docs.blindinsight.io/download/) (`blind` CLI).
2. Install Python deps and obtain demo data ([generate](#step-2-demo-data) or [download from demo-datasets](https://github.com/blind-insight/demo-datasets/tree/main/datasets/blind-ml)).
3. Create a BI dataset + train/test schemas and upload JSON batches.
4. Copy [`.env.example`](.env.example) → `.env` with your email, password, and org slug.
5. Run **one** notebook and compare encrypted vs plaintext accuracy.

**Time:** ~1–2 hours the first time (proxy setup + upload). After data is indexed, the fraud notebook trains in ~35s locally (~2 min on cloud).

**Prerequisites:** Python 3.11+, Blind Insight account, proxy binary from [docs.blindinsight.io/download](https://docs.blindinsight.io/download).

---

## What is Blind Insight?

Blind Insight is **searchable encryption** for structured data. You upload records encrypted; queries return **counts and aggregates**, not decrypted rows. This repo shows how to train classifiers from those aggregates alone — the same math sklearn uses, without pulling plaintext off the server.

Official docs: [docs.blindinsight.io](https://docs.blindinsight.io) · Deeper ML architecture: [APPROACH.md](APPROACH.md)

---

## Choose a demo

| Start here | Notebook | Best for |
|------------|----------|----------|
| **Fraud (recommended)** | [`fraud.ipynb`](fraud.ipynb) | Three algorithms, large-scale financial data, cross-border story |
| **Healthcare** | [`breast_cancer.ipynb`](breast_cancer.ipynb) | HIPAA, clinical risk models, Gail/BCSC comparison |

Each demo has its own guide below. Setup is the same pattern; only schemas, generators, and config helpers differ.

**Configuration split:**
- **`.env`** — shared login: `BI_EMAIL`, `BI_PASSWORD`, `BI_ORG` (see [`.env.example`](.env.example))
- **Notebook config** — dataset/schema slugs: `get_fraud_demo_config()` in `blind_ml/demo_helpers.py`, `get_bc_demo_config()` in `blind_ml/healthcare.py`

> **Two logins:** `./blind login` configures the proxy CLI/keyring. The notebook uses **HTTP basic auth** from `.env` for API calls — both are required.

---

## Quick Start: Fraud demo

### Prerequisites checklist

- [ ] Blind Insight account; `./blind login` works
- [ ] `./blind users self` or `./blind organization list` succeeds
- [ ] Keyring created if first time: `./blind keyring create` ([docs](https://docs.blindinsight.io/getting-started/using-the-blind-proxy/#create-your-keyring))
- [ ] Proxy running: `./blind proxy` (keep terminal open)
- [ ] At least one training batch + test batch uploaded to BI (see [Upload data](#upload-data-to-blind-insight) below)
- [ ] Local SQLite generated: `demo_data/plaintext/fraud_train.db` and `fraud_test.db`

### Step 1: Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Demo data

JSON upload batches and SQLite files are **not** in this repo. Use either path:

**Generate locally** (full control over scale):

```bash
python3 scripts/generate_fraud_data.py
# optional harder task:
python3 scripts/generate_fraud_data.py --append-noise
```

**Or download sample batches** from [demo-datasets](https://github.com/blind-insight/demo-datasets/tree/main/datasets/blind-ml):

```bash
git clone --depth 1 https://github.com/blind-insight/demo-datasets.git /tmp/demo-datasets
cp /tmp/demo-datasets/datasets/blind-ml/*.json demo_data/upload_batches/
```

You still need SQLite for plaintext benchmarks — run the generator at least once, or copy matching `.db` files if published alongside the JSON in demo-datasets.

Details: [`demo_data/README.md`](demo_data/README.md).

### Step 3: Start the Blind Proxy

The notebook talks to the proxy at `https://local.blindinsight.io` (override with `BI_PROXY_URL` in `.env`):

```bash
./blind proxy
```

### Step 4: Configure `.env`

```bash
cp .env.example .env
# Edit .env — your BI email, password, and org slug
```

```env
BI_EMAIL=your-email@example.com
BI_PASSWORD=your-password
BI_ORG=your-org-slug
```

If your dataset/schema slugs differ from the defaults (`fraud-demo`, `train`, `test`), edit `get_fraud_demo_config()` in `blind_ml/demo_helpers.py`.

### Step 5: Create dataset & schemas in Blind Insight

Use your org slug from `./blind organization list`.

```bash
./blind dataset create --organization YOUR_ORG --name "Fraud Data" --description "Fraud demo"
./blind schema create --name Train --dataset YOUR_DATASET_SLUG --organization YOUR_ORG \
  --description "Fraud training records" --file schemas/fraud.json
./blind schema create --name Test --dataset YOUR_DATASET_SLUG --organization YOUR_ORG \
  --description "Fraud test records" --file schemas/fraud.json
```

Slugs are derived from names (e.g. `train`, `test`). `schemas/fraud.json` matches the generated demo data.

### Step 6: Upload data to Blind Insight

See [Upload data](#upload-data-to-blind-insight) below. **You choose the scale** — one 50K batch is enough to run the notebook; upload more batches for larger training sets (up to the full ~600K if you generate and upload everything).

### Step 7: Run the notebook

```bash
source venv/bin/activate
jupyter notebook fraud.ipynb
```

Verify before **Run All**:

```bash
curl -sk https://local.blindinsight.io/api/health/
ls demo_data/plaintext/fraud_train.db demo_data/plaintext/fraud_test.db
python -c "import pandas, sklearn; print('OK')"
```

The notebook loads local SQLite for plaintext benchmarks, trains encrypted models via ~90 aggregate queries, compares F1 to sklearn, runs validation and a realtime demo.

**Expected runtime:** ~35s encrypted training on local BI (~2 min cloud) at full scale; faster with fewer uploaded records.

---

## Quick Start: Breast cancer demo

Same flow as fraud — different schema, generator, and notebook.

### Steps 1–4

Follow fraud Steps 1–4 (venv, deps, proxy, `.env`).

### Step 5: Generate healthcare data

```bash
python3 scripts/generate_healthcare_data.py
```

Writes `demo_data/upload_batches/bc_train_batch_*.json`, `bc_test_batch_01.json`, and `demo_data/plaintext/bc_*.db`.

### Step 6: Create dataset & schemas

```bash
./blind dataset create --organization YOUR_ORG --name "Breast Cancer Risk" --description "BC risk demo"
./blind schema create --name Train --dataset YOUR_DATASET_SLUG --organization YOUR_ORG \
  --description "BC training records" --file schemas/breast_cancer.json
./blind schema create --name Test --dataset YOUR_DATASET_SLUG --organization YOUR_ORG \
  --description "BC test records" --file schemas/breast_cancer.json
```

Defaults live in `get_bc_demo_config()` in `blind_ml/healthcare.py` — update there if your slugs differ.

### Step 7: Upload batches

```bash
python3 scripts/upload_bc_batches.py
```

Reads `.env`, uploads each batch, polls jobs to completion. Or use the same Web UI / curl methods as the fraud demo.

### Step 8: Run the notebook

```bash
jupyter notebook breast_cancer.ipynb
```

Trains Naive Bayes, Decision Tree, and Logistic Regression on encrypted aggregates (HIPAA k=11 binning) and benchmarks against Gail-model / SEER relative risks.

---

## Upload data to Blind Insight

### How much to upload?

Each batch has the **same feature distributions**, so the demo works at any scale:

| Fraud | Batch size | Upload |
|-------|------------|--------|
| Quick try | 1 train + 1 test file | ~50K train records |
| Partial | Any subset of `fraud_train_batch_*.json` | Your choice |
| Full generated set | All train batches + test (+ optional noise) | Up to ~600K train / ~54K test |

Upload **as many training batches as you want** — whatever your account and patience allow. The notebook compares against the local SQLite mirror, which includes the full generated dataset regardless of how much you uploaded to BI.

Breast cancer: 20K records per `bc_train_batch_*.json` — same idea, upload one or all.

### Replace schema URLs (curl / CLI upload)

For bulk upload, replace the placeholder in batch JSON with your schema IDs from `blind schema list`:

```bash
# Fraud train batches
sed -i '' 's|REPLACE_WITH_YOUR_SCHEMA_URL|https://api.app.blindinsight.io/api/schemas/YOUR_TRAIN_SCHEMA_ID/|g' \
  demo_data/upload_batches/fraud_train_batch_*.json

# Fraud test batch
sed -i '' 's|REPLACE_WITH_YOUR_SCHEMA_URL|https://api.app.blindinsight.io/api/schemas/YOUR_TEST_SCHEMA_ID/|g' \
  demo_data/upload_batches/fraud_test_batch_01.json
```

**Web UI:** drag-and-drop at `https://local.blindinsight.io` — schema is often auto-detected, so you can skip `sed`.

### Method 1: Web UI (easiest)

1. Open the app behind the proxy → your dataset → **train** schema.
2. Drag a `fraud_train_batch_*.json` onto the upload area; wait for completion.
3. Repeat for as many training batches as you want.
4. Switch to **test** schema → upload `fraud_test_batch_01.json`.

[Web UI docs](https://docs.blindinsight.io/getting-started/uploading-data/#web-ui)

### Method 2: Proxy REST API (bulk)

With `./blind proxy` running and credentials in `.env`:

```bash
export $(grep -v '^#' .env | xargs)   # load BI_EMAIL, BI_PASSWORD

for f in demo_data/upload_batches/fraud_train_batch_*.json; do
  echo "Uploading $f..."
  JOB_ID=$(curl -s -X POST 'https://local.blindinsight.io/api/jobs/upload/' \
    -u "$BI_EMAIL:$BI_PASSWORD" \
    -H 'Content-Type: application/json' \
    --data-binary "@$f" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
  while true; do
    sleep 10
    STATUS=$(curl -s "https://local.blindinsight.io/api/jobs/$JOB_ID/" -u "$BI_EMAIL:$BI_PASSWORD")
    echo "$STATUS"
    echo "$STATUS" | grep -q '"complete"' && break
    echo "$STATUS" | grep -q '"failed"' && exit 1
  done
done

curl -X POST 'https://local.blindinsight.io/api/jobs/upload/' \
  -u "$BI_EMAIL:$BI_PASSWORD" \
  -H 'Content-Type: application/json' \
  --data-binary '@demo_data/upload_batches/fraud_test_batch_01.json'
```

### Method 3: CLI

```bash
blind jobs upload --data demo_data/upload_batches/fraud_train_batch_01.json
```

If uploads don't land via the local proxy, use Method 2 ([known issue with `--host` routing](https://docs.blindinsight.io/getting-started/uploading-data/)).

More detail: [`demo_data/README.md`](demo_data/README.md) · [Uploading data docs](https://docs.blindinsight.io/getting-started/uploading-data/)

---

## What's happening under the hood?

**Problem:** Fraud (or PHI) can't be shared across borders or institutions in plaintext.

**Approach:** Train from **aggregate counts** only:

```
P(fraud_type = "mule_account" | high_risk) = count(mule_account AND high_risk) / count(high_risk)
```

~90 encrypted queries feed Naive Bayes, Decision Tree (Gini), and Logistic Regression (OLS + IRLS). DT and LR reuse NB marginals — no extra BI round-trips.

Blind Insight uses **two keys per field**: a query key (search/aggregate) and a field key (decrypt). Aggregates need only the query key — so a party without decrypt permission can still train accurate models. See [Key sharing](https://docs.blindinsight.io/getting-started/key-sharing/) and [APPROACH.md](APPROACH.md).

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR LAPTOP (Jupyter)                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Notebook Cell  │───▶│  Python Client  │───▶│ Blind Proxy  │ │
│  │  (ML Training)  │    │  (HTTP Request) │    │ (local)      │ │
│  └─────────────────┘    └─────────────────┘    └──────┬───────┘ │
└────────────────────────────────────────────────────────┼────────┘
                                                         ▼
                                              ┌───────────────────┐
                                              │  Blind Insight    │
                                              │  Returns: COUNT   │
                                              │  (not raw rows)   │
                                              └───────────────────┘
```

| Traditional ML | Blind Insight ML |
|----------------|------------------|
| Data decrypted for training | Data stays encrypted |
| Raw records exposed | Only aggregates returned |
| Siloed by compliance | Cross-org collaboration on encrypted data |

**Data integrity:** Local SQLite (`scripts/generate_fraud_data.py`) matches upload batches record-for-record. BI is the source of truth for encrypted training; SQLite is for plaintext comparison only.

---

## Fraud notebook walkthrough

| Section | What it does |
|---------|----------------|
| Load data | SQLite mirror + proxy warm-up |
| Train Naive Bayes | ~90 BI aggregate queries vs plaintext NB |
| Train Decision Tree | Gini/CART from counts; sklearn comparison |
| Train Logistic Regression | OLS from X'X, X'y + IRLS |
| Three-model comparison | F1, sensitivity, specificity, confusion matrices |
| Real-time demo | Encrypted vs decrypted side-by-side |
| Test validation | Encrypted vs plaintext on held-out records |
| Scaling calculator | Plaintext vs BI vs FHE extrapolation |

### Live demo tips

- Re-running cells is fast once results are cached.
- Call out **"Data Decrypted: NEVER"** in the training summary table.
- Talking points: same accuracy, different privacy posture; counts not rows; enables data sharing that wasn't possible before.

---

## Results (fraud demo, full scale)

Validated at ~600K train / ~54K test when all batches are uploaded and generated locally:

| Model | sklearn F1 | Encrypted F1 | BI queries | Data decrypted |
|-------|-----------|-------------|------------|----------------|
| Naive Bayes | 0.942 | 0.942 | ~90 | Never |
| Decision Tree | 0.942 | 0.942 | 0 (reuses NB) | Never |
| Logistic Regression | 0.942 | 0.942 | 0 (reuses NB) | Never |

| Aspect | Plaintext | Blind Insight |
|--------|-----------|---------------|
| NB training | ~0.01s | ~35s local / ~2min cloud |
| DT / LR | ~1–3s each | Seconds (local math on NB counts) |

Training time scales sub-linearly with record count — doubling rows does not double query time. See [APPROACH.md](APPROACH.md) for algorithms, query syntax, and extension ideas.

---

## Troubleshooting

### Connection refused / proxy errors

Run `./blind proxy` and verify:

```bash
curl -sk https://local.blindinsight.io/api/health/
```

### Proxy auth not configured

1. `./blind login` for the proxy process.
2. `.env` must have `BI_EMAIL`, `BI_PASSWORD`, `BI_ORG`.

Re-run the notebook setup cell — you should see `Proxy warm-up` with timing.

### Slow queries or zero counts

Indexes build after upload. Wait 30+ seconds, retry. Counts should be > 0 when ready.

### Wrong aggregate counts

- Syntax: `risk_level:count(50~100)` not `count(50, 100)`
- Integer schema `maximum` = actual max + 2 (see `schemas/fraud.json`)
- Re-upload after schema fixes

### Local vs BI mismatch

Regenerate: `python3 scripts/generate_fraud_data.py`

### Keyring / seed phrase errors

[Keyring docs](https://docs.blindinsight.io/getting-started/using-the-blind-proxy/#create-your-keyring) — `./blind keyring create` then `./blind keyring inspect`

### Import errors

```bash
pip install -r requirements.txt
```

### Notebook hangs

Training uses parallel queries; if timeouts persist, check proxy health and network, or reduce parallel load in `blind_ml/demo_helpers.py`.

---

## File reference

| File | Purpose |
|------|---------|
| `fraud.ipynb` | Fraud demo (NB + DT + LR) |
| `breast_cancer.ipynb` | Healthcare risk demo |
| `blind_ml/` | Client, models, demo helpers |
| `blind_ml/demo_helpers.py` | `get_fraud_demo_config()`, fraud training UI |
| `blind_ml/healthcare.py` | `get_bc_demo_config()`, BC training |
| `scripts/generate_fraud_data.py` | Fraud SQLite + JSON batches |
| `scripts/generate_healthcare_data.py` | BC SQLite + JSON batches |
| `scripts/upload_bc_batches.py` | Serial BC upload helper |
| `schemas/fraud.json` / `schemas/breast_cancer.json` | BI schema definitions |
| `demo_data/` | Placeholder dirs for generated/uploaded data (see [demo-datasets](https://github.com/blind-insight/demo-datasets)) |
| `.cursor/` | Contributor rules and skills for Cursor (optional) |
| `.env.example` | Credential template |
| `APPROACH.md` | Algorithms, architecture, contribution guide |
| `scripts/smoke_test.py` | Import + config validation (`python3 scripts/smoke_test.py`) |

---

## Learn more & contribute

- [APPROACH.md](APPROACH.md) — how ML on encrypted data works; supported algorithms
- [Blind Insight docs](https://docs.blindinsight.io)
- [Key sharing](https://docs.blindinsight.io/getting-started/key-sharing/)
- [Fuzzy matching demo (video)](https://www.youtube.com/watch?v=ZMBVsJOwJ4k)

Contributions welcome: new algorithm demos, datasets, performance work, docs. Open an issue before large PRs.

**Questions?** Troubleshooting above → [APPROACH.md](APPROACH.md) → [docs.blindinsight.io](https://docs.blindinsight.io)
