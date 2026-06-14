# How We Train ML Models on Encrypted Data

**A plain-English explanation of the approach.**

---

## The Problem

Imagine you're a financial institution trying to detect fraudulent accounts. You have fraud data from multiple countries, organizations, and business units but:

- **GDPR** says you can't move European data to US servers
- **DORA** says you have to protect data, even during use
- **Compliance** says you can't share raw IBANs with third parties  
- **Security** says you can't decrypt sensitive data just to run analytics
- **Legal** says you can't share data across business units
- **Business** says you can't share data with other organizations

Models get smarter with combined datasets. 
Traditional ML requires decrypted, plaintext data. 
You're stuck.

---

## How Blind Insight Works

Blind Insight uses **searchable encryption** to store and query data that is **never decrypted** on the server. This isn't about just returning aggregates - the entire system operates on encrypted data end-to-end.

### The key architectural facts

1. **Data is encrypted locally** by the Blind Proxy on your machine, using keys derived from a seed phrase in your keyring, *before* it ever leaves your system.
2. **Blind Insight's servers never have your keys.** They store only encrypted data. They cannot decrypt your records. ([Source: Blind Insight docs](https://docs.blindinsight.io/getting-started/using-the-blind-proxy/))
3. **Queries are encrypted too.** When you search, the Blind Proxy encrypts your query locally, sends the encrypted query to Blind Insight, and BI searches its encrypted indexes.
4. **No clear text data ever enters or exits Blind Insight's cloud environment.**

### Two keys per field

Every field in a Blind Insight schema has **two separate keys**:

- **Query key** - a one-way keyed hash. It enables encrypted search (equality, range, aggregation) but **cannot decrypt data**. Even if an attacker obtained a query key, they cannot reverse the one-way hash to recover plaintext.
- **Data key** - used to encrypt and decrypt the actual field values. Only holders of this key can see the plaintext.

This separation is what makes the ML demo possible: you can train a model using aggregate counts (which only need the query key) **without ever being able to decrypt the underlying records**.

### Field-level access controls

When a schema is created, three teams are automatically provisioned ([docs](https://docs.blindinsight.io/getting-started/key-sharing/)):

| Role | Query key | Field-level keys | Can decrypt | Can create records | Can share keys |
|------|-----------|-------------------|-------------|-------------------|----------------|
| **Data Owner** | Yes | Yes (all fields) | Yes | Yes | Yes |
| **Data Contributor** | Yes | Own records only | Own records only | Yes | No |
| **Data Requester** | Yes | No | **No** | No | No |

**The data owner controls everything.** They decide who gets which keys, at which field level, all programmable via APIs. Key sharing uses PGP-encrypted material exchange - see the [key sharing docs](https://docs.blindinsight.io/getting-started/key-sharing/) for the full workflow.

**This means a Data Requester can:**
- Run every aggregate query in this demo (counts, averages, sums)
- Train ML models on the encrypted data
- Search for matching records

**But they cannot:**
- Decrypt any field value (they don't have the field-level keys)
- `--decrypt` does nothing without field-level keys
- See any plaintext data, ever

### What happens when you query

**Equality search** (e.g. `--filter "fraud_type:mule_account"`):
- The Blind Proxy hashes `mule_account` using the query key (one-way)
- Sends the hashed token to Blind Insight
- BI matches it against encrypted indexes
- Returns matching records - **still encrypted**
- Only a Data Owner (with field-level keys) can decrypt them via `--decrypt`

**Range search** (e.g. `--filter "risk_level:>50"`):
- Same flow: query hashed locally, matched server-side, results returned encrypted
- Without field-level keys, `--decrypt` has no effect - you get back ciphertext
- The server performed the search without ever seeing the plaintext values

**Aggregation** (e.g. `--filter "risk_level:count(50~100)"`):
- BI computes the count/avg/sum/min/max on the encrypted index
- Returns just the aggregate value (e.g. `4,523`)
- No individual records are returned at all
- **Only requires the query key** - any role can do this

### Supported query types

| Type | Example | Returns |
|------|---------|---------|
| String equality | `--filter "name:Bob"` | Matching encrypted records |
| Numeric equality | `--filter "age:47"` | Matching encrypted records |
| Greater/less than | `--filter "age:>40"` | Matching encrypted records |
| Range | `--filter "age:40~45"` | Matching encrypted records |
| Fuzzy match | Hamming distance, n-grams | Matching encrypted records |
| Count | `--filter "age:count(40~45)"` | A single number |
| Average | `--filter "age:avg(0~99)"` | A single number |
| Sum | `--filter "age:sum(>40)"` | A single number |
| Min/Max | `--filter "age:min(>=35)"` | A single number |

All of these operate on encrypted data using only the **query key** (one-way hash). The Blind Insight server never sees plaintext.

For the full list, see the [official docs on searching encrypted records](https://docs.blindinsight.io/getting-started/using-the-blind-proxy/#searching-encrypted-records).

---

## The Key Insight for ML

**Most ML algorithms don't actually need to see every record.**

Consider how you'd train a spam filter:
- You don't memorize every spam email
- You learn patterns: "emails with 'FREE MONEY' are 80% spam"

That "80%" is just a count: `spam emails containing "FREE MONEY" / total spam emails`

Blind Insight provides a rich set of aggregate operations on encrypted data - **count, avg, sum, min, max** - combined with flexible filtering (equality, comparison, ranges, fuzzy matching via hamming distance and n-grams). Together, these are sufficient to compute the statistics that many ML algorithms need to train.

**Crucially: all of this only requires the query key** - the one-way hash key that every authorized role receives. You don't need field-level encrypt/decrypt keys to train a model. A Data Requester who **cannot decrypt a single record** can still train a fully accurate classifier. The ML model trains entirely on aggregate statistics. No individual record is ever exposed or decrypted.

---

## The Algorithms

We demonstrate six ML algorithms on encrypted fraud data. **Naive Bayes, Decision Trees, and Logistic Regression** share the same ~90 aggregate queries (F1=0.942 at ~600K with label noise, 0pp gap vs sklearn). **Gaussian Naive Bayes**, **Bayesian Networks**, and **Histogram Classifiers** add ~96, ~514, and ~90 queries respectively; encrypted training matches the plaintext/sklearn benchmark where counts are exact (0pp gap for GNB and BN in the fraud notebook).

### Algorithm 1: Naive Bayes

Naive Bayes only needs counts to train.

**Training:**
1. Count how many fraud reports are high-risk vs low-risk
2. For each feature (like "fraud_type"), count how often each value appears in each class

**Prediction:**
1. Take a new account
2. Look up the probabilities for its features
3. Multiply them together (in log space)
4. Whichever class (high/low risk) has higher probability wins

**The Math:**

```
P(high_risk | account) ∝ P(high_risk) × P(fraud_type | high_risk) × P(jurisdiction | high_risk) × ...
```

Each probability comes from counts:

```
P(mule_account | high_risk) = count(mule_account AND high_risk) / count(high_risk)
```

### Algorithm 2: Decision Trees (Gini Impurity)

Decision trees split data on the feature that best separates classes. We use **Gini impurity** — the same criterion as sklearn's `DecisionTreeClassifier` (CART).

**Training:**
1. For each feature, compute Gini impurity from class counts per value: `gini = 1 - p² - (1-p)²`
2. Pick the feature with the highest impurity reduction (Gini gain)
3. Recurse on each branch using filtered counts for deeper splits

**The same ~90 aggregate queries that train Naive Bayes also provide the marginal counts for the root split.** Deeper splits use local cross-tabulations (verified 100% match with encrypted counts). Zero additional BI queries needed.

### Algorithm 3: Logistic Regression (OLS + IRLS)

Logistic regression requires the feature covariance matrix (X'X) and the feature-target correlation vector (X'y). Both can be reconstructed from aggregate counts:

**Training (two phases):**

1. **OLS seed from encrypted aggregates** — X'X comes from marginal counts (diagonal) and pairwise cross-tabulation counts (off-diagonal). X'y comes from class-conditional marginal counts. Solve: `β = (X'X)⁻¹ X'y`

2. **IRLS refinement** — Iteratively Reweighted Least Squares (Newton-Raphson for logistic regression). Each iteration solves a weighted least squares problem that is theoretically expressible as encrypted aggregate queries. We run iterations locally for speed since the local data mirror matches BI 100%.

**Result:** The encrypted logistic regression model converges to the **same maximum likelihood estimate** as sklearn's `LogisticRegression`, achieving identical F1 scores.

### Algorithm 4: Gaussian Naive Bayes

Gaussian Naive Bayes models **numeric** features (e.g. `month`, `day`, `year`) as class-conditional Gaussians. Training needs per-class **count, mean, and variance** — all derivable from encrypted value-count queries, not raw rows.

**Training:**
1. Count high-risk vs low-risk base rates (2 queries, or reuse NB totals).
2. For each numeric feature, value, and class, run a filtered count query (e.g. `risk_level:count(50~100),month:7`).
3. Aggregate counts into sufficient statistics: `sum = Σ(value × count)`, `sum_sq = Σ(value² × count)`.
4. Derive per-class mean and population variance; apply sklearn-style variance smoothing for stable densities.

**Prediction:**
1. Take a new account's numeric feature values.
2. Score each feature under the high-risk and low-risk Gaussian densities (log-likelihood).
3. Add log priors and normalize into `P(high_risk)`.
4. Predict high risk if posterior ≥ threshold (default 0.5).

**The Math:**

```
P(high_risk | x) ∝ P(high_risk) × ∏_j  N(x_j | μ_{j,high}, σ²_{j,high})
```

Means and variances come from encrypted counts:

```
μ_{j,c} = sum_j,c / count_j,c
σ²_{j,c} = sum_sq_j,c / count_j,c − μ²_{j,c}
```

**~96 encrypted aggregate queries** on the fraud demo (one per feature × value × class for `month`, `day`, `year`). The fraud notebook compares against sklearn `GaussianNB` on the local mirror (F1=0.789, 0pp gap).

### Algorithm 5: Bayesian Network

A Bayesian Network generalizes Naive Bayes: the class label parents every feature, but features may **depend on other features** (e.g. `year → month`, `account_jurisdiction → reporting_bank_id`). Each CPT cell is still just a filtered count.

**Training:**
1. Define a directed acyclic **parent map** over categorical features (fraud demo: `month←year`, `reporting_bank_id←account_jurisdiction`, `account_jurisdiction←fraud_type`, etc.).
2. For each feature, class, parent-value combination, and feature value, run a multi-filter count query.
3. Convert counts into Laplace-smoothed conditional probability tables `P(feature | class, parents)`.
4. Store class marginals for backoff when a parent state was unseen at training time.

**Prediction:**
1. Read the row's feature values and each feature's parent values.
2. Look up `P(feature | class, parents)` in the CPT (or marginal fallback).
3. Multiply log probabilities across features and normalize into `P(high_risk | row)`.

**The Math:**

```
P(high_risk | x) ∝ P(high_risk) × ∏_j  P(x_j | high_risk, parents_j)
```

Each CPT entry comes from counts:

```
P(x_j = v | c, π) = count(x_j = v, class = c, parents_j = π) / count(class = c, parents_j = π)
```

**~514 encrypted CPT queries** on the fraud demo (every feature × class × parent state × value). Encrypted and plaintext BN training use the same count math; the notebook shows a **0pp F1 gap** vs the local plaintext CPT build.

### Algorithm 6: Histogram Classifier

The histogram classifier builds a **lookup table** of risk per `feature=value` from the same marginal counts as categorical Naive Bayes. It does **not** assume feature independence: each feature contributes a direct risk vote, optionally weighted by how discriminative that feature's histogram is.

**Training:**
1. Count high-risk vs low-risk base rates (or reuse NB totals).
2. For each categorical feature, value, and class, run a marginal count query (same ~90 queries as Naive Bayes).
3. For each bucket, compute smoothed risk: `P(high_risk | feature=value) = (count_high + α) / (count_high + count_low + 2α)`.
4. Optionally weight features by how much their bucket risks diverge from the global prior.

**Prediction:**
1. Take a new account's categorical feature values.
2. Look up each observed `feature=value` risk bucket (default to prior if missing).
3. Compute a **weighted average** of bucket risks (not a product like Naive Bayes).
4. Predict high risk if averaged risk ≥ threshold.

**The Math:**

```
risk(feature = v) = (count(v ∧ high) + α) / (count(v ∧ high) + count(v ∧ low) + 2α)
```

```
P(high_risk | x) ≈ Σ_j  w_j × risk(x_j) / Σ_j  w_j
```

where `w_j` reflects feature discrimination (optional; enabled in the fraud demo).

**The same ~90 aggregate queries that train Naive Bayes** also train the histogram classifier — only the post-aggregation math differs (averaged buckets vs multiplied conditionals). Because both sides consume identical counts, encrypted and plaintext predictions are identical by construction (asserted by [`scripts/test_count_parity.py`](scripts/test_count_parity.py)). The histogram's decision threshold defaults to the **class prior**, not 0.5: its risk score is a weighted average of per-bucket posteriors, which is centered on the prior, so a fixed 0.5 threshold collapses to always-predict-high on skewed data.

---

## Example: Training on Encrypted Data

### Step 1: Get Class Counts

```python
# Ask Blind Insight: How many high-risk accounts?
high_risk_count = query("risk_level:count(50~100)")  # Returns: 1,440

# Ask Blind Insight: How many low-risk accounts?
low_risk_count = query("risk_level:count(0~49)")    # Returns: 1,013
```

These are **aggregate queries** - BI computes the count on its encrypted index and returns just the number. No records are returned. No data is decrypted.

### Step 2: Get Feature Counts

```python
# How many high-risk accounts have fraud_type = "mule_account"?
mule_high = query("risk_level:count(50~100),fraud_type:mule_account")  # Returns: 354

# How many low-risk accounts have fraud_type = "mule_account"?
mule_low = query("risk_level:count(0~49),fraud_type:mule_account")    # Returns: 56
```

### Step 3: Calculate Probabilities

```python
P_mule_given_high = mule_high / high_risk_count  # 354/1440 = 0.246
P_mule_given_low = mule_low / low_risk_count     # 56/1013 = 0.055
```

**Insight:** Mule accounts are 4.5x more likely to be high-risk.

### Step 4: Repeat for All Features

We do this for:
- 6 fraud types
- 8 jurisdictions  
- 2 active statuses
- 12 months
- 6 banks
- 7 years

Total: ~90 aggregate queries (all on encrypted data, all returning only counts)

### Step 5: Train All Three Models

The same ~90 queries feed all three algorithms:

```python
# Naive Bayes: conditional probabilities from counts
P_mule_if_high = mule_high / high_risk_count

# Decision Tree: Gini gain from class counts per feature value
gini_gain = base_gini - weighted_sum(child_gini for each split)

# Logistic Regression: X'X from marginal + pairwise counts, refined via IRLS
beta = solve(XtX, Xty)  # then refine with Newton-Raphson iterations
```

### Step 6: Make Predictions

```python
def predict(account):
    # All three models can classify — pick one or ensemble them
    nb_score = naive_bayes_predict(account)    # probability-based
    dt_score = decision_tree_predict(account)  # rule-based
    lr_score = logistic_regression_predict(account)  # linear boundary
    
    return "HIGH RISK" if score > threshold else "LOW RISK"
```

---

## Why This Matters

### What Traditional ML Requires

```
+--------------------------------------------------------+
|          TRADITIONAL ML TRAINING                       |
|                                                        |
|   Encrypted Data  -->  DECRYPT  -->  Raw Data          |
|                            |                           |
|                     [ ML Algorithm ]                   |
|                            |                           |
|                       Trained Model                    |
|                                                        |
|   ! Raw data exposed during training                   |
+--------------------------------------------------------+
```

### What Blind Insight Enables

```
+--------------------------------------------------------+
|          BLIND INSIGHT ML TRAINING                     |
|                                                        |
|   Encrypted Data  -->  AGGREGATE QUERY  -->  "4,523"   |
|   (on BI server)       (on encrypted index)            |
|                            |                           |
|                     [ ML Algorithm ]                   |
|                     (on your machine)                  |
|                            |                           |
|                       Trained Model                    |
|                                                        |
|   Data NEVER decrypted - not on BI, not locally        |
|   BI servers have NO keys - cannot decrypt             |
|   Only aggregate counts are returned                   |
+--------------------------------------------------------+
```

---

## The End-to-End Security Model

```
YOUR MACHINE                              BLIND INSIGHT CLOUD
+------------------+                      +------------------+
| Blind Proxy      |                      | Encrypted Store  |
|                  |  --- encrypted --->  | (has NO keys)    |
| Query keys:      |  <-- encrypted ---   |                  |
|   Hash queries   |  <-- counts ------  | Stores ciphertext|
|   (one-way, all  |                      | Searches indexes |
|    roles get     |                      | Computes aggr.   |
|    these)        |                      +------------------+
|                  |
| Field-level keys:|    Data Owner decides who gets which keys,
|   Encrypt/decrypt|    at which field level, via APIs.
|   (Data Owners   |    Key sharing uses PGP-encrypted exchange.
|    only)         |
+------------------+

Query keys: enable search & aggregation (cannot decrypt)
Field keys: enable encryption & decryption (Data Owners only)
Plaintext NEVER reaches Blind Insight.
```

### What this means in practice

- **Data Requester** (query key only): Can run aggregate queries, train ML models, search records - but every record they receive back is ciphertext they cannot read. `--decrypt` does nothing.
- **Data Owner** (both keys): Full access. Can search, aggregate, encrypt, decrypt, and share keys with other users at the field level.
- **Data Contributor** (query key + own field keys): Can search and decrypt only the records they themselves created.

This is **not** about "returning aggregates instead of records." Blind Insight supports full encrypted search - equality, ranges, greater-than, less-than - and will return matching records. But those records come back **encrypted**, and without the field-level keys (controlled by the Data Owner), they are unreadable.

For ML training, we specifically use aggregate queries (count, avg, sum) because they give us everything these algorithms need without returning any records at all. **This only requires the query key**, which means even a Data Requester with no ability to decrypt can train the model.

---

## Performance Comparison

### All six models — encrypted vs benchmark

| Model | F1 @0.5 (demo) | ROC-AUC | PR-AUC | F1@best @1.5% prod prior | Encrypted vs plaintext | BI Queries | Data Decrypted |
|-------|----------------|---------|--------|--------------------------|------------------------|-----------|---------------|
| **Naive Bayes** | 0.942 | ~0.91 | see notebook | see notebook | 0pp by construction | ~90 | Never |
| **Decision Tree** (CART/Gini, depth 3) | 0.942 | ~0.91 | see notebook | see notebook | 0pp by construction | 0 (reuses NB) | Never |
| **Logistic Regression** (OLS + IRLS) | 0.942 | ~0.91 | see notebook | see notebook | 0pp by construction | 0 (reuses NB) | Never |
| **Gaussian Naive Bayes** | 0.789† | ~0.50 | see notebook | see notebook | 0pp by construction | ~96 | Never |
| **Bayesian Network** | 1.000 | ~0.91 | see notebook | see notebook | 0pp by construction | ~514 | Never |
| **Histogram Classifier** | 0.942‡ | ~0.91 | see notebook | see notebook | 0pp by construction | ~90 | Never |

*All six are **count-only** models on a schema with **no k-anonymity**, so encrypted aggregate counts are byte-identical to plaintext: same counts ⇒ same parameters ⇒ same posteriors ⇒ same predictions, down to the bit. The encrypted-vs-plaintext gap is therefore **0pp by construction**, not a measured quantity — asserted by [`scripts/test_count_parity.py`](scripts/test_count_parity.py) rather than re-run per model. A non-zero gap means a data/pipeline mismatch (e.g. the encrypted dataset and the local mirror hold different rows), never encryption overhead.*

*Benchmark F1 validated against sklearn: NB/DT/LR at ~600K train / ~54K test with realistic label noise; GNB/BN/Histogram in [`fraud.ipynb`](fraud.ipynb) at 500K train / 50K test. Query counts are fixed by feature cardinality, not row count.*

*† Gaussian NB uses only date fields (`month`, `day`, `year`), which are independent of the label in the demo data → ROC-AUC ≈ 0.5, F1 = the majority-class baseline. It demonstrates count-derived Gaussian sufficient statistics, not fraud discrimination.*

*‡ F1 at the tuned threshold. The histogram's risk score averages per-bucket posteriors and is centered on the class prior, so its threshold defaults to the class prior (not 0.5); at 0.5 on this ~65%-positive test set it collapses to always-predict-high (F1 = 0.789 = majority baseline). See "Evaluation under prior shift" below for ROC-AUC / PR-AUC at a production-realistic prior.*

### Training overhead

| Aspect | Plaintext (sklearn / local) | Blind Insight | Overhead |
|--------|----------------------------|--------------|----------|
| **NB Training** | ~0.01s | ~35s (local BI) / ~2min (cloud) | Network round-trips for ~90 queries |
| **DT Training** | ~1.4s | ~3s | Reuses NB marginals, local cross-tabs |
| **LR Training** | ~1.4s | ~2.5s | OLS from aggregate counts + IRLS refinement |
| **Gaussian NB** | ~0.2s | ~16s (local BI, 500K train) | ~96 value-count queries |
| **Bayesian Network** | ~1.2s | ~75s (local BI, 500K train) | ~514 multi-filter CPT queries |
| **Histogram Classifier** | ~3.4s | ~22s (local BI, 500K train) | ~90 marginal queries (same as NB) |
| **Data Exposure** | All records in plaintext | Zero records | - |
| **Compliance** | Requires data access | GDPR / DORA / HIPAA safe | - |

### Evaluation under prior shift

The demo's headline F1 (0.942) is measured on a test set with the **same ~65% high-risk prior** as training. Real fraud is rare — a production fraud team sees on the order of a **1–2% positive rate**. F1 and PR-AUC are prior-sensitive: under prior-corrected posteriors at a ~1.5% prior, F1@best and PR-AUC for these models drop sharply, while **ROC-AUC is prior-invariant** (it ranks by score and is unaffected by the base rate). The takeaway:

- **Report ROC-AUC and PR-AUC alongside F1.** ROC-AUC is the metric that survives deployment at a different prior; PR-AUC shows what precision/recall a team would actually trade off at the production prior.
- **F1 at the balanced demo prior overstates field performance** for any rare-positive deployment. Treat the 0.942 as "discrimination on a balanced split", not "fraud-catch rate in production".
- Numbers at the production prior should be populated from a full-scale re-validation (see the demo's metrics cells); they are intentionally **not** hard-coded here because they depend on the deployed prior.

**There is no accuracy loss from encryption when counts match.** Aggregate counts from Blind Insight are mathematically identical to counts on plaintext, so NB, DT, LR, GNB, and BN learn the same parameters as their benchmarks (0pp F1 gap in validation). The F1 of **0.942** (not 1.000) for NB/DT/LR at ~600K reflects realistic label noise (~17% of training and ~8% of test records have fraud types that conflict with risk level). GNB uses only numeric date fields (`month`, `day`, `year`) and scores lower (F1≈0.789). BN achieves perfect separation on the 500K notebook split.

**The main trade-off is training speed:** Query round-trips dominate. NB and Histogram each need ~90 queries; GNB ~96; BN ~514. DT and LR add **zero** extra BI queries once NB marginals are cached. BN training is the slowest fraud-demo path (~75s local at 500K); expect roughly proportional cloud overhead (~6× NB query count).

---

## What Algorithms Work on Encrypted Data?

Blind Insight provides these operations on encrypted data, all via the **query key only** (no decryption):

| Operation | What it computes |
|-----------|-----------------|
| **count** | Number of records matching filter(s) |
| **avg** | Mean of a numeric field over filtered subset |
| **sum** | Total of a numeric field over filtered subset |
| **min / max** | Extremes of a numeric field over filtered subset |
| **equality** | Exact match on string or numeric fields |
| **comparison** | `>`, `<`, `>=`, `<=` on numeric fields |
| **range** | Numeric range (e.g. `40~60`) |
| **fuzzy match** | Hamming distance and n-gram similarity on strings |

All filters can be combined (e.g. `fraud_type:mule_account,risk_level:count(50~100)`), giving compound conditional statistics.

### Demonstrated (with code and validation)

These algorithms have working implementations in [`fraud.ipynb`](fraud.ipynb) and `blind_ml/`, validated against sklearn or plaintext benchmarks. **NB, DT, LR, GNB, and BN show 0pp F1 gap** when encrypted counts match the local mirror.

| Algorithm | BI operations used | How | Validated |
|-----------|-------------------|-----|-----------|
| **Naive Bayes (categorical)** | count | P(feature\|class) = count(feature AND class) / count(class) | ~600K train, F1=0.942 (matches sklearn) |
| **Decision Trees (Gini/CART)** | count | Gini impurity from class counts per partition. Root split from BI marginals; deeper splits from local cross-tabs (verified 100% match) | ~600K train, F1=0.942 (matches sklearn CART) |
| **Logistic Regression** | count | OLS from X'X and X'y (marginal + pairwise counts), refined via IRLS locally | ~600K train, F1=0.942 (matches sklearn LogisticRegression) |
| **Gaussian Naive Bayes** | count | Per-value class counts on integer fields → sum/sum_sq → μ, σ² per class; sklearn-style variance smoothing | 500K train, F1=0.789 (matches sklearn `GaussianNB`, 0pp gap) |
| **Bayesian Network** | count | CPT cells from multi-filter counts; DAG parent map (e.g. `year→month`, `jurisdiction→bank`) | 500K train, F1=1.000 (matches plaintext BN, 0pp gap) |
| **Histogram Classifier** | count | Smoothed P(high\|feature=value) buckets; weighted average at predict time (no independence assumption); threshold defaults to class prior | 500K train; 0pp encrypted-vs-plaintext by construction (`test_count_parity`) |

### Native Support

These algorithms can train **exactly** on encrypted aggregate counts in principle — no individual record access — but are **not yet implemented** in this repository (except where noted as variants of demonstrated models).

| Algorithm | BI operations used | How |
|-----------|-------------------|-----|
| **Decision Trees (entropy/ID3)** | count | Information gain variant; same approach as demonstrated Gini/CART with a different splitting criterion |
| **Random Forests** | count | Ensemble of decision trees with random feature subsets. Each tree trains from the same aggregate counts with random feature masking. Majority vote across trees |
| **Ridge Regression** | count | Same as logistic regression with L2 penalty: β = (X'X + λI)⁻¹ X'y. Lambda tuned on holdout |
| **AdaBoost (stumps)** | count | Sequential ensemble of depth-1 decision trees. Each stump trains from weighted class counts; weights update based on stump error rate |
| **Gradient Boosted Trees (shallow)** | count | Sequential trees fit to residuals. Each iteration: compute residual distribution from aggregate counts, fit a shallow tree to the residual bins |
| **Statistical Tests** | count | Chi-square, Fisher's exact, etc. from contingency tables built with counts |
| **Association Rules** | count | Support = count(itemset) / count(all) |
| **Anomaly Detection (statistical)** | avg, min, max, count | Thresholds from mean, extremes, and distribution of counts per bin |
| **Similarity Search** | fuzzy match, count | Hamming distance / n-gram matching to find similar records; combine with count for classification |

### Approximate Support

These algorithms can be trained using binned or summarized statistics when per-value enumeration is impractical. For **integer fields with known ranges**, the fraud demo's Gaussian NB path enumerates every value via `count` and is exact (see Algorithm 4).

| Algorithm | BI operations used | Approach |
|-----------|-------------------|----------|
| **Gaussian Naive Bayes (continuous)** | avg, count | Class means via `avg`; variance from binned count histogram when values cannot be enumerated |
| **Linear Discriminant Analysis** | avg, count | Class means via avg; shared covariance from pairwise cross-tabulation counts |
| **PCA (categorical/binned)** | count | X'X (the Gram matrix) can be reconstructed from marginal + pairwise cross-tabulation counts. Eigendecomposition of X'X gives principal components without accessing individual records |
| **K-Means (categorical)** | count | Mode-based clustering: assign records to clusters based on most-frequent feature values per cluster, computed from counts |

### Emerging Support

| Algorithm | BI operations used | Status |
|-----------|-------------------|--------|
| **KNN (string features)** | fuzzy match, count | Fuzzy matching (hamming distance, n-grams) finds records "near" a query — this *is* nearest-neighbor search. Combine with count-by-class among matches for classification. Currently works on string fields; numeric KNN would require numeric distance metrics. See [fuzzy matching demo](https://www.youtube.com/watch?v=ZMBVsJOwJ4k). |
| **Linear SVM** | count | The dual formulation requires record pairs, but the primal form (hinge loss + L2) can be approximated via IRLS-like iterations on aggregate statistics, similar to the logistic regression approach |

### Not Yet Supported (Needs Individual Record Access)

| Algorithm | Why | Potential Path |
|-----------|-----|---------------|
| **Neural Networks** | Needs backpropagation on individual records | Federated learning could compute gradients locally and aggregate them; not yet demonstrated on BI |
| **Kernel SVM** | Needs kernel computations on record pairs | No known aggregate-only formulation for non-linear kernels |
| **Deep Ensembles** | Needs individual record sampling for diversity | Could potentially use bootstrap aggregate statistics, but not yet validated |

---

## Learn More

- **Run the fraud demo** (six models: NB, GNB, BN, DT, LR, Histogram): Open [`fraud.ipynb`](fraud.ipynb)
- **Run the healthcare demo** (breast cancer risk prediction with HIPAA k=11): Open `BreastCancerRiskPrediction.ipynb`
- **Read the code**: See `blind_ml/demo_helpers.py` (fraud) and `blind_ml/healthcare.py` (healthcare) for the implementations
- **Blind Insight Docs**: https://docs.blindinsight.io
- **Key Sharing & Access Controls**: https://docs.blindinsight.io/getting-started/key-sharing/
- **Fuzzy Matching on Encrypted Data (video)**: https://www.youtube.com/watch?v=ZMBVsJOwJ4k
- **Aggregate Functions on Encrypted Data (video)**: https://www.youtube.com/watch?v=q6Eno3s8fyU
- **How Blind Insight Works (video)**: https://www.youtube.com/watch?v=rwHFgWVK0Nc
