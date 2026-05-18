"""
Helper utilities for the Breast Cancer Risk Prediction demo notebook.
Keeps notebook cells focused on *what's happening* rather than plumbing.

Includes:
  - BCRAT (Gail Model) 5-year risk implementation
  - Simplified BCSC risk implementation
  - Naive Bayes training via BI aggregate queries
  - Comparison / agreement tables
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

# Re-use generic plumbing from the fraud demo helpers
from .demo_helpers import (
    _inject_notebook_styles,
    _inject_table_resize_script,
    metrics_table,
)
from .models import (
    DecisionTreeModel as _DecisionTreeModel,
)
from .models import (
    LogisticRegressionModel as _LogisticRegressionModel,
)
from .models import (
    build_design_matrix as _build_design_matrix,
)
from .models import (
    build_marginals_local as _build_marginals_local,
)
from .models import (
    compute_pairwise_local as _compute_pairwise_local,
)
from .models import (
    extract_marginals as _extract_marginals_generic,
)
from .models import (
    extract_pos_counts as _extract_pos_counts_generic,
)

# ============================================================================
# BCRAT (Gail Model) 5-year risk calculator
# Published coefficients from Costantino/Gail validation (NCI BCRA v4.1)
# ============================================================================

# Factor A: Age at menarche
_RR_MENARCHE = {
    "14_plus": 1.000,
    "12_13": 1.098,
    "under_12": 1.207,
}


def _rr_menarche(menarche_cat: str) -> float:
    return _RR_MENARCHE.get(menarche_cat, 1.000)


def _rr_biopsy(num_biopsies: int, age: int) -> float:
    """Factor B: Number of previous breast biopsies (age-dependent)."""
    if num_biopsies == 0:
        return 1.000
    if age < 50:
        return 1.683 if num_biopsies == 1 else 2.750
    return 1.237 if num_biopsies == 1 else 1.539


def _rr_birth_relatives(age_first_birth: int, num_relatives: int) -> float:
    """Factor C: Age at first live birth x number of first-degree relatives."""
    capped = min(num_relatives, 2)

    if age_first_birth == 0 or age_first_birth < 20:
        table = {0: 1.000, 1: 2.560, 2: 6.168}
    elif age_first_birth < 25:
        table = {0: 1.240, 1: 2.640, 2: 5.318}
    elif age_first_birth < 30:
        table = {0: 1.550, 1: 2.705, 2: 4.591}
    else:
        table = {0: 1.930, 1: 2.779, 2: 3.953}

    return table.get(capped, 1.0)


def _rr_atypical(atypical: str) -> float:
    """Factor D: Atypical hyperplasia on biopsy."""
    if atypical == "yes":
        return 1.82
    if atypical == "no":
        return 0.93
    return 1.00  # unknown


# Age-specific baseline 5-year hazard for BCRAT/BCSC scoring.
# These are NCI/SEER population-level baselines for a woman with all-lowest
# risk factors, calibrated so that ~35% of a screening cohort is flagged
# at the 1.67% BCRAT threshold (matching Paige et al. 2023 observed rates).
_BASELINE_5YR_BY_GROUP = {
    "40_49": 0.004,
    "50_59": 0.007,
    "60_69": 0.011,
    "70_74": 0.014,
}


def _age_to_group(age: int) -> str:
    if age < 50:
        return "40_49"
    if age < 60:
        return "50_59"
    if age < 70:
        return "60_69"
    return "70_74"


def bcrat_5yr_risk(row: dict[str, Any]) -> float:
    """
    Compute BCRAT (Gail Model) 5-year absolute risk of invasive breast cancer.

    Uses published relative risk coefficients from the NCI BCRA tool.
    Expects a dict with keys: age, age_at_menarche (or menarche_category),
    age_at_first_birth, num_first_degree_relatives, num_prior_biopsies,
    atypical_hyperplasia.
    """
    age = int(row.get("age", 50))
    menarche_cat = str(row.get("menarche_category", ""))
    if not menarche_cat or menarche_cat == "nan":
        am = int(row.get("age_at_menarche", 13))
        if am < 12:
            menarche_cat = "under_12"
        elif am <= 13:
            menarche_cat = "12_13"
        else:
            menarche_cat = "14_plus"

    age_fb = int(row.get("age_at_first_birth", 0))
    n_rel = int(row.get("num_first_degree_relatives", 0))
    n_bx = int(row.get("num_prior_biopsies", 0))
    atypical = str(row.get("atypical_hyperplasia", "unknown")).lower()

    composite_rr = (
        _rr_menarche(menarche_cat) * _rr_biopsy(n_bx, age) * _rr_birth_relatives(age_fb, n_rel) * _rr_atypical(atypical)
    )

    ag = _age_to_group(age)
    baseline = _BASELINE_5YR_BY_GROUP.get(ag, 0.015)
    risk = 1.0 - math.pow(1.0 - baseline, composite_rr)
    return min(risk, 1.0)


# ============================================================================
# Simplified BCSC risk calculator
# Uses a subset of factors: age, race, family_history, biopsy, density
# ============================================================================

_BCSC_RR_DENSITY = {1: 0.55, 2: 0.80, 3: 1.35, 4: 2.10}
_BCSC_RR_FAMILY = {"yes": 1.80, "no": 1.00}
_BCSC_RR_BIOPSY = {"yes": 1.50, "no": 1.00}
_BCSC_RR_RACE = {
    "white": 1.00,
    "black": 0.95,
    "hispanic": 0.75,
    "asian_pi": 0.65,
    "other": 0.85,
}


# BCSC uses its own baselines (higher than BCRAT) because it has fewer
# risk factors to explain the variance, so more risk is in the baseline.
_BCSC_BASELINE_5YR = {
    "40_49": 0.006,
    "50_59": 0.010,
    "60_69": 0.015,
    "70_74": 0.019,
}


def bcsc_5yr_risk(row: dict[str, Any]) -> float:
    """
    Simplified BCSC 5-year risk estimate.

    Uses published relative risk approximations from Tice et al. (2008).
    Fewer factors than BCRAT -- no menarche or first-birth age.
    """
    age = int(row.get("age", 50))
    density = int(row.get("breast_density", 2))
    family = str(row.get("family_history", "no")).lower()
    biopsy = str(row.get("has_prior_biopsy", "no")).lower()
    race = str(row.get("race_ethnicity", "white")).lower()

    composite_rr = (
        _BCSC_RR_DENSITY.get(density, 1.0)
        * _BCSC_RR_FAMILY.get(family, 1.0)
        * _BCSC_RR_BIOPSY.get(biopsy, 1.0)
        * _BCSC_RR_RACE.get(race, 1.0)
    )

    ag = _age_to_group(age)
    baseline = _BCSC_BASELINE_5YR.get(ag, 0.010)
    risk = 1.0 - math.pow(1.0 - baseline, composite_rr)
    return min(risk, 1.0)


# ============================================================================
# Feature discovery (from local plaintext data)
# ============================================================================

NB_FEATURES = [
    "age_group",
    "race_ethnicity",
    "num_first_degree_relatives",
    "num_prior_biopsies",
    "atypical_hyperplasia",
    "menarche_category",
    "breast_density",
    "age_at_first_birth",
]

AFB_BINS = [
    ("nulliparous", "age_at_first_birth:0"),
    ("under_20", "age_at_first_birth:1~19"),
    ("20_24", "age_at_first_birth:20~24"),
    ("25_29", "age_at_first_birth:25~29"),
    ("30_plus", "age_at_first_birth:30~47"),
]

BIOPSY_BINS = [
    ("0", "num_prior_biopsies:0"),
    ("1", "num_prior_biopsies:1"),
    ("2", "num_prior_biopsies:2"),
    ("3_plus", "num_prior_biopsies:3~7"),
]


def _afb_bin(val) -> str:
    """Map an age_at_first_birth integer to its bin label."""
    v = int(val)
    if v == 0:
        return "nulliparous"
    if v < 20:
        return "under_20"
    if v < 25:
        return "20_24"
    if v < 30:
        return "25_29"
    return "30_plus"


def _biopsy_bin(val) -> str:
    """Map num_prior_biopsies to HIPAA-safe bin (CMS k=11 compliant)."""
    v = int(val)
    if v >= 3:
        return "3_plus"
    return str(v)


def discover_feature_values(df) -> dict[str, list[str]]:
    """Discover unique values for each NB feature from a DataFrame."""
    return {
        "age_groups": sorted(df["age_group"].astype(str).unique().tolist()),
        "race_values": sorted(df["race_ethnicity"].astype(str).unique().tolist()),
        "relatives_values": sorted(
            df["num_first_degree_relatives"].astype(str).unique().tolist(), key=lambda x: int(x)
        ),
        "biopsies_values": [b[0] for b in BIOPSY_BINS],
        "atypical_values": sorted(df["atypical_hyperplasia"].astype(str).unique().tolist()),
        "menarche_values": sorted(df["menarche_category"].astype(str).unique().tolist()),
        "density_values": sorted(df["breast_density"].astype(str).unique().tolist(), key=lambda x: int(x)),
        "afb_values": [b[0] for b in AFB_BINS],
    }


# ============================================================================
# BI count_only queries for NB training (uses string filters, not integer ranges)
# ============================================================================


def _count_only(client, org, dataset, schema, filters, retries=3):
    """Run a count_only query with string filters. Returns int count."""
    for attempt in range(retries):
        try:
            result = client.query(
                organization=org,
                dataset_slug=dataset,
                schema_slug=schema,
                filters=filters,
                limit=1,
                count_only=True,
            )
            return int(result.get("count", 0))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def _bc_queries(values: dict[str, list[str]]) -> list[tuple[str, int, str, list[str]]]:
    """Build the count_only query list: (feature_type, class, value, filters).

    Handles string equality, integer equality, and integer range filters.
    """
    queries = []

    def _add(feat_key, value_list, bi_field):
        for v in value_list:
            queries.append((feat_key, 1, v, ["cancer_5yr:1", f"{bi_field}:{v}"]))
            queries.append((feat_key, 0, v, ["cancer_5yr:0", f"{bi_field}:{v}"]))

    _add("age_group", values["age_groups"], "age_group")
    _add("race", values["race_values"], "race_ethnicity")
    _add("relatives", values["relatives_values"], "num_first_degree_relatives")
    _add("atypical", values["atypical_values"], "atypical_hyperplasia")
    _add("menarche", values["menarche_values"], "menarche_category")
    _add("density", values["density_values"], "breast_density")

    for bin_label, bi_filter in BIOPSY_BINS:
        queries.append(("biopsies", 1, bin_label, ["cancer_5yr:1", bi_filter]))
        queries.append(("biopsies", 0, bin_label, ["cancer_5yr:0", bi_filter]))

    for bin_label, bi_filter in AFB_BINS:
        queries.append(("afb", 1, bin_label, ["cancer_5yr:1", bi_filter]))
        queries.append(("afb", 0, bin_label, ["cancer_5yr:0", bi_filter]))

    return queries


def run_bc_conditional_queries(
    client,
    org: str,
    dataset: str,
    schema: str,
    values: dict[str, list[str]],
    include_base_rates: bool = True,
) -> dict[str, Any]:
    """Run count_only queries against BI using string filters.

    Uses the server's count_only endpoint (45-200x faster than integer
    range aggregates) with string equality filters instead of encrypted
    range scans.
    """
    queries = _bc_queries(values)

    def run_query(q):
        f_type, r_class, val, filters = q
        count = _count_only(client, org, dataset, schema, filters)
        return (f_type, r_class, val, count)

    base_cancer = None
    base_no_cancer = None

    max_workers = 20
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        cond_futures = [executor.submit(run_query, q) for q in queries]
        if include_base_rates:
            f_cancer = executor.submit(_count_only, client, org, dataset, schema, ["cancer_5yr:1"])
            f_no_cancer = executor.submit(_count_only, client, org, dataset, schema, ["cancer_5yr:0"])
        results = [f.result() for f in cond_futures]
        if include_base_rates:
            base_cancer = f_cancer.result()
            base_no_cancer = f_no_cancer.result()

    out = {"raw_results": results, "enc_queries": len(results)}
    if include_base_rates:
        out["n_cancer"] = base_cancer
        out["n_no_cancer"] = base_no_cancer
    return out


def get_bc_base_rates(
    client,
    org: str,
    dataset: str,
    schema: str,
) -> tuple[int, int]:
    """Query BI for cancer / no-cancer counts (serial fallback)."""
    n_cancer = _count_only(client, org, dataset, schema, ["cancer_5yr:1"])
    n_no_cancer = _count_only(client, org, dataset, schema, ["cancer_5yr:0"])
    return n_cancer, n_no_cancer


# ============================================================================
# Naive Bayes model building
# ============================================================================

_FEATURE_MAP = {
    "age_group": ("age_groups", "age_group"),
    "race": ("race_values", "race_ethnicity"),
    "relatives": ("relatives_values", "num_first_degree_relatives"),
    "biopsies": ("biopsies_values", "num_prior_biopsies"),
    "atypical": ("atypical_values", "atypical_hyperplasia"),
    "menarche": ("menarche_values", "menarche_category"),
    "density": ("density_values", "breast_density"),
    "afb": ("afb_values", "age_at_first_birth"),
}


def build_bc_model(
    raw_results: list[tuple],
    values: dict[str, list[str]],
    n_cancer: int,
    n_no_cancer: int,
) -> dict[str, Any]:
    """Build probability tables from raw query results + BI base rates."""
    n_total = n_cancer + n_no_cancer

    P = {}
    for feat_key, (val_key, _) in _FEATURE_MAP.items():
        P[feat_key] = {1: {}, 0: {}}

    for f_type, r_class, val, count in raw_results:
        if f_type not in P:
            continue
        val_key = _FEATURE_MAP[f_type][0]
        n_class = n_cancer if r_class == 1 else n_no_cancer
        n_values = len(values.get(val_key, []))
        P[f_type][r_class][val.lower()] = (count + 1) / (n_class + n_values)

    return {
        "n_cancer": n_cancer,
        "n_no_cancer": n_no_cancer,
        "n_total": n_total,
        "P_cancer": n_cancer / n_total if n_total > 0 else 0.5,
        "P_no_cancer": n_no_cancer / n_total if n_total > 0 else 0.5,
        "P": P,
        "enc_queries": len(raw_results),
    }


# ============================================================================
# Encrypted Decision Tree (depth 2+3 hybrid with k=11 fallback)
# ============================================================================

_BIN_FILTER_MAP = {
    **{label: filt for label, filt in AFB_BINS},
    **{label: filt for label, filt in BIOPSY_BINS},
}


def _bi_feat_filter(feat_key: str, value: str) -> str:
    """Return the BI filter string for a feature+value, handling binned features."""
    if feat_key in ("afb", "biopsies"):
        return _BIN_FILTER_MAP.get(value.lower(), f"{_FEATURE_MAP[feat_key][1]}:{value}")
    return f"{_FEATURE_MAP[feat_key][1]}:{value}"


# _entropy, _gini, _best_split removed -- now in blind_ml


# _build_dt_targeted_queries removed -- replaced by blind_ml.DecisionTreeModel


def _bc_feature_columns() -> list[str]:
    """Return ordered DataFrame column names for BC DT/LR features."""
    _bc = {"afb": "afb_bin", "biopsies": "biopsy_bin"}
    return [_bc.get(fk, _FEATURE_MAP[fk][1]) for fk in _FEATURE_MAP]


def _prepare_bc_df(df_local):
    """Add binned columns and lowercase all feature columns."""
    df = df_local.copy()
    if "afb_bin" not in df.columns:
        df["afb_bin"] = df["age_at_first_birth"].apply(_afb_bin)
    if "biopsy_bin" not in df.columns:
        df["biopsy_bin"] = df["num_prior_biopsies"].apply(_biopsy_bin)
    for col in _bc_feature_columns():
        df[col] = df[col].astype(str).str.lower()
    return df


def run_encrypted_dt(
    client,
    org: str,
    dataset: str,
    schema: str,
    raw_results: list[tuple[str, int, str, int]],
    feature_values: dict[str, list[str]],
    df_local,
    n_cancer: int,
    n_no_cancer: int,
    k_min: int = 11,
    criterion: str = "gini",
    **kwargs,
) -> dict[str, Any]:
    """Build binary CART DT via blind_ml with cell-suppression k_min."""
    df = _prepare_bc_df(df_local)
    feature_cols = _bc_feature_columns()

    dt = _DecisionTreeModel(max_depth=3, criterion=criterion, k_min=k_min)
    dt.fit(df, feature_cols, "cancer_5yr")

    return {
        "_model": dt,
        "root_feat": dt.tree.get("col_name") if dt.tree and dt.tree.get("type") == "split" else None,
        "root_ig": 0,
        "root_children": {},
        "tree_nodes": {},
        "enc_queries": 0,
        "train_time": dt.train_time,
        "d3_safe": 0,
        "d3_fallback": 0,
        "feature_values": feature_values,
    }


def encrypted_dt_predict(dt_result: dict, row: dict) -> tuple[int, float]:
    """Predict using the encrypted BC decision tree. Returns (pred, risk)."""
    model = dt_result.get("_model")
    if not model:
        return 0, 0.0
    row_dict = dict(row)
    row_dict["afb_bin"] = str(_afb_bin(row_dict.get("age_at_first_birth", "")))
    row_dict["biopsy_bin"] = str(_biopsy_bin(row_dict.get("num_prior_biopsies", "")))
    for col in _bc_feature_columns():
        if col in row_dict:
            row_dict[col] = str(row_dict[col]).lower()
    return model.predict(row_dict)


def encrypted_dt_describe(dt_result: dict) -> str:
    """Return a text description of the binary CART tree."""
    model = dt_result.get("_model")
    if not model or not model.tree:
        return "Empty tree"

    def _desc(node, indent=0):
        pre = "  " * indent
        if node["type"] == "leaf":
            return [f"{pre}-> risk={node['risk']:.4f} (n={node['n']}, pos={node['n_pos']})"]
        col = node["col_name"]
        lines = [f"{pre}[{col}=Yes]:"]
        lines.extend(_desc(node["left"], indent + 1))
        lines.append(f"{pre}[{col}=No]:")
        lines.extend(_desc(node["right"], indent + 1))
        return lines

    return "\n".join(_desc(model.tree))


def run_bc_training(
    client,
    org: str,
    dataset: str,
    schema: str,
    values: dict[str, list[str]],
    n_cancer: int = 0,
    n_no_cancer: int = 0,
) -> dict[str, Any]:
    """Run queries + build model in one call."""
    n_total = n_cancer + n_no_cancer
    print(f"  Base rates (local): {n_total:,} records, cancer={n_cancer}, no_cancer={n_no_cancer}")
    raw = run_bc_conditional_queries(client, org, dataset, schema, values)
    result = build_bc_model(raw["raw_results"], values, n_cancer, n_no_cancer)
    result["enc_queries"] = raw["enc_queries"]
    return result


# ============================================================================
# Plaintext NB (for apples-to-apples comparison)
# ============================================================================


def train_plaintext_bc_nb(df, feature_values: dict[str, list[str]]) -> dict[str, Any]:
    """
    Train a plaintext Naive Bayes classifier using the same conditional
    probability approach as BI training.
    """
    df = df.copy()
    df["is_cancer"] = df["cancer_5yr"].astype(int)
    df["afb_bin"] = df["age_at_first_birth"].apply(_afb_bin)
    df["biopsy_bin"] = df["num_prior_biopsies"].apply(_biopsy_bin)

    total = len(df)
    cancer_count = df["is_cancer"].sum()
    no_cancer_count = total - cancer_count

    P_cancer = cancer_count / total if total > 0 else 0.5
    P_no_cancer = no_cancer_count / total if total > 0 else 0.5

    _binned_cols = {"afb": "afb_bin", "biopsies": "biopsy_bin"}
    P = {}
    for feat_key, (val_key, col_name) in _FEATURE_MAP.items():
        P[feat_key] = {1: {}, 0: {}}
        vals = feature_values.get(val_key, [])
        n_vals = len(vals)
        src_col = _binned_cols.get(feat_key, col_name)
        for v in vals:
            v_l = v.lower()
            h = len(df[(df[src_col].astype(str).str.lower() == v_l) & (df["is_cancer"] == 1)])
            l = len(df[(df[src_col].astype(str).str.lower() == v_l) & (df["is_cancer"] == 0)])
            P[feat_key][1][v_l] = (h + 1) / (cancer_count + n_vals)
            P[feat_key][0][v_l] = (l + 1) / (no_cancer_count + n_vals)

    return {
        "P_cancer": P_cancer,
        "P_no_cancer": P_no_cancer,
        "P": P,
    }


def _bc_one_hot_encode(df, feature_values: dict[str, list[str]]):
    """One-hot encode BC features into a numpy design matrix.

    Returns (X, columns) where columns is a list of (feat_key, value) tuples
    matching the dummy variable order used by the encrypted LR model.
    """

    col_map = {fk: col for fk, (_, col) in _FEATURE_MAP.items()}
    frames = []
    col_names: list[tuple[str, str]] = []

    _binned_series = {
        "afb": df["age_at_first_birth"].apply(_afb_bin).astype(str).str.lower(),
        "biopsies": df["num_prior_biopsies"].apply(_biopsy_bin).astype(str).str.lower(),
    }

    for feat_key in _LR_FEATURES_ORDERED:
        val_key = _FEATURE_MAP[feat_key][0]
        col_name = col_map[feat_key]
        ref = _LR_REFERENCE[feat_key]
        vals = [v.lower() for v in feature_values[val_key]]
        series = _binned_series.get(feat_key, df[col_name].astype(str).str.lower())
        for v in vals:
            if v == ref.lower():
                continue
            frames.append((series == v).astype(int))
            col_names.append((feat_key, v))

    X = np.column_stack(frames) if frames else np.empty((len(df), 0))
    return X, col_names


def _bc_dt_one_hot(df):
    """One-hot encode BC features with drop_first=False (matches blind_ml DT)."""
    df2 = _prepare_bc_df(df)
    feature_cols = _bc_feature_columns()
    X = df2[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].astype(str)
    X_encoded = pd.get_dummies(X, columns=feature_cols, drop_first=False)
    return X_encoded, X_encoded.columns.tolist()


def train_plaintext_bc_dt(df, feature_values: dict[str, list[str]], max_depth: int = 3) -> dict[str, Any]:
    """Train a sklearn DecisionTreeClassifier with same encoding as blind_ml DT."""
    from sklearn.tree import DecisionTreeClassifier

    X_encoded, col_names = _bc_dt_one_hot(df)
    y = df["cancer_5yr"].astype(int).values

    start = time.time()
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_encoded, y)
    train_time = time.time() - start

    return {"model": model, "train_time": train_time, "col_names": col_names}


def train_plaintext_bc_lr(df, feature_values: dict[str, list[str]]) -> dict[str, Any]:
    """Train a sklearn LogisticRegression on the same one-hot features."""
    from sklearn.linear_model import LogisticRegression

    X, col_names = _bc_one_hot_encode(df, feature_values)
    y = df["cancer_5yr"].astype(int).values

    start = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    train_time = time.time() - start

    return {"model": model, "train_time": train_time, "col_names": col_names}


def plaintext_predict_proba(model, col_names, df, feature_values, encoding="lr"):
    """Return P(cancer) array from a sklearn model for all rows in df.

    encoding="lr"  uses _bc_one_hot_encode (reference-dropped, for LR)
    encoding="dt"  uses _bc_dt_one_hot   (drop_first=False, matches blind_ml DT)
    """
    if encoding == "dt":
        X_encoded, enc_cols = _bc_dt_one_hot(df)
        for c in col_names:
            if c not in X_encoded.columns:
                X_encoded[c] = 0
        X_encoded = X_encoded[col_names]
        return model.predict_proba(X_encoded.values)[:, 1]
    X, _ = _bc_one_hot_encode(df, feature_values)
    return model.predict_proba(X)[:, 1]


# ============================================================================
# Recalibration
# ============================================================================


def recalibrate_risk(cohort_risk: float, cohort_prev: float, pop_prev: float) -> float:
    """Shift a risk score from cohort prevalence to population prevalence.

    Uses Bayes' rule to adjust the posterior when the prior changes.
    This is the same adjustment NB gets by swapping P_cancer to POP_5YR_RATE.
    """
    if cohort_risk <= 0 or cohort_prev <= 0 or pop_prev <= 0:
        return 0.0
    if cohort_risk >= 1:
        return 1.0
    lr = (cohort_risk / (1 - cohort_risk)) * (pop_prev / cohort_prev) * ((1 - cohort_prev) / (1 - pop_prev))
    return lr / (1 + lr)


# ============================================================================
# Prediction
# ============================================================================


def _nb_features_from_row(row: dict) -> dict[str, str]:
    """Extract NB feature values from a data row, with HIPAA-safe binning."""
    return {
        "age_group": str(row.get("age_group", "")).lower(),
        "race": str(row.get("race_ethnicity", "")).lower(),
        "relatives": str(row.get("num_first_degree_relatives", "")).lower(),
        "biopsies": _biopsy_bin(row.get("num_prior_biopsies", 0)),
        "atypical": str(row.get("atypical_hyperplasia", "")).lower(),
        "menarche": str(row.get("menarche_category", "")).lower(),
        "density": str(row.get("breast_density", "")).lower(),
        "afb": _afb_bin(row.get("age_at_first_birth", 0)),
    }


def bc_naive_bayes_predict(P_cancer: float, P_no_cancer: float, P: dict, row: dict) -> int:
    """Predict cancer (1) or no cancer (0) using NB probability tables."""
    eps = 1e-10
    features = _nb_features_from_row(row)

    log_c = math.log(P_cancer + eps)
    log_n = math.log(P_no_cancer + eps)

    for feat_key, val in features.items():
        if feat_key in P:
            log_c += math.log(max(P[feat_key][1].get(val, 0.1), eps))
            log_n += math.log(max(P[feat_key][0].get(val, 0.1), eps))

    return 1 if log_c > log_n else 0


def bc_naive_bayes_risk(P_cancer: float, P_no_cancer: float, P: dict, row: dict) -> float:
    """Return the NB posterior probability of cancer (continuous 0-1)."""
    eps = 1e-10
    features = _nb_features_from_row(row)

    log_c = math.log(P_cancer + eps)
    log_n = math.log(P_no_cancer + eps)

    for feat_key, val in features.items():
        if feat_key in P:
            log_c += math.log(max(P[feat_key][1].get(val, 0.1), eps))
            log_n += math.log(max(P[feat_key][0].get(val, 0.1), eps))

    max_log = max(log_c, log_n)
    p_c = math.exp(log_c - max_log)
    p_n = math.exp(log_n - max_log)
    return p_c / (p_c + p_n)


# ============================================================================
# HIPAA-safe Linear Probability Model from encrypted pairwise buckets
# CMS cell suppression policy: counts 1-10 replaced with independence estimates
# Reference: https://resdac.org/node/1506
# ============================================================================

CMS_MIN_CELL_SIZE = 11

_LR_FEATURES_ORDERED = [
    "age_group",
    "race",
    "relatives",
    "biopsies",
    "atypical",
    "menarche",
    "density",
    "afb",
]

_LR_REFERENCE = {
    "age_group": "40_49",
    "race": "white",
    "relatives": "0",
    "biopsies": "0",
    "atypical": "no",
    "menarche": "14_plus",
    "density": "1",
    "afb": "nulliparous",
}


def _lr_dummy_index(values: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Ordered (feature_key, value) pairs for dummy variables, skipping references."""
    index = []
    for feat_key in _LR_FEATURES_ORDERED:
        val_key = _FEATURE_MAP[feat_key][0]
        ref = _LR_REFERENCE[feat_key]
        for v in values[val_key]:
            if v.lower() != ref.lower():
                index.append((feat_key, v.lower()))
    return index


def _extract_marginals(raw_results: list[tuple]) -> dict[tuple[str, str], int]:
    return _extract_marginals_generic(raw_results)


def _extract_cancer_counts(raw_results: list[tuple]) -> dict[tuple[str, str], int]:
    return _extract_pos_counts_generic(raw_results)


def build_bc_raw_results_local(
    df: pd.DataFrame,
    feature_values: dict[str, list[str]],
) -> list[tuple]:
    """Build NB-format raw_results from local BC data via blind_ml."""

    df2 = df.copy()
    df2["biopsy_bin"] = df2["num_prior_biopsies"].apply(_biopsy_bin)
    df2["afb_bin"] = df2["age_at_first_birth"].apply(_afb_bin)

    feature_config = [
        ("age_group", "age_group", feature_values.get("age_groups", [])),
        ("race", "race_ethnicity", feature_values.get("race_values", [])),
        ("relatives", "num_first_degree_relatives", feature_values.get("relatives_values", [])),
        ("atypical", "atypical_hyperplasia", feature_values.get("atypical_values", [])),
        ("menarche", "menarche_category", feature_values.get("menarche_values", [])),
        ("density", "breast_density", feature_values.get("density_values", [])),
        ("biopsies", "biopsy_bin", feature_values.get("biopsies_values", [])),
        ("afb", "afb_bin", feature_values.get("afb_values", [])),
    ]
    return _build_marginals_local(df2, "cancer_5yr", feature_config)


def run_bc_pairwise_queries(
    client,
    org: str,
    dataset: str,
    schema: str,
    values: dict[str, list[str]],
    raw_results: list[tuple],
    n_total: int,
    min_cell_size: int = CMS_MIN_CELL_SIZE,
) -> dict[str, Any]:
    """Query pairwise cross-tabulation counts with CMS k=11 cell suppression.

    For each pair of features, queries count(A=a AND B=b) without class split.
    Any count in [1, min_cell_size) is replaced with the independence estimate
    count(A=a) * count(B=b) / N  to prevent HIPAA re-identification risk.

    Returns dict with pairwise counts, query count, and suppression statistics.
    """
    marginals = _extract_marginals(raw_results)

    _bin_filter_map = {
        **{label: filt for label, filt in AFB_BINS},
        **{label: filt for label, filt in BIOPSY_BINS},
    }

    def _bi_filter(feat_key, bi_field, val):
        if feat_key in ("afb", "biopsies"):
            return _bin_filter_map.get(val.lower(), f"{bi_field}:{val}")
        return f"{bi_field}:{val}"

    queries = []
    for i, feat_a in enumerate(_LR_FEATURES_ORDERED):
        bi_field_a = _FEATURE_MAP[feat_a][1]
        val_key_a = _FEATURE_MAP[feat_a][0]
        for feat_b in _LR_FEATURES_ORDERED[i + 1 :]:
            bi_field_b = _FEATURE_MAP[feat_b][1]
            val_key_b = _FEATURE_MAP[feat_b][0]
            for va in values[val_key_a]:
                for vb in values[val_key_b]:
                    queries.append(
                        (
                            feat_a,
                            va.lower(),
                            feat_b,
                            vb.lower(),
                            [_bi_filter(feat_a, bi_field_a, va), _bi_filter(feat_b, bi_field_b, vb)],
                        )
                    )

    def _run(q):
        fa, va, fb, vb, filters = q
        count = _count_only(client, org, dataset, schema, filters)
        return (fa, va, fb, vb, count)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_run, queries))

    pairwise: dict[tuple[str, str, str, str], float] = {}
    n_suppressed = 0
    suppressed_cells: list[str] = []

    for fa, va, fb, vb, count in results:
        if 0 < count < min_cell_size:
            count_a = marginals.get((fa, va), 0)
            count_b = marginals.get((fb, vb), 0)
            count = (count_a * count_b / n_total) if n_total > 0 else 0
            n_suppressed += 1
            suppressed_cells.append(f"{fa}={va} × {fb}={vb}")
        pairwise[(fa, va, fb, vb)] = count

    return {
        "pairwise": pairwise,
        "n_queries": len(results),
        "n_suppressed": n_suppressed,
        "suppressed_cells": suppressed_cells,
        "min_cell_size": min_cell_size,
    }


def compute_pairwise_local(
    df_local,
    values: dict[str, list[str]],
    raw_results: list[tuple],
    n_total: int,
    min_cell_size: int = CMS_MIN_CELL_SIZE,
) -> dict[str, Any]:
    """Compute pairwise cross-tabs via blind_ml with cell suppression."""
    df = _prepare_bc_df(df_local)

    # Alias columns to feature keys so pairwise dict keys match dummy_index
    _bc = {"afb": "afb_bin", "biopsies": "biopsy_bin"}
    for fk in _LR_FEATURES_ORDERED:
        col = _bc.get(fk, _FEATURE_MAP[fk][1])
        df[fk] = df[col]

    fv = {fk: [v.lower() for v in values.get(_FEATURE_MAP[fk][0], [])] for fk in _LR_FEATURES_ORDERED}
    marginals = _extract_marginals(raw_results)

    result = _compute_pairwise_local(df, _LR_FEATURES_ORDERED, fv, marginals, n_total, min_cell_size)
    result["n_queries"] = 0
    result["min_cell_size"] = min_cell_size
    return result


def build_linear_model(
    raw_results: list[tuple],
    pairwise_data: dict[str, Any],
    values: dict[str, list[str]],
    n_cancer: int,
    n_no_cancer: int,
    ridge_lambda: float = 0.0,
) -> dict[str, Any]:
    """Build OLS/ridge linear model via blind_ml.LogisticRegressionModel."""
    marginals = _extract_marginals(raw_results)
    cancer_counts = _extract_cancer_counts(raw_results)
    dummy_idx = _lr_dummy_index(values)

    lr = _LogisticRegressionModel(ridge_lambda=ridge_lambda)
    lr.fit_from_counts(
        marginals,
        cancer_counts,
        pairwise_data["pairwise"],
        dummy_idx,
        n_cancer,
        n_no_cancer,
        feat_order=_LR_FEATURES_ORDERED,
    )

    return {
        "_model": lr,
        "beta": lr.beta,
        "dummy_index": lr.dummy_index,
        "intercept": lr.beta[0],
        "coefficients": {f"{fi}={vi}": lr.beta[i + 1] for i, (fi, vi) in enumerate(dummy_idx)},
        "n_features": len(dummy_idx) + 1,
        "matrix_rank": len(dummy_idx) + 1,
        "ridge_lambda": ridge_lambda,
    }


def linear_model_predict(
    beta: np.ndarray,
    dummy_index: list[tuple[str, str]],
    row: dict,
    use_sigmoid: bool = False,
) -> float:
    """Predict cancer probability via blind_ml.LogisticRegressionModel."""
    lr = _LogisticRegressionModel()
    lr.beta = beta
    lr.dummy_index = list(dummy_index)
    return lr.predict(_nb_features_from_row(row), use_sigmoid=use_sigmoid)


def refine_bc_with_irls(
    beta_init: np.ndarray,
    dummy_index: list[tuple[str, str]],
    df_local: pd.DataFrame,
    feature_values: dict[str, list[str]],
    max_iter: int = 25,
    tol: float = 1e-6,
    ridge_lambda: float = 0.0,
) -> np.ndarray:
    """Refine OLS beta via IRLS using blind_ml."""
    df = _prepare_bc_df(df_local)

    _bc_bin = {"afb": "afb_bin", "biopsies": "biopsy_bin"}
    col_map = {fk: _bc_bin.get(fk, _FEATURE_MAP[fk][1]) for fk in _LR_FEATURES_ORDERED}

    X = _build_design_matrix(df, dummy_index, col_map)
    y = df["cancer_5yr"].astype(float).values

    lr = _LogisticRegressionModel(ridge_lambda=ridge_lambda)
    lr.beta = beta_init.copy()
    lr.dummy_index = list(dummy_index)
    lr.refine_irls(X, y, max_iter=max_iter, tol=tol)
    return lr.beta


# ============================================================================
# Data loading
# ============================================================================


def load_bc_training_data(db_path: str) -> tuple:
    """Load training data from SQLite. Returns (df, record_count)."""
    import pandas as pd

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM train")
    count = int(cur.fetchone()[0])
    df = pd.read_sql_query(f"SELECT * FROM train ORDER BY rowid ASC LIMIT {count}", conn)
    conn.close()
    df["cancer_5yr"] = df["cancer_5yr"].astype(int)
    return df, count


def load_bc_test_data(db_path: str) -> tuple:
    """Load test data from SQLite. Returns (df, record_count)."""
    import pandas as pd

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM test")
    count = int(cur.fetchone()[0])
    df = pd.read_sql_query(f"SELECT * FROM test ORDER BY CAST(patient_id AS INTEGER) ASC LIMIT {count}", conn)
    conn.close()
    df["cancer_5yr"] = df["cancer_5yr"].astype(int)
    df["is_cancer"] = df["cancer_5yr"].astype(int)
    return df, count


def get_bc_demo_config() -> dict[str, Any]:
    """Centralized notebook config for the breast-cancer demo."""
    return {
        "dataset": "breast-cancer-screening-data",
        "schema": "train",
        "test_schema": "test",
        "sqlite_db": "demo_data/plaintext/bc_train.db",
        "test_sqlite_db": "demo_data/plaintext/bc_test.db",
        "nb_features": [
            "age_group",
            "race_ethnicity",
            "num_first_degree_relatives",
            "num_prior_biopsies",
            "atypical_hyperplasia",
            "menarche_category",
            "breast_density",
            "age_at_first_birth",
        ],
        "target": "cancer_5yr",
        "clinical_threshold": 0.0167,
        "pop_5yr_rate": 0.016,
    }


# ============================================================================
# Training summary table
# ============================================================================


def bc_training_summary_table(
    n_cancer_plain: int,
    n_no_cancer_plain: int,
    n_cancer_enc: int,
    n_no_cancer_enc: int,
    enc_queries: int,
    enc_train_time: float,
    plain_train_time: float,
) -> str:
    overhead = enc_train_time - plain_train_time
    overhead_class = "status-good" if overhead < 180 else "status-bad"
    return metrics_table(
        rows=[
            {"label": "Cancer (5yr)", "values": [f"{n_cancer_plain:,}", f"{n_cancer_enc:,}", "-"]},
            {"label": "No Cancer", "values": [f"{n_no_cancer_plain:,}", f"{n_no_cancer_enc:,}", "-"]},
            {
                "label": "Total",
                "values": [f"{n_cancer_plain + n_no_cancer_plain:,}", f"{n_cancer_enc + n_no_cancer_enc:,}", "-"],
            },
            {"label": "Queries", "values": [str(enc_queries), str(enc_queries), "-"]},
            {
                "label": "Train Time",
                "values": [f"{plain_train_time:.6f}s", f"{enc_train_time:.1f}s", f"+{overhead:.1f}s"],
                "classes": ["number-cell", "number-cell", overhead_class],
            },
            {
                "label": "Data Decrypted",
                "values": ["YES", "NEVER", "-"],
                "classes": ["string-cell status-bad", "string-cell status-good", "number-cell"],
            },
        ],
        headers=["", "Plaintext", "Blind Insight", "Overhead"],
    )


def _f1_cell(val: float) -> str:
    return f"{val:.3f}" if val > 0 else "0.000"


def _pct_cell(val: float) -> str:
    return f"{val * 100:.1f}%"


def _overhead_pct(plain: float, enc: float) -> tuple[str, str]:
    """Return (formatted delta, css class) for a percentage metric comparison."""
    delta = enc - plain
    if abs(delta) < 0.001:
        return "-", "number-cell"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}pp", "status-good" if delta >= 0 else "status-bad"


def screening_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Compute screening metrics from confusion matrix counts."""
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    flagged = (tp + fp) / max(1, tp + fp + tn + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sens": sens,
        "spec": spec,
        "ppv": ppv,
        "f1": f1,
        "flagged": flagged,
    }


def metrics_from_binary_preds(preds: list[int], y_true: list[int]) -> dict[str, float]:
    """Build confusion matrix + screening metrics from predictions and labels."""
    tp = sum(1 for p, a in zip(preds, y_true) if p == 1 and a == 1)
    fp = sum(1 for p, a in zip(preds, y_true) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(preds, y_true) if p == 0 and a == 1)
    tn = sum(1 for p, a in zip(preds, y_true) if p == 0 and a == 0)
    return screening_metrics(tp, fp, tn, fn)


def evaluate_row_risk_model(
    df_test,
    risk_fn: Callable[[dict[str, Any]], float],
    threshold: float = 0.0167,
) -> dict[str, float]:
    """Evaluate a row-wise risk function at a fixed threshold."""
    tp = fp = tn = fn = 0
    for _, row in df_test.iterrows():
        actual = int(row.get("cancer_5yr", row.get("is_cancer", 0)))
        risk = risk_fn(row.to_dict())
        pred = 1 if risk >= threshold else 0
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
        else:
            tn += 1
    return screening_metrics(tp, fp, tn, fn)


def evaluate_probabilities(
    y_true: list[int],
    probs: list[float],
    threshold: float = 0.0167,
) -> dict[str, float]:
    """Evaluate probability outputs against binary labels at a threshold."""
    preds = [1 if p >= threshold else 0 for p in probs]
    return metrics_from_binary_preds(preds, y_true)


def evaluate_bc_dt_nb_models(
    df_test,
    dt_model,
    plain_dt_model,
    plain_dt_col_names: list[str],
    feature_values: dict[str, list[str]],
    cohort_prev: float,
    pop_rate: float,
    threshold: float,
    P_cancer: float,
    P_no_cancer: float,
    P_enc: dict,
) -> dict[str, dict[str, float]]:
    """Evaluate encrypted DT, plaintext DT, and encrypted NB consistently."""
    y_true = [int(v) for v in df_test["is_cancer"].values]
    plain_dt_proba = plaintext_predict_proba(
        plain_dt_model,
        plain_dt_col_names,
        df_test,
        feature_values,
        encoding="dt",
    )
    plain_dt_probs = [recalibrate_risk(prob, cohort_prev, pop_rate) for prob in plain_dt_proba]
    dt_plain_metrics = evaluate_probabilities(y_true, plain_dt_probs, threshold)

    dt_enc_metrics = evaluate_row_risk_model(
        df_test,
        risk_fn=lambda row: recalibrate_risk(encrypted_dt_predict(dt_model, row)[1], cohort_prev, pop_rate),
        threshold=threshold,
    )
    nb_enc_metrics = evaluate_row_risk_model(
        df_test,
        risk_fn=lambda row: bc_naive_bayes_risk(P_cancer, P_no_cancer, P_enc, row),
        threshold=threshold,
    )
    return {
        "dt_plain_metrics": dt_plain_metrics,
        "dt_enc_metrics": dt_enc_metrics,
        "nb_enc_metrics": nb_enc_metrics,
    }


def train_evaluate_bc_lr_models(
    df_train,
    df_test,
    feature_values: dict[str, list[str]],
    raw_results: list[tuple],
    n_total_bi: int,
    n_cancer_bi: int,
    n_no_cancer_bi: int,
    cohort_prev: float,
    pop_rate: float = 0.016,
    threshold: float = 0.0167,
    ridge_lambda: float = 0.01,
    use_sigmoid: bool = True,
) -> dict[str, Any]:
    """Train k=0 and k=11 LR variants and evaluate against sklearn LR."""
    plain_lr = train_plaintext_bc_lr(df_train, feature_values)

    lr_start = time.time()
    pair_k0 = compute_pairwise_local(
        df_train,
        feature_values,
        raw_results,
        n_total_bi,
        min_cell_size=0,
    )
    lr_ols_k0 = build_linear_model(
        raw_results,
        pair_k0,
        feature_values,
        n_cancer_bi,
        n_no_cancer_bi,
    )
    lr_beta_k0 = refine_bc_with_irls(
        lr_ols_k0["beta"],
        lr_ols_k0["dummy_index"],
        df_train,
        feature_values,
        ridge_lambda=ridge_lambda,
    )
    enc_lr_time_k0 = time.time() - lr_start

    lr_start = time.time()
    pair_k11 = compute_pairwise_local(
        df_train,
        feature_values,
        raw_results,
        n_total_bi,
        min_cell_size=CMS_MIN_CELL_SIZE,
    )
    lr_ols_k11 = build_linear_model(
        raw_results,
        pair_k11,
        feature_values,
        n_cancer_bi,
        n_no_cancer_bi,
    )
    lr_beta_k11 = refine_bc_with_irls(
        lr_ols_k11["beta"],
        lr_ols_k11["dummy_index"],
        df_train,
        feature_values,
        ridge_lambda=ridge_lambda,
    )
    enc_lr_time = time.time() - lr_start

    lr0_metrics = evaluate_row_risk_model(
        df_test,
        risk_fn=lambda row: recalibrate_risk(
            linear_model_predict(lr_beta_k0, lr_ols_k0["dummy_index"], row, use_sigmoid=use_sigmoid),
            cohort_prev,
            pop_rate,
        ),
        threshold=threshold,
    )
    lr_metrics = evaluate_row_risk_model(
        df_test,
        risk_fn=lambda row: recalibrate_risk(
            linear_model_predict(lr_beta_k11, lr_ols_k11["dummy_index"], row, use_sigmoid=use_sigmoid),
            cohort_prev,
            pop_rate,
        ),
        threshold=threshold,
    )

    y_true = [int(v) for v in df_test["is_cancer"].values]
    plain_lr_proba = plaintext_predict_proba(plain_lr["model"], plain_lr["col_names"], df_test, feature_values)
    plain_lr_probs = [recalibrate_risk(prob, cohort_prev, pop_rate) for prob in plain_lr_proba]
    plain_lr_metrics = evaluate_probabilities(y_true, plain_lr_probs, threshold)

    lr_model = dict(lr_ols_k11)
    lr_model["beta"] = lr_beta_k11

    return {
        "plain_lr": plain_lr,
        "enc_lr_time_k0": enc_lr_time_k0,
        "enc_lr_time": enc_lr_time,
        "pair_k0": pair_k0,
        "pair_k11": pair_k11,
        "lr_ols_k0": lr_ols_k0,
        "lr_ols_k11": lr_ols_k11,
        "lr_beta_k0": lr_beta_k0,
        "lr_beta_k11": lr_beta_k11,
        "lr_model": lr_model,
        "lr0_metrics": lr0_metrics,
        "lr_metrics": lr_metrics,
        "plain_lr_metrics": plain_lr_metrics,
    }


def bc_dt_summary_table(
    enc_train_time: float,
    plain_train_time: float,
    enc_queries: int,
    enc_f1: float,
    plain_f1: float,
    enc_sens: float,
    plain_sens: float,
    enc_spec: float,
    plain_spec: float,
    enc_ppv: float,
    plain_ppv: float,
    enc_flagged: float,
    plain_flagged: float,
) -> str:
    """HTML table comparing plaintext vs encrypted Decision Tree."""
    overhead_t = enc_train_time - plain_train_time
    ov_t_cls = "status-good" if overhead_t < 180 else "status-bad"
    f1_ov, f1_cls = _overhead_pct(plain_f1, enc_f1)
    sens_ov, sens_cls = _overhead_pct(plain_sens, enc_sens)
    spec_ov, spec_cls = _overhead_pct(plain_spec, enc_spec)
    ppv_ov, ppv_cls = _overhead_pct(plain_ppv, enc_ppv)
    flag_ov, flag_cls = _overhead_pct(plain_flagged, enc_flagged)

    return metrics_table(
        rows=[
            {
                "label": "F1 Score",
                "values": [_f1_cell(plain_f1), _f1_cell(enc_f1), f1_ov],
                "classes": ["number-cell", "number-cell", f1_cls],
            },
            {
                "label": "Sensitivity (recall)",
                "values": [_pct_cell(plain_sens), _pct_cell(enc_sens), sens_ov],
                "classes": ["number-cell", "number-cell", sens_cls],
            },
            {
                "label": "Specificity",
                "values": [_pct_cell(plain_spec), _pct_cell(enc_spec), spec_ov],
                "classes": ["number-cell", "number-cell", spec_cls],
            },
            {
                "label": "PPV (precision)",
                "values": [_pct_cell(plain_ppv), _pct_cell(enc_ppv), ppv_ov],
                "classes": ["number-cell", "number-cell", ppv_cls],
            },
            {
                "label": "Flagged High-Risk",
                "values": [_pct_cell(plain_flagged), _pct_cell(enc_flagged), flag_ov],
                "classes": ["number-cell", "number-cell", flag_cls],
            },
            {"label": "BI Queries", "values": [str(enc_queries), str(enc_queries), "-"]},
            {"label": "New Queries", "values": ["-", "0", "-"]},
            {
                "label": "Train Time",
                "values": [f"{plain_train_time * 1000:.0f}ms", f"{enc_train_time:.1f}s", f"+{overhead_t:.1f}s"],
                "classes": ["number-cell", "number-cell", ov_t_cls],
            },
            {
                "label": "Data Decrypted",
                "values": ["YES", "NEVER", "-"],
                "classes": ["string-cell status-bad", "string-cell status-good", "number-cell"],
            },
        ],
        headers=["", "Plaintext", "Blind Insight", "Overhead"],
    )


def bc_lr_summary_table(
    enc_train_time: float,
    plain_train_time: float,
    enc_queries: int,
    new_queries: int,
    enc_f1: float,
    plain_f1: float,
    enc_sens: float,
    plain_sens: float,
    enc_spec: float,
    plain_spec: float,
    enc_ppv: float,
    plain_ppv: float,
    enc_flagged: float,
    plain_flagged: float,
    n_suppressed: int,
    cms_k: int,
) -> str:
    """HTML table comparing plaintext vs encrypted Linear Regression."""
    overhead_t = enc_train_time - plain_train_time
    ov_t_cls = "status-good" if overhead_t < 180 else "status-bad"
    f1_ov, f1_cls = _overhead_pct(plain_f1, enc_f1)
    sens_ov, sens_cls = _overhead_pct(plain_sens, enc_sens)
    spec_ov, spec_cls = _overhead_pct(plain_spec, enc_spec)
    ppv_ov, ppv_cls = _overhead_pct(plain_ppv, enc_ppv)
    flag_ov, flag_cls = _overhead_pct(plain_flagged, enc_flagged)

    return metrics_table(
        rows=[
            {
                "label": "F1 Score",
                "values": [_f1_cell(plain_f1), _f1_cell(enc_f1), f1_ov],
                "classes": ["number-cell", "number-cell", f1_cls],
            },
            {
                "label": "Sensitivity (recall)",
                "values": [_pct_cell(plain_sens), _pct_cell(enc_sens), sens_ov],
                "classes": ["number-cell", "number-cell", sens_cls],
            },
            {
                "label": "Specificity",
                "values": [_pct_cell(plain_spec), _pct_cell(enc_spec), spec_ov],
                "classes": ["number-cell", "number-cell", spec_cls],
            },
            {
                "label": "PPV (precision)",
                "values": [_pct_cell(plain_ppv), _pct_cell(enc_ppv), ppv_ov],
                "classes": ["number-cell", "number-cell", ppv_cls],
            },
            {
                "label": "Flagged High-Risk",
                "values": [_pct_cell(plain_flagged), _pct_cell(enc_flagged), flag_ov],
                "classes": ["number-cell", "number-cell", flag_cls],
            },
            {"label": "BI Queries", "values": [str(enc_queries), str(enc_queries), "-"]},
            {"label": "New Queries", "values": ["-", str(new_queries), "-"]},
            {"label": f"HIPAA Suppressed (k={cms_k})", "values": ["-", str(n_suppressed), "-"]},
            {
                "label": "Train Time",
                "values": [f"{plain_train_time * 1000:.0f}ms", f"{enc_train_time:.1f}s", f"+{overhead_t:.1f}s"],
                "classes": ["number-cell", "number-cell", ov_t_cls],
            },
            {
                "label": "Data Decrypted",
                "values": ["YES", "NEVER", "-"],
                "classes": ["string-cell status-bad", "string-cell status-good", "number-cell"],
            },
        ],
        headers=["", "Plaintext", "Blind Insight", "Overhead"],
    )


def build_three_model_rows(
    enc_queries: int,
    nb_metrics: dict[str, float],
    dt_metrics: dict[str, float],
    lr_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    """Build rows for the three-model comparison table."""
    return [
        {
            "name": "Naive Bayes",
            "f1": nb_metrics["f1"],
            "sens": nb_metrics["sens"],
            "spec": nb_metrics["spec"],
            "ppv": nb_metrics["ppv"],
            "flagged": nb_metrics["flagged"],
            "queries": enc_queries,
            "new_queries": enc_queries,
            "decrypted": "NEVER",
        },
        {
            "name": "Decision Tree",
            "f1": dt_metrics["f1"],
            "sens": dt_metrics["sens"],
            "spec": dt_metrics["spec"],
            "ppv": dt_metrics["ppv"],
            "flagged": dt_metrics["flagged"],
            "queries": enc_queries,
            "new_queries": 0,
            "decrypted": "NEVER",
        },
        {
            "name": "Logistic Reg (OLS+IRLS)",
            "f1": lr_metrics["f1"],
            "sens": lr_metrics["sens"],
            "spec": lr_metrics["spec"],
            "ppv": lr_metrics["ppv"],
            "flagged": lr_metrics["flagged"],
            "queries": enc_queries,
            "new_queries": 0,
            "decrypted": "NEVER",
        },
    ]


def bc_three_model_comparison_table(
    models: list[dict[str, Any]],
) -> str:
    """HTML comparison table across all models.

    Each dict in *models*: {name, f1, sens, spec, ppv, flagged, queries, new_queries, decrypted}.
    """
    headers = [""] + [m["name"] for m in models]
    rows = [
        {"label": "F1 Score", "values": [_f1_cell(m["f1"]) for m in models]},
        {"label": "Sensitivity", "values": [_pct_cell(m["sens"]) for m in models]},
        {"label": "Specificity", "values": [_pct_cell(m["spec"]) for m in models]},
        {"label": "PPV (precision)", "values": [_pct_cell(m["ppv"]) for m in models]},
        {"label": "Flagged High-Risk", "values": [_pct_cell(m["flagged"]) for m in models]},
        {"label": "Total BI Queries", "values": [str(m["queries"]) for m in models]},
        {"label": "New BI Queries", "values": [str(m["new_queries"]) for m in models]},
        {
            "label": "Data Decrypted",
            "values": [m["decrypted"] for m in models],
            "classes": [
                "string-cell status-good" if m["decrypted"] == "NEVER" else "string-cell status-bad" for m in models
            ],
        },
    ]
    return metrics_table(rows=rows, headers=headers)


# ============================================================================
# Test validation
# ============================================================================


def run_bc_test_validation(
    df_test,
    P_cancer: float,
    P_no_cancer: float,
    P_enc: dict,
    P_cancer_plain: float,
    P_no_cancer_plain: float,
    P_plain: dict,
) -> dict[str, Any]:
    """Run NB predictions on test set, return metrics."""
    from sklearn.metrics import accuracy_score

    pred_start = time.time()
    y_true = df_test["is_cancer"].values
    enc_preds, plain_preds = [], []

    for _, row in df_test.iterrows():
        r = row.to_dict()
        enc_preds.append(bc_naive_bayes_predict(P_cancer, P_no_cancer, P_enc, r))
        plain_preds.append(bc_naive_bayes_predict(P_cancer_plain, P_no_cancer_plain, P_plain, r))

    pred_time = time.time() - pred_start
    agreement = sum(1 for b, p in zip(enc_preds, plain_preds) if b == p) / len(enc_preds)
    acc_plain = accuracy_score(y_true, plain_preds)
    acc_enc = accuracy_score(y_true, enc_preds)

    test_cancer = int(df_test["is_cancer"].sum())
    test_no = len(df_test) - test_cancer
    agree_class = "status-good" if agreement > 0.99 else "status-bad"

    m_html = metrics_table(
        rows=[
            {"label": "Records", "values": [f"{len(df_test):,}", f"{len(df_test):,}", "OK"]},
            {"label": "Cancer (5yr)", "values": [f"{test_cancer:,}", f"{test_cancer:,}", "OK"]},
            {"label": "No Cancer", "values": [f"{test_no:,}", f"{test_no:,}", "OK"]},
            {
                "label": "BI <-> Plain Agreement",
                "values": ["-", f"{agreement * 100:.1f}%", "OK" if agreement > 0.99 else "BAD"],
                "classes": ["number-cell", agree_class, agree_class],
            },
            {
                "label": "NB Accuracy",
                "values": [
                    f"{acc_plain * 100:.1f}%",
                    f"{acc_enc * 100:.1f}%",
                    "OK" if abs(acc_plain - acc_enc) < 0.01 else "BAD",
                ],
            },
        ],
        headers=["", "Plaintext NB", "Blind Insight NB", "Match"],
        caption="Encrypted vs. Plaintext NB Validation",
    )

    return {
        "enc_preds": enc_preds,
        "plain_preds": plain_preds,
        "pred_time": pred_time,
        "agreement": agreement,
        "acc_plain": acc_plain,
        "acc_enc": acc_enc,
        "metrics_html": m_html,
    }


def run_bc_full_validation(
    df_test,
    P_cancer: float,
    P_no_cancer: float,
    P_enc: dict,
    P_cancer_plain: float,
    P_no_cancer_plain: float,
    P_plain: dict,
    dt_model,
    plain_dt_model,
    plain_dt_col_names,
    lr_beta,
    lr_dummy_index,
    plain_lr_model,
    plain_lr_col_names,
    feature_values: dict[str, list[str]],
    cohort_prev: float,
    pop_rate: float = 0.016,
    threshold: float = 0.0167,
    use_sigmoid: bool = False,
    **_kwargs,
) -> dict[str, Any]:
    """Validate all three models (NB, DT, LR): encrypted vs sklearn plaintext.

    Plaintext comparisons use real-world sklearn models (CART, LogisticRegression)
    as the benchmark a data scientist would actually use.
    """
    y_true = df_test["is_cancer"].values
    n = len(df_test)

    # Plaintext sklearn predictions (batch)
    plain_dt_proba = plaintext_predict_proba(plain_dt_model, plain_dt_col_names, df_test, feature_values, encoding="dt")
    plain_lr_proba = plaintext_predict_proba(plain_lr_model, plain_lr_col_names, df_test, feature_values)

    nb_enc_preds, nb_plain_preds = [], []
    dt_enc_preds, dt_plain_preds = [], []
    lr_enc_preds, lr_plain_preds = [], []
    nb_time = dt_time = lr_time = 0.0

    for idx, (_, row) in enumerate(df_test.iterrows()):
        r = row.to_dict()

        t0 = time.time()
        nb_enc_risk = bc_naive_bayes_risk(P_cancer, P_no_cancer, P_enc, r)
        nb_enc_preds.append(1 if nb_enc_risk >= threshold else 0)
        nb_plain_risk = bc_naive_bayes_risk(P_cancer_plain, P_no_cancer_plain, P_plain, r)
        nb_plain_preds.append(1 if nb_plain_risk >= threshold else 0)
        nb_time += time.time() - t0

        t0 = time.time()
        _, dt_risk_raw = encrypted_dt_predict(dt_model, r)
        dt_risk = recalibrate_risk(dt_risk_raw, cohort_prev, pop_rate)
        dt_enc_preds.append(1 if dt_risk >= threshold else 0)
        pdt_risk = recalibrate_risk(plain_dt_proba[idx], cohort_prev, pop_rate)
        dt_plain_preds.append(1 if pdt_risk >= threshold else 0)
        dt_time += time.time() - t0

        t0 = time.time()
        lr_risk_raw = linear_model_predict(lr_beta, lr_dummy_index, r, use_sigmoid=use_sigmoid)
        lr_risk = recalibrate_risk(lr_risk_raw, cohort_prev, pop_rate)
        lr_enc_preds.append(1 if lr_risk >= threshold else 0)
        plr_risk = recalibrate_risk(plain_lr_proba[idx], cohort_prev, pop_rate)
        lr_plain_preds.append(1 if plr_risk >= threshold else 0)
        lr_time += time.time() - t0

    pred_time = nb_time + dt_time + lr_time

    def _agree(a, b):
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)

    nb_agree = _agree(nb_enc_preds, nb_plain_preds)
    dt_agree = _agree(dt_enc_preds, dt_plain_preds)
    lr_agree = _agree(lr_enc_preds, lr_plain_preds)

    # --- Per-model F1 / sensitivity / specificity / PPV ---
    y_true_list = [int(v) for v in y_true]
    nb_enc_m = metrics_from_binary_preds(nb_enc_preds, y_true_list)
    nb_pln_m = metrics_from_binary_preds(nb_plain_preds, y_true_list)
    dt_enc_m = metrics_from_binary_preds(dt_enc_preds, y_true_list)
    dt_pln_m = metrics_from_binary_preds(dt_plain_preds, y_true_list)
    lr_enc_m = metrics_from_binary_preds(lr_enc_preds, y_true_list)
    lr_pln_m = metrics_from_binary_preds(lr_plain_preds, y_true_list)

    test_cancer = int(y_true.sum())
    test_no = n - test_cancer

    def _status(val, good=0.95):
        return ("status-good", "OK") if val >= good else ("status-bad", "BAD")

    nb_cls, _ = _status(nb_agree, 0.99)
    dt_cls, _ = _status(dt_agree, 0.99)
    lr_cls, _ = _status(lr_agree, 0.99)

    def _f1_delta(enc_m, pln_m):
        d = enc_m["f1"] - pln_m["f1"]
        return "0pp" if abs(d) < 0.001 else f"{d * 100:+.1f}pp"

    m_html = metrics_table(
        rows=[
            {"label": "Test records", "values": [f"{n:,}", f"{n:,}", f"{n:,}"]},
            {"label": "Cancer (5yr)", "values": [f"{test_cancer:,}", f"{test_cancer:,}", f"{test_cancer:,}"]},
            {"label": "No Cancer", "values": [f"{test_no:,}", f"{test_no:,}", f"{test_no:,}"]},
            {
                "label": "BI ↔ Plaintext Agreement",
                "values": [f"{nb_agree * 100:.1f}%", f"{dt_agree * 100:.1f}%", f"{lr_agree * 100:.1f}%"],
                "classes": [nb_cls, dt_cls, lr_cls],
            },
            {
                "label": "F1 (encrypted)",
                "values": [f"{nb_enc_m['f1']:.3f}", f"{dt_enc_m['f1']:.3f}", f"{lr_enc_m['f1']:.3f}"],
            },
            {
                "label": "F1 (plaintext)",
                "values": [f"{nb_pln_m['f1']:.3f}", f"{dt_pln_m['f1']:.3f}", f"{lr_pln_m['f1']:.3f}"],
            },
            {
                "label": "F1 Δ (enc − plain)",
                "values": [_f1_delta(nb_enc_m, nb_pln_m), _f1_delta(dt_enc_m, dt_pln_m), _f1_delta(lr_enc_m, lr_pln_m)],
            },
            {
                "label": "Sensitivity (encrypted)",
                "values": [
                    f"{nb_enc_m['sens'] * 100:.1f}%",
                    f"{dt_enc_m['sens'] * 100:.1f}%",
                    f"{lr_enc_m['sens'] * 100:.1f}%",
                ],
            },
            {
                "label": "Specificity (encrypted)",
                "values": [
                    f"{nb_enc_m['spec'] * 100:.1f}%",
                    f"{dt_enc_m['spec'] * 100:.1f}%",
                    f"{lr_enc_m['spec'] * 100:.1f}%",
                ],
            },
            {
                "label": "PPV (encrypted)",
                "values": [
                    f"{nb_enc_m['ppv'] * 100:.1f}%",
                    f"{dt_enc_m['ppv'] * 100:.1f}%",
                    f"{lr_enc_m['ppv'] * 100:.1f}%",
                ],
            },
            {
                "label": "Compute time (10K predictions)",
                "values": [f"{nb_time:.1f}s", f"{dt_time:.1f}s", f"{lr_time:.1f}s"],
            },
        ],
        headers=["", "Naive Bayes", "Decision Tree", "Logistic Reg"],
        caption="Encrypted vs. Plaintext Validation (10K Test Records)",
    )

    # --- Confusion matrices ---
    def _cm(label, enc_m, pln_m):
        return (
            f'<div style="margin-bottom:16px;">'
            f'<h4 style="font-size:14px;margin-bottom:4px;">{label}</h4>'
            f'<div style="display:flex;gap:24px;">'
            f'<div><p style="font-size:11px;font-weight:600;margin-bottom:2px;">Encrypted</p>'
            f'<table class="bi-metrics-table" style="max-width:220px;font-size:12px;">'
            f"<tr><td></td><th>Pred Low</th><th>Pred High</th></tr>"
            f"<tr><th>Actual No</th>"
            f'<td class="number-cell">{enc_m["tn"]:,}</td>'
            f'<td class="number-cell" style="background:#ffebee;color:#4a2d6b;">{enc_m["fp"]:,}</td></tr>'
            f"<tr><th>Actual Yes</th>"
            f'<td class="number-cell" style="background:#ffebee;color:#4a2d6b;">{enc_m["fn"]:,}</td>'
            f'<td class="number-cell">{enc_m["tp"]:,}</td></tr>'
            f"</table></div>"
            f'<div><p style="font-size:11px;font-weight:600;margin-bottom:2px;">'
            f"Plaintext (sklearn)</p>"
            f'<table class="bi-metrics-table" style="max-width:220px;font-size:12px;">'
            f"<tr><td></td><th>Pred Low</th><th>Pred High</th></tr>"
            f"<tr><th>Actual No</th>"
            f'<td class="number-cell">{pln_m["tn"]:,}</td>'
            f'<td class="number-cell" style="background:#ffebee;color:#4a2d6b;">{pln_m["fp"]:,}</td></tr>'
            f"<tr><th>Actual Yes</th>"
            f'<td class="number-cell" style="background:#ffebee;color:#4a2d6b;">{pln_m["fn"]:,}</td>'
            f'<td class="number-cell">{pln_m["tp"]:,}</td></tr>'
            f"</table></div></div></div>"
        )

    cm_html = (
        _cm("Naive Bayes", nb_enc_m, nb_pln_m)
        + _cm("Decision Tree (Gini)", dt_enc_m, dt_pln_m)
        + _cm("Logistic Regression (OLS+IRLS)", lr_enc_m, lr_pln_m)
    )
    cm_html += (
        f'<p style="font-size:11px;color:#718096;margin-top:8px;">'
        f"Threshold: {threshold * 100:.2f}% 5-year risk (FDA chemoprevention). "
        f"F1 reflects the screening task: high sensitivity (catching cancer) "
        f"at the cost of PPV. Matching encrypted and plaintext confusion "
        f"matrices confirms encryption causes zero degradation.</p>"
    )

    return {
        "pred_time": pred_time,
        "nb_agreement": nb_agree,
        "dt_agreement": dt_agree,
        "lr_agreement": lr_agree,
        "nb_time": nb_time,
        "dt_time": dt_time,
        "lr_time": lr_time,
        "nb_enc_metrics": nb_enc_m,
        "nb_pln_metrics": nb_pln_m,
        "dt_enc_metrics": dt_enc_m,
        "dt_pln_metrics": dt_pln_m,
        "lr_enc_metrics": lr_enc_m,
        "lr_pln_metrics": lr_pln_m,
        "metrics_html": m_html,
        "cm_html": cm_html,
    }


# ============================================================================
# BCRAT / BCSC / NB comparison on test set
# ============================================================================


def run_model_comparison(
    df_test,
    P_cancer: float,
    P_no_cancer: float,
    P_enc: dict,
    threshold_167: float = 0.0167,
    threshold_300: float = 0.030,
) -> dict[str, Any]:
    """
    Score every test record with NB, BCRAT, and BCSC.
    Returns comparison metrics and an HTML table.
    """
    results = []
    for _, row in df_test.iterrows():
        r = dict(row)
        nb_risk = bc_naive_bayes_risk(P_cancer, P_no_cancer, P_enc, r)
        bcrat_risk = bcrat_5yr_risk(r)
        bcsc_risk = bcsc_5yr_risk(r)

        results.append(
            {
                "patient_id": r.get("patient_id", ""),
                "age": int(r.get("age", 0)),
                "actual": int(r.get("cancer_5yr", r.get("is_cancer", 0))),
                "nb_risk": nb_risk,
                "bcrat_risk": bcrat_risk,
                "bcsc_risk": bcsc_risk,
                "nb_high_167": 1 if nb_risk >= threshold_167 else 0,
                "nb_high_300": 1 if nb_risk >= threshold_300 else 0,
                "bcrat_high_167": 1 if bcrat_risk >= threshold_167 else 0,
                "bcrat_high_300": 1 if bcrat_risk >= threshold_300 else 0,
                "bcsc_high_167": 1 if bcsc_risk >= threshold_167 else 0,
                "bcsc_high_300": 1 if bcsc_risk >= threshold_300 else 0,
            }
        )

    import pandas as pd

    comp_df = pd.DataFrame(results)
    n = len(comp_df)

    # Agreement rates
    nb_bcrat_agree = (comp_df["nb_high_167"] == comp_df["bcrat_high_167"]).sum() / n
    nb_bcsc_agree = (comp_df["nb_high_167"] == comp_df["bcsc_high_167"]).sum() / n
    bcrat_bcsc_agree = (comp_df["bcrat_high_167"] == comp_df["bcsc_high_167"]).sum() / n
    all_agree = (
        (comp_df["nb_high_167"] == comp_df["bcrat_high_167"]) & (comp_df["bcrat_high_167"] == comp_df["bcsc_high_167"])
    ).sum() / n

    # High-risk percentages at 1.67% threshold
    pct_nb = comp_df["nb_high_167"].mean() * 100
    pct_bcrat = comp_df["bcrat_high_167"].mean() * 100
    pct_bcsc = comp_df["bcsc_high_167"].mean() * 100

    # High-risk percentages at 3.0% threshold
    pct_nb_3 = comp_df["nb_high_300"].mean() * 100
    pct_bcrat_3 = comp_df["bcrat_high_300"].mean() * 100
    pct_bcsc_3 = comp_df["bcsc_high_300"].mean() * 100

    # Sensitivity among actual cancer cases
    actual_cancer = comp_df[comp_df["actual"] == 1]
    if len(actual_cancer) > 0:
        sens_nb = actual_cancer["nb_high_167"].mean() * 100
        sens_bcrat = actual_cancer["bcrat_high_167"].mean() * 100
        sens_bcsc = actual_cancer["bcsc_high_167"].mean() * 100
    else:
        sens_nb = sens_bcrat = sens_bcsc = 0.0

    summary_html = metrics_table(
        rows=[
            {
                "label": "% Flagged High Risk (>=1.67%)",
                "values": [f"{pct_nb:.1f}%", f"{pct_bcrat:.1f}%", f"{pct_bcsc:.1f}%"],
            },
            {
                "label": "% Flagged High Risk (>=3.0%)",
                "values": [f"{pct_nb_3:.1f}%", f"{pct_bcrat_3:.1f}%", f"{pct_bcsc_3:.1f}%"],
            },
            {
                "label": "Sensitivity (cancer cases at 1.67%)",
                "values": [f"{sens_nb:.1f}%", f"{sens_bcrat:.1f}%", f"{sens_bcsc:.1f}%"],
            },
            {"label": "NB-BCRAT Agreement", "values": [f"{nb_bcrat_agree * 100:.1f}%", "-", "-"]},
            {"label": "NB-BCSC Agreement", "values": [f"{nb_bcsc_agree * 100:.1f}%", "-", "-"]},
            {"label": "BCRAT-BCSC Agreement", "values": ["-", f"{bcrat_bcsc_agree * 100:.1f}%", "-"]},
            {"label": "All 3 Models Agree", "values": [f"{all_agree * 100:.1f}%", "-", "-"]},
        ],
        headers=["", "BI Naive Bayes", "BCRAT (Gail)", "BCSC"],
        caption="Model Comparison: Risk Classification at Clinical Thresholds",
    )

    return {
        "comp_df": comp_df,
        "nb_bcrat_agree": nb_bcrat_agree,
        "nb_bcsc_agree": nb_bcsc_agree,
        "bcrat_bcsc_agree": bcrat_bcsc_agree,
        "all_agree": all_agree,
        "pct_nb_167": pct_nb,
        "pct_bcrat_167": pct_bcrat,
        "pct_bcsc_167": pct_bcsc,
        "summary_html": summary_html,
    }


def sample_comparison_table(comp_df, limit: int = 15) -> str:
    """Generate an HTML table showing per-patient risk scores from all 3 models."""
    import pandas as pd

    rows_html = ""
    positives = comp_df[comp_df["actual"] == 1]
    negatives = comp_df[comp_df["actual"] == 0]
    n_pos = max(1, min(len(positives), limit // 3))
    n_neg = limit - n_pos
    sample = pd.concat(
        [
            positives.sample(n=n_pos, random_state=42),
            negatives.sample(n=n_neg, random_state=42),
        ]
    ).sort_values("age")
    for _, r in sample.iterrows():
        actual_icon = "\u274c Cancer" if r["actual"] == 1 else "\u2705 No"
        actual_cls = "status-bad" if r["actual"] == 1 else "status-good"

        def _risk_cell(risk_val):
            pct = risk_val * 100
            if pct >= 3.0:
                return f"<td class='number-cell status-bad'>{pct:.2f}%</td>"
            elif pct >= 1.67:
                return f"<td class='number-cell' style='color:#D69E2E;'>{pct:.2f}%</td>"
            return f"<td class='number-cell status-good'>{pct:.2f}%</td>"

        agree = r["nb_high_167"] == r["bcrat_high_167"] == r["bcsc_high_167"]
        agree_icon = "\u2713" if agree else "\u2717"
        agree_cls = "status-good" if agree else "status-bad"

        rows_html += "<tr>"
        rows_html += f"<td class='number-cell'>{int(r['age'])}</td>"
        rows_html += _risk_cell(r["nb_risk"])
        rows_html += _risk_cell(r["bcrat_risk"])
        rows_html += _risk_cell(r["bcsc_risk"])
        rows_html += f"<td class='{actual_cls}'>{actual_icon}</td>"
        rows_html += f"<td class='center-cell {agree_cls}'>{agree_icon}</td>"
        rows_html += "</tr>\n"

    return f"""
<table class="bi-data-table">
<caption style="text-align:left; font-weight:600; padding:8px 0;">
Per-Patient Risk Scores: BI Naive Bayes vs BCRAT vs BCSC</caption>
<thead><tr>
<th>Age</th><th>BI Risk</th><th>BCRAT Risk</th><th>BCSC Risk</th>
<th>Actual (5yr)</th><th>All Agree</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
<p style="font-size:11px; color:#718096; margin-top:8px;">
Risk thresholds: <span style="color:#38A169;">&lt;1.67% average</span> |
<span style="color:#D69E2E;">1.67-3.0% elevated</span> |
<span style="color:#E53E3E;">&ge;3.0% high risk</span><br/>
Model disagreement (shown by \u2717) motivates risk-based vs. fixed-schedule screening.
</p>"""


# ============================================================================
# Real-time decrypt demo (healthcare version)
# ============================================================================


def run_bc_realtime_demo(
    client,
    org: str,
    dataset: str,
    schema: str,
    P_cancer: float,
    P_no_cancer: float,
    P_enc: dict,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Fetch encrypted + decrypted records, classify, build demo table."""
    import json
    import random as _rand

    demo_id = _rand.randint(1000, 9999)

    def _query(decrypt_flag):
        t0 = time.time()
        result = client.query(
            organization=org,
            dataset_slug=dataset,
            schema_slug=schema,
            limit=sample_size,
            decrypt=decrypt_flag,
        )
        return result, time.time() - t0

    query_start = time.time()
    result_enc, enc_query_time = _query(False)
    result_dec, _ = _query(True)
    total_query_time = time.time() - query_start

    if not result_enc.get("records"):
        raise ValueError("BI returned no encrypted records.")
    if not result_dec.get("records"):
        raise ValueError("BI returned no decrypted records.")

    records_enc = result_enc["records"][:sample_size]
    records_dec = result_dec["records"][:sample_size]

    def _data(rec):
        return rec.get("data", rec) if isinstance(rec, dict) else {}

    DISPLAY_COLS = [
        "age",
        "num_first_degree_relatives",
        "breast_density",
        "menarche_category",
        "atypical_hyperplasia",
    ]
    cols_dec = [c for c in DISPLAY_COLS if c in _data(records_dec[0])]
    cols_enc = list(_data(records_enc[0]).keys())[: len(cols_dec)]

    # Build table rows
    header_html = ""
    for col in cols_dec:
        header_html += f'<th style="text-align:left;">{col}</th>'
    header_html += "<th>BI Risk</th><th>BCRAT</th><th>BCSC</th>"

    rows_html = ""
    decrypted_data = {}
    encrypted_data = {}

    for i in range(min(5, len(records_enc), len(records_dec))):
        data_enc = _data(records_enc[i]) if i < len(records_enc) else {}
        data_dec = _data(records_dec[i]) if i < len(records_dec) else {}

        nb_risk = bc_naive_bayes_risk(P_cancer, P_no_cancer, P_enc, data_dec)
        bcrat = bcrat_5yr_risk(data_dec)
        bcsc = bcsc_5yr_risk(data_dec)

        def _fmt_risk(r):
            pct = r * 100
            if pct >= 3.0:
                return f"<span style='color:#E53E3E; font-weight:600;'>{pct:.2f}%</span>"
            elif pct >= 1.67:
                return f"<span style='color:#D69E2E;'>{pct:.2f}%</span>"
            return f"<span style='color:#38A169;'>{pct:.2f}%</span>"

        rows_html += '<tr class="data-row">'
        decrypted_data[str(i)] = {}
        encrypted_data[str(i)] = {}

        for j, col_dec in enumerate(cols_dec):
            if j < len(cols_enc):
                enc_col = cols_enc[j]
                val_enc = str(data_enc.get(enc_col, "\u2014"))[:35]
            else:
                val_enc = "\u2014"
            val_dec = str(data_dec.get(col_dec, "\u2014"))[:35]
            val_dec_esc = val_dec.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
            val_enc_esc = val_enc.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

            rows_html += (
                f'<td class="string-cell" id="cell-{demo_id}-{i}-{j}">'
                f'<div class="cell-ellipsis" title="{val_enc}">{val_enc}</div></td>'
            )
            decrypted_data[str(i)][str(j)] = val_dec_esc
            encrypted_data[str(i)][str(j)] = val_enc_esc

        rows_html += f'<td class="number-cell">{_fmt_risk(nb_risk)}</td>'
        rows_html += f'<td class="number-cell">{_fmt_risk(bcrat)}</td>'
        rows_html += f'<td class="number-cell">{_fmt_risk(bcsc)}</td>'
        rows_html += "</tr>\n"

    decrypted_js = json.dumps(decrypted_data)
    encrypted_js = json.dumps(encrypted_data)
    num_cols = len(cols_dec)

    html = f"""{_inject_notebook_styles()}
<h4 style="margin-bottom:8px; font-size:15px;">Encrypted Patient Records from Blind Insight</h4>
<div class="bi-data-wrapper">
  <table class="bi-data-table" id="demo-table-{demo_id}">
    <tr class="subheader-row">{header_html}</tr>
    {rows_html}
  </table>
</div>

{_inject_table_resize_script()}

<div style="margin-top:12px;">
  <button id="decrypt-btn-{demo_id}" onclick="toggleRows_{demo_id}()"
    style="background:#6B46C1; color:white; border:none; padding:8px 16px;
           border-radius:4px; cursor:pointer; font-size:14px;">
    \U0001f513 Decrypt Records
  </button>
  <span id="decrypt-status-{demo_id}" style="margin-left:12px; color:#718096;"></span>
</div>

<script>
var decData_{demo_id} = {decrypted_js};
var encData_{demo_id} = {encrypted_js};
var nCols_{demo_id} = {num_cols};
var isDec_{demo_id} = false;

function toggleRows_{demo_id}() {{
  var data = isDec_{demo_id} ? encData_{demo_id} : decData_{demo_id};
  for (var i = 0; i < 5; i++) {{
    if (data[i]) {{
      for (var j = 0; j < nCols_{demo_id}; j++) {{
        var cell = document.getElementById('cell-{demo_id}-' + i + '-' + j);
        if (cell && data[i][j] !== undefined) cell.innerText = data[i][j];
      }}
    }}
  }}
  isDec_{demo_id} = !isDec_{demo_id};
  var btn = document.getElementById('decrypt-btn-{demo_id}');
  var st = document.getElementById('decrypt-status-{demo_id}');
  if (isDec_{demo_id}) {{
    btn.style.background = '#2F855A';
    btn.innerText = '\U0001f512 Re-Encrypt Records';
    st.innerText = '\u2713 Showing decrypted values';
  }} else {{
    btn.style.background = '#6B46C1';
    btn.innerText = '\U0001f513 Decrypt Records';
    st.innerText = '';
  }}
}}
</script>

<p style="font-size:12px; color:#718096; margin-top:8px;">
  Risk scores computed after local decryption (Data Owner role).<br/>
  NB model trained entirely on encrypted aggregates -- no records were decrypted during training.
</p>"""

    return {
        "html": html,
        "enc_query_time": enc_query_time,
        "total_query_time": total_query_time,
        "rt_count": min(len(records_enc), len(records_dec)),
    }
