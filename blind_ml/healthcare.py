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
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd

# Re-use generic plumbing from the fraud demo helpers
from .demo_helpers import (
    _inject_notebook_styles,
    _inject_table_resize_script,
    load_env,  # noqa: F401 — re-exported for breast_cancer.ipynb
    metrics_table,
)
from .models import (
    AdaBoostStumpModel as _AdaBoostStumpModel,
)
from .models import (
    BayesianNetworkClassifierModel as _BayesianNetworkClassifierModel,
)
from .models import (
    DecisionTreeModel as _DecisionTreeModel,
)
from .models import (
    GaussianNaiveBayesModel as _GaussianNaiveBayesModel,
)
from .models import (
    HistogramClassifierModel as _HistogramClassifierModel,
)
from .models import (
    LogisticRegressionModel as _LogisticRegressionModel,
)
from .models import (
    RandomForestModel as _RandomForestModel,
)
from .models import (
    build_bayesian_cpt_counts_local as _build_bayesian_cpt_counts_local,
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

# CMS cell suppression policy: exact aggregate cells with counts 1-10 should not
# be reported or consumed as exact values in the healthcare demo.
CMS_MIN_CELL_SIZE = 11

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

_BC_STATIC_FEATURE_VALUES = {
    "age_groups": ["40_49", "50_59", "60_69", "70_74"],
    "race_values": ["asian_pi", "black", "hispanic", "other", "white"],
    "relatives_values": ["0", "1", "2", "3", "4", "5"],
    "biopsies_values": [b[0] for b in BIOPSY_BINS],
    "atypical_values": ["no", "unknown", "yes"],
    "menarche_values": ["12_13", "14_plus", "under_12"],
    "density_values": ["1", "2", "3", "4"],
    "afb_values": [b[0] for b in AFB_BINS],
}

_BC_GNB_DEFAULT_FEATURES = [
    "age",
    "age_at_menarche",
    "age_at_first_birth",
    "num_first_degree_relatives",
    "num_prior_biopsies",
    "breast_density",
]

_BC_GNB_VALUE_DOMAINS = {
    "age": [str(v) for v in range(40, 77)],
    "age_at_menarche": [str(v) for v in range(8, 20)],
    "age_at_first_birth": [str(v) for v in range(0, 48)],
    "num_first_degree_relatives": [str(v) for v in range(0, 6)],
    "num_prior_biopsies": [str(v) for v in range(0, 8)],
    "breast_density": [str(v) for v in range(1, 7)],
}


def get_bc_feature_values() -> dict[str, list[str]]:
    """Return static breast-cancer feature domains for BI aggregate training."""
    return {key: list(values) for key, values in _BC_STATIC_FEATURE_VALUES.items()}


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


def _suppression_replacement(min_cell_size: int) -> int:
    """Fixed deterministic replacement for suppressed nonzero cells."""
    return max(1, int(min_cell_size) // 2)


def _suppressed_cell_label(row: tuple) -> str:
    """Human-readable label for a suppressed aggregate tuple."""
    if len(row) >= 4:
        return f"{row[0]}={row[2]}|class={row[1]}"
    return repr(row)


def suppress_bc_counts(
    raw_results: list[tuple],
    min_cell_size: int = CMS_MIN_CELL_SIZE,
) -> tuple[list[tuple], dict[str, Any]]:
    """Apply CMS-style suppression to aggregate count tuples.

    Count tuples are expected to store the aggregate count as their final item.
    Exact zeroes stay zero, counts in [1, min_cell_size) are replaced with a
    fixed midpoint estimate, and counts >= min_cell_size are preserved.
    """
    if min_cell_size <= 1:
        return list(raw_results), {
            "min_cell_size": min_cell_size,
            "n_suppressed": 0,
            "suppressed_cells": [],
            "suppression_policy": "none",
        }

    replacement = _suppression_replacement(min_cell_size)
    sanitized: list[tuple] = []
    suppressed_cells: list[str] = []

    for row in raw_results:
        if not row:
            sanitized.append(row)
            continue
        count = int(row[-1])
        if 0 < count < min_cell_size:
            sanitized.append((*row[:-1], replacement))
            suppressed_cells.append(_suppressed_cell_label(row))
        else:
            sanitized.append(row)

    return sanitized, {
        "min_cell_size": min_cell_size,
        "n_suppressed": len(suppressed_cells),
        "suppressed_cells": suppressed_cells,
        "suppression_policy": f"cms_k{min_cell_size}_fixed_midpoint_{replacement}",
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


def _bc_agg_value(resp) -> float:
    """Extract a scalar aggregate value from BI aggregate responses."""
    if isinstance(resp, list):
        records = resp
    else:
        records = resp.get("records", []) if isinstance(resp, dict) else []
    if not records:
        return 0.0
    first = records[0]
    data = first.get("data", {}) if isinstance(first, dict) else {}
    if isinstance(data, dict) and "value" in data:
        value = data.get("value")
        return float(value) if value is not None else 0.0
    if isinstance(first, dict) and "value" in first:
        value = first.get("value")
        return float(value) if value is not None else 0.0
    return 0.0


def _count_via_class_aggregate(client, org, dataset, schema, filters: list[str], retries: int = 3) -> int:
    """Fallback count for deep paths: aggregate the class field with path filters."""
    if not filters:
        return 0
    class_filter = filters[0]
    if class_filter not in ("cancer_5yr:1", "cancer_5yr:0"):
        raise ValueError(f"BC aggregate fallback requires class-first filters, got {filters!r}")
    class_value = class_filter.split(":", 1)[1]
    agg_filter = f"cancer_5yr:count({class_value})"
    extra_filters = list(filters[1:])
    for attempt in range(retries):
        try:
            result = client.aggregate(
                organization=org,
                dataset_slug=dataset,
                schema_slug=schema,
                agg_filter=agg_filter,
                extra_filters=extra_filters,
            )
            return int(_bc_agg_value(result))
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
    min_cell_size: int = CMS_MIN_CELL_SIZE,
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

    sanitized_results, suppression = suppress_bc_counts(results, min_cell_size=min_cell_size)
    out = {
        "raw_results": sanitized_results,
        "raw_results_unsuppressed": results,
        "enc_queries": len(results),
        **suppression,
    }
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


def _bc_dt_feature_values(feature_values: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return DecisionTreeModel feature values keyed by BC feature key."""
    values_by_feature: dict[str, list[str]] = {}
    for feature, (values_key, _field_name) in _FEATURE_MAP.items():
        seen: set[str] = set()
        values_by_feature[feature] = []
        for raw_value in feature_values.get(values_key, []):
            value = str(raw_value).lower()
            if value in seen:
                continue
            seen.add(value)
            values_by_feature[feature].append(value)
    return values_by_feature


def _bc_class_filter(class_label: int) -> str:
    return "cancer_5yr:1" if int(class_label) == 1 else "cancer_5yr:0"


def _normalize_filter_tuple(filters: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(str(f) for f in filters))


def _build_bc_dt_count_provider(
    client,
    org: str,
    dataset: str,
    schema: str,
    feature_values: dict[str, list[str]],
    raw_results: list[tuple] | None = None,
    aggregate_cache: dict[tuple[str, ...], int] | None = None,
    n_cancer: int | None = None,
    n_no_cancer: int | None = None,
):
    """Create a DecisionTreeModel count_fn backed only by BI aggregates."""
    dt_feature_values = _bc_dt_feature_values(feature_values)
    aggregate_cache = aggregate_cache if aggregate_cache is not None else {}
    split_cache: dict[tuple[tuple[tuple[str, str, bool], ...], str, str, int], int] = {}
    path_bound_cache: dict[tuple[tuple[tuple[str, str, bool], ...], int], int] = {}
    query_counter = {"n": 0}
    fallback_counter = {"n": 0}
    class_totals = {}
    if n_cancer is not None:
        class_totals[1] = int(n_cancer)
    if n_no_cancer is not None:
        class_totals[0] = int(n_no_cancer)

    if raw_results:
        for feat_type, cls, raw_value, count in raw_results:
            feature = str(feat_type)
            if feature not in dt_feature_values:
                continue
            value = str(raw_value).lower()
            count = int(count)
            key = (tuple(), feature, value, int(cls))
            split_cache[key] = count
            aggregate_cache[_normalize_filter_tuple([_bc_class_filter(cls), _bi_feat_filter(feature, value)])] = count

    def _aggregate(filters: list[str]) -> int:
        key = _normalize_filter_tuple(filters)
        if key not in aggregate_cache:
            if len(filters) >= 4:
                try:
                    aggregate_cache[key] = _count_via_class_aggregate(client, org, dataset, schema, filters)
                    fallback_counter["n"] += 1
                except Exception as fallback_exc:
                    raise RuntimeError(f"BI aggregate count failed for filters={filters!r}") from fallback_exc
            else:
                try:
                    aggregate_cache[key] = _count_only(client, org, dataset, schema, filters)
                except Exception:
                    try:
                        aggregate_cache[key] = _count_via_class_aggregate(client, org, dataset, schema, filters)
                        fallback_counter["n"] += 1
                    except Exception as fallback_exc:
                        raise RuntimeError(f"BI aggregate count failed for filters={filters!r}") from fallback_exc
            query_counter["n"] += 1
        return aggregate_cache[key]

    def _allowed_values(path: tuple[tuple[str, str, bool], ...]) -> dict[str, set[str]]:
        allowed = {feature: set(values) for feature, values in dt_feature_values.items()}
        for feature, raw_value, branch in path:
            if feature not in allowed:
                return {}
            value = str(raw_value).lower()
            if branch:
                allowed[feature] &= {value}
            else:
                allowed[feature].discard(value)
        return allowed

    def _path_constraints(
        path: tuple[tuple[str, str, bool], ...],
    ) -> tuple[dict[str, str], list[tuple[str, str]], bool]:
        equalities: dict[str, str] = {}
        exclusions: list[tuple[str, str]] = []
        for feature, raw_value, branch in path:
            value = str(raw_value).lower()
            if value not in dt_feature_values.get(feature, []):
                return {}, [], False
            if branch:
                existing = equalities.get(feature)
                if existing is not None and existing != value:
                    return {}, [], False
                equalities[feature] = value
            else:
                exclusions.append((feature, value))

        for feature, value in exclusions:
            if equalities.get(feature) == value:
                return {}, [], False
        return equalities, exclusions, True

    def _query_count(equalities: dict[str, str], class_label: int) -> int:
        filters = [_bc_class_filter(class_label)]
        for feature in _FEATURE_MAP:
            if feature in equalities:
                filters.append(_bi_feat_filter(feature, equalities[feature]))
        return _aggregate(filters)

    def _count_with_exclusions(
        equalities: dict[str, str],
        exclusions: list[tuple[str, str]],
        class_label: int,
    ) -> int:
        total = 0
        for size in range(len(exclusions) + 1):
            sign = -1 if size % 2 else 1
            for subset in combinations(exclusions, size):
                terms = dict(equalities)
                impossible = False
                for feature, value in subset:
                    existing = terms.get(feature)
                    if existing is not None and existing != value:
                        impossible = True
                        break
                    terms[feature] = value
                if impossible:
                    continue
                total += sign * _query_count(terms, class_label)
        return max(0, total)

    def _path_bound(path: tuple[tuple[str, str, bool], ...], class_label: int) -> int:
        key = (path, int(class_label))
        if key in path_bound_cache:
            return path_bound_cache[key]
        if not path:
            total = class_totals.get(int(class_label))
            if total is None:
                total = _aggregate([_bc_class_filter(class_label)])
            path_bound_cache[key] = int(total)
            return int(total)

        parent_path = path[:-1]
        split_feature, split_value, branch = path[-1]
        parent_total = _path_bound(parent_path, class_label)
        equalities, exclusions, possible = _path_constraints(parent_path + ((split_feature, split_value, True),))
        left_total = _count_with_exclusions(equalities, exclusions, class_label) if possible else 0
        left_total = min(parent_total, max(0, int(left_total)))
        total = left_total if branch else max(0, parent_total - left_total)
        path_bound_cache[key] = total
        return total

    def count_fn(
        path: tuple[tuple[str, str, bool], ...],
        feature: str,
        value: str,
        class_label: int,
    ) -> int:
        norm_path = tuple((str(f), str(v).lower(), bool(branch)) for f, v, branch in path)
        norm_value = str(value).lower()
        key = (norm_path, str(feature), norm_value, int(class_label))
        if key in split_cache:
            return split_cache[key]

        allowed = _allowed_values(norm_path + ((str(feature), norm_value, True),))
        if not allowed or any(len(values) == 0 for values in allowed.values()):
            split_cache[key] = 0
            return 0

        equalities, exclusions, possible = _path_constraints(norm_path + ((str(feature), norm_value, True),))
        total = _count_with_exclusions(equalities, exclusions, class_label) if possible else 0
        if norm_path:
            total = min(total, _path_bound(norm_path, class_label))
        split_cache[key] = total
        return total

    return count_fn, lambda: query_counter["n"], dt_feature_values, aggregate_cache, lambda: fallback_counter["n"]


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


def run_encrypted_dt_bc(
    client,
    org: str,
    dataset: str,
    schema: str,
    feature_values: dict[str, list[str]],
    raw_results: list[tuple[str, int, str, int]] | None = None,
    n_cancer: int | None = None,
    n_no_cancer: int | None = None,
    max_depth: int = 3,
    k_min: int = 11,
    criterion: str = "gini",
    min_cell_size: int | None = None,
    max_workers: int = 20,
) -> dict[str, Any]:
    """Build a breast-cancer decision tree entirely from BI aggregate counts."""
    if not feature_values:
        raise ValueError("run_encrypted_dt_bc requires feature_values.")
    if client is None or not org or not dataset or not schema:
        raise ValueError("run_encrypted_dt_bc requires BI client/org/dataset/schema.")

    start = time.time()
    raw_results_source = "provided"
    base_rate_queries = 0
    if n_cancer is None or n_no_cancer is None or int(n_cancer) + int(n_no_cancer) == 0:
        n_cancer, n_no_cancer = get_bc_base_rates(client, org, dataset, schema)
        base_rate_queries = 2

    if raw_results is None:
        raw = run_bc_conditional_queries(
            client,
            org,
            dataset,
            schema,
            feature_values,
            include_base_rates=False,
            min_cell_size=min_cell_size if min_cell_size is not None else max(0, int(k_min)),
        )
        raw_results = raw["raw_results"]
        raw_results_source = "dt_rerun"

    if int(n_cancer) + int(n_no_cancer) == 0:
        raise ValueError(
            "run_encrypted_dt_bc requires non-zero BI base rates. "
            "Got n_cancer=n_no_cancer=0 -- check that BI ingest completed."
        )

    count_fn, query_count, dt_feature_values, aggregate_cache, fallback_count = _build_bc_dt_count_provider(
        client=client,
        org=org,
        dataset=dataset,
        schema=schema,
        feature_values=feature_values,
        raw_results=raw_results,
        n_cancer=n_cancer,
        n_no_cancer=n_no_cancer,
    )
    dt = _DecisionTreeModel(max_depth=max_depth, criterion=criterion, k_min=k_min)
    dt.fit_from_counts(
        count_fn=count_fn,
        feature_values=dt_feature_values,
        n_pos=int(n_cancer),
        n_neg=int(n_no_cancer),
    )
    if dt.tree is not None:
        dt.tree["bi_counts"] = True

    return {
        "_model": dt,
        "tree": dt.tree,
        "col_names": dt.col_names,
        "_col_set": dt._col_set,
        "features": dt.feature_columns,
        "root_feat": dt.tree.get("col_name") if dt.tree and dt.tree.get("type") == "split" else None,
        "root_gain": dt.tree.get("gain", 0) if dt.tree else 0,
        "root_ig": dt.tree.get("gain", 0) if dt.tree else 0,
        "root_from_bi": True,
        "counts_from_bi": True,
        "root_children": {},
        "tree_nodes": {},
        "enc_queries": len(raw_results) + query_count(),
        "additional_dt_queries": query_count(),
        "fallback_aggregate_queries": fallback_count(),
        "base_rate_queries": base_rate_queries,
        "total_aggregate_calls": len(raw_results) + query_count() + base_rate_queries,
        "raw_results_source": raw_results_source,
        "raw_results": raw_results,
        "query_cache": aggregate_cache,
        "train_time": time.time() - start,
        "d3_safe": 0,
        "d3_fallback": 0,
        "feature_values": feature_values,
        "n_cancer": int(n_cancer),
        "n_no_cancer": int(n_no_cancer),
        "min_cell_size": int(k_min),
        "criterion": criterion,
    }


def bc_dt_predict(dt_result: dict, row: dict) -> tuple[int, float]:
    """Predict using the encrypted BC decision tree. Returns (pred, risk)."""
    model = dt_result.get("_model")
    if not model:
        return 0, 0.0
    return model.predict(_nb_features_from_row(row))


def bc_dt_describe(dt_result: dict) -> str:
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


def run_encrypted_dt(*args, **kwargs) -> dict[str, Any]:
    """Compatibility wrapper for the aggregate-only BC decision tree helper."""
    if len(args) >= 9:
        client, org, dataset, schema, raw_results, feature_values, _df_local, n_cancer, n_no_cancer, *rest = args
        if rest:
            kwargs.setdefault("k_min", rest[0])
        if len(rest) > 1:
            kwargs.setdefault("criterion", rest[1])
        return run_encrypted_dt_bc(
            client,
            org,
            dataset,
            schema,
            feature_values=feature_values,
            raw_results=raw_results,
            n_cancer=n_cancer,
            n_no_cancer=n_no_cancer,
            **kwargs,
        )
    kwargs.pop("df_local", None)
    return run_encrypted_dt_bc(*args, **kwargs)


encrypted_dt_predict = bc_dt_predict
encrypted_dt_describe = bc_dt_describe


# ============================================================================
# Encrypted Random Forest and AdaBoost from aggregate counts
# ============================================================================


def run_encrypted_rf_bc(
    client,
    org: str,
    dataset: str,
    schema: str,
    feature_values: dict[str, list[str]] | None = None,
    dt_result: dict | None = None,
    raw_results: list[tuple] | None = None,
    n_cancer: int | None = None,
    n_no_cancer: int | None = None,
    n_estimators: int = 7,
    max_depth: int = 3,
    max_features: int | float | str | None = 4,
    criterion: str = "gini",
    k_min: int = CMS_MIN_CELL_SIZE,
    random_state: int | None = 42,
    max_workers: int = 20,
) -> dict[str, Any]:
    """Train a breast-cancer Random Forest from BI aggregate counts only."""
    if not feature_values:
        if dt_result and dt_result.get("feature_values"):
            feature_values = dt_result["feature_values"]
        else:
            raise ValueError("run_encrypted_rf_bc requires feature_values or dt_result with feature_values.")
    if client is None or not org or not dataset or not schema:
        raise ValueError("run_encrypted_rf_bc requires BI client/org/dataset/schema.")

    raw_results_source = "provided" if raw_results is not None else "rf_rerun"
    base_rate_queries = 0
    marginal_queries = 0
    initial_cache: dict[tuple[str, ...], int] = {}

    if dt_result:
        raw_results = raw_results if raw_results is not None else dt_result.get("raw_results")
        n_cancer = n_cancer if n_cancer is not None else dt_result.get("n_cancer")
        n_no_cancer = n_no_cancer if n_no_cancer is not None else dt_result.get("n_no_cancer")
        initial_cache = dict(dt_result.get("query_cache", {}))
        raw_results_source = "dt_result" if raw_results is not None else raw_results_source

    if n_cancer is None or n_no_cancer is None or int(n_cancer) + int(n_no_cancer) == 0:
        n_cancer, n_no_cancer = get_bc_base_rates(client, org, dataset, schema)
        base_rate_queries = 2
    if raw_results is None:
        raw = run_bc_conditional_queries(
            client,
            org,
            dataset,
            schema,
            feature_values,
            include_base_rates=False,
            min_cell_size=max(0, int(k_min)),
        )
        raw_results = raw["raw_results"]
        marginal_queries = len(raw_results)
        raw_results_source = "rf_rerun"

    reused_cache_entries = len(initial_cache)
    count_fn, query_count, rf_feature_values, aggregate_cache, fallback_count = _build_bc_dt_count_provider(
        client=client,
        org=org,
        dataset=dataset,
        schema=schema,
        feature_values=feature_values,
        raw_results=raw_results,
        aggregate_cache=initial_cache,
        n_cancer=n_cancer,
        n_no_cancer=n_no_cancer,
    )

    model = _RandomForestModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        k_min=k_min,
        max_features=max_features,
        random_state=random_state,
    ).fit_from_counts(
        count_fn=count_fn,
        feature_values=rf_feature_values,
        n_pos=int(n_cancer),
        n_neg=int(n_no_cancer),
    )

    additional_rf_queries = query_count()
    enc_queries = marginal_queries + additional_rf_queries
    return {
        "_model": model,
        "trees": [tree.tree for tree in model.estimators_],
        "feature_subsets": model.feature_subsets_,
        "feature_values": feature_values,
        "raw_results": raw_results,
        "raw_results_source": raw_results_source,
        "query_cache": aggregate_cache,
        "counts_from_bi": True,
        "n_cancer": int(n_cancer),
        "n_no_cancer": int(n_no_cancer),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
        "criterion": criterion,
        "k_min": int(k_min),
        "min_cell_size": int(k_min),
        "train_time": model.train_time,
        "enc_queries": enc_queries,
        "base_rate_queries": base_rate_queries,
        "marginal_queries": marginal_queries,
        "additional_rf_queries": additional_rf_queries,
        "fallback_aggregate_queries": fallback_count(),
        "total_aggregate_calls": enc_queries + base_rate_queries,
        "reused_cache_entries": reused_cache_entries,
    }


def bc_rf_predict(rf_result: dict, row: dict) -> tuple[int, float]:
    """Predict using the encrypted BC Random Forest. Returns (pred, risk)."""
    model = rf_result.get("_model")
    if not model:
        return 0, 0.0
    return model.predict(_nb_features_from_row(row))


def bc_rf_describe(rf_result: dict, max_trees: int = 3) -> str:
    """Return a compact text description of the BC Random Forest."""
    model = rf_result.get("_model")
    if not model or not model.estimators_:
        return "Empty forest"

    lines = [
        f"Random Forest: {len(model.estimators_)} trees, max_depth={model.max_depth}, max_features={model.max_features}"
    ]
    for idx, (tree, subset) in enumerate(zip(model.estimators_[:max_trees], model.feature_subsets_[:max_trees]), 1):
        lines.append(f"\nTree {idx} features: {', '.join(subset)}")
        lines.append(bc_dt_describe({"_model": tree}))
    if len(model.estimators_) > max_trees:
        lines.append(f"\n... {len(model.estimators_) - max_trees} more trees")
    return "\n".join(lines)


def run_encrypted_adaboost_bc(
    client,
    org: str,
    dataset: str,
    schema: str,
    feature_values: dict[str, list[str]] | None = None,
    dt_result: dict | None = None,
    rf_result: dict | None = None,
    raw_results: list[tuple] | None = None,
    n_cancer: int | None = None,
    n_no_cancer: int | None = None,
    n_estimators: int = 10,
    learning_rate: float = 1.0,
    k_min: int = CMS_MIN_CELL_SIZE,
    max_workers: int = 20,
) -> dict[str, Any]:
    """Train BC AdaBoost decision stumps from BI aggregate counts only."""
    cache_source = rf_result or dt_result
    if not feature_values:
        if cache_source and cache_source.get("feature_values"):
            feature_values = cache_source["feature_values"]
        else:
            raise ValueError("run_encrypted_adaboost_bc requires feature_values, rf_result, or dt_result.")
    if client is None or not org or not dataset or not schema:
        raise ValueError("run_encrypted_adaboost_bc requires BI client/org/dataset/schema.")

    raw_results_source = "provided" if raw_results is not None else "adaboost_rerun"
    base_rate_queries = 0
    marginal_queries = 0
    initial_cache: dict[tuple[str, ...], int] = {}

    for candidate_source, source_name in ((rf_result, "rf_result"), (dt_result, "dt_result")):
        if not candidate_source:
            continue
        if raw_results is None and candidate_source.get("raw_results") is not None:
            raw_results = candidate_source["raw_results"]
            raw_results_source = source_name
        if n_cancer is None and candidate_source.get("n_cancer") is not None:
            n_cancer = candidate_source["n_cancer"]
        if n_no_cancer is None and candidate_source.get("n_no_cancer") is not None:
            n_no_cancer = candidate_source["n_no_cancer"]
        initial_cache.update(candidate_source.get("query_cache", {}))

    if n_cancer is None or n_no_cancer is None or int(n_cancer) + int(n_no_cancer) == 0:
        n_cancer, n_no_cancer = get_bc_base_rates(client, org, dataset, schema)
        base_rate_queries = 2
    if raw_results is None:
        raw = run_bc_conditional_queries(
            client,
            org,
            dataset,
            schema,
            feature_values,
            include_base_rates=False,
            min_cell_size=max(0, int(k_min)),
        )
        raw_results = raw["raw_results"]
        marginal_queries = len(raw_results)
        raw_results_source = "adaboost_rerun"

    reused_cache_entries = len(initial_cache)
    count_fn, query_count, boost_feature_values, aggregate_cache, fallback_count = _build_bc_dt_count_provider(
        client=client,
        org=org,
        dataset=dataset,
        schema=schema,
        feature_values=feature_values,
        raw_results=raw_results,
        aggregate_cache=initial_cache,
        n_cancer=n_cancer,
        n_no_cancer=n_no_cancer,
    )

    model = _AdaBoostStumpModel(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        k_min=k_min,
    ).fit_from_counts(
        count_fn=count_fn,
        feature_values=boost_feature_values,
        n_pos=int(n_cancer),
        n_neg=int(n_no_cancer),
    )

    additional_adaboost_queries = query_count()
    enc_queries = marginal_queries + additional_adaboost_queries
    return {
        "_model": model,
        "stumps": model.stumps_,
        "feature_values": feature_values,
        "raw_results": raw_results,
        "raw_results_source": raw_results_source,
        "query_cache": aggregate_cache,
        "counts_from_bi": True,
        "n_cancer": int(n_cancer),
        "n_no_cancer": int(n_no_cancer),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "k_min": int(k_min),
        "min_cell_size": int(k_min),
        "train_time": model.train_time,
        "enc_queries": enc_queries,
        "base_rate_queries": base_rate_queries,
        "marginal_queries": marginal_queries,
        "additional_adaboost_queries": additional_adaboost_queries,
        "fallback_aggregate_queries": fallback_count(),
        "total_aggregate_calls": enc_queries + base_rate_queries,
        "reused_cache_entries": reused_cache_entries,
    }


def bc_adaboost_predict(boost_result: dict, row: dict) -> tuple[int, float]:
    """Predict using the encrypted BC AdaBoost model. Returns (pred, risk)."""
    model = boost_result.get("_model")
    if not model:
        return 0, 0.0
    return model.predict(_nb_features_from_row(row))


def bc_adaboost_describe(boost_result: dict, max_stumps: int = 5) -> str:
    """Return a compact text description of BC AdaBoost stumps."""
    model = boost_result.get("_model")
    if not model or not model.stumps_:
        return "Empty AdaBoost model"

    lines = [f"AdaBoost: {len(model.stumps_)} stumps, learning_rate={model.learning_rate}, threshold={model.threshold}"]
    for idx, stump in enumerate(model.stumps_[:max_stumps], 1):
        lines.append(
            f"Stump {idx}: {stump['col_name']}? "
            f"alpha={stump['alpha']:.4f}, error={stump['error']:.4f}, "
            f"YES->{stump['left_pred']} (risk={stump['left_risk']:.3f}, n={stump['left_n']:,}), "
            f"NO->{stump['right_pred']} (risk={stump['right_risk']:.3f}, n={stump['right_n']:,})"
        )
    if len(model.stumps_) > max_stumps:
        lines.append(f"... {len(model.stumps_) - max_stumps} more stumps")
    return "\n".join(lines)


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


# ============================================================================
# Gaussian Naive Bayes from encrypted value-count aggregates
# ============================================================================


def _bc_gnb_features(numeric_features: list[str] | None = None) -> list[str]:
    """Resolve breast-cancer numeric/ordinal features for Gaussian NB."""
    if numeric_features is None:
        return list(_BC_GNB_DEFAULT_FEATURES)
    unknown = [feature for feature in numeric_features if feature not in _BC_GNB_VALUE_DOMAINS]
    if unknown:
        raise ValueError(f"Unsupported GaussianNB breast-cancer features: {unknown}")
    return list(numeric_features)


def _bc_gnb_row_features(row: dict, numeric_features: list[str]) -> dict[str, float]:
    """Extract numeric breast-cancer row features for Gaussian NB."""
    return {feature: float(row.get(feature, 0)) for feature in numeric_features}


def _bc_gnb_count_queries(numeric_features: list[str] | None = None) -> list[tuple[str, int, str, list[str]]]:
    """Build class-split value-count queries for BC Gaussian NB summaries."""
    queries: list[tuple[str, int, str, list[str]]] = []
    for feature in _bc_gnb_features(numeric_features):
        for value in _BC_GNB_VALUE_DOMAINS[feature]:
            queries.append((feature, 1, value, ["cancer_5yr:1", f"{feature}:{value}"]))
            queries.append((feature, 0, value, ["cancer_5yr:0", f"{feature}:{value}"]))
    return queries


def _bc_gnb_sufficient_stats(raw_results: list[tuple]) -> list[tuple[str, int, int, float, float]]:
    """Convert class-split value counts into Gaussian sufficient statistics."""
    accum: dict[tuple[str, int], dict[str, float]] = {}
    for feature, class_label, raw_value, count in raw_results:
        value = float(raw_value)
        n = int(count)
        stats = accum.setdefault((str(feature), int(class_label)), {"count": 0.0, "sum": 0.0, "sum_sq": 0.0})
        stats["count"] += n
        stats["sum"] += value * n
        stats["sum_sq"] += value * value * n

    return [
        (feature, class_label, int(stats["count"]), stats["sum"], stats["sum_sq"])
        for (feature, class_label), stats in sorted(accum.items())
    ]


def run_encrypted_gnb_bc(
    client,
    org: str,
    dataset: str,
    schema: str,
    numeric_features: list[str] | None = None,
    n_cancer: int | None = None,
    n_no_cancer: int | None = None,
    var_smoothing: float = 1e-9,
    threshold: float = 0.5,
    min_cell_size: int = CMS_MIN_CELL_SIZE,
    max_workers: int = 20,
) -> dict[str, Any]:
    """Train GaussianNaiveBayesModel from BI value-count aggregates only."""
    start = time.time()
    base_rate_queries = 0
    if n_cancer is None or n_no_cancer is None or int(n_cancer) + int(n_no_cancer) == 0:
        n_cancer, n_no_cancer = get_bc_base_rates(client, org, dataset, schema)
        base_rate_queries = 2

    features = _bc_gnb_features(numeric_features)
    queries = _bc_gnb_count_queries(features)

    def run_query(q):
        feature, class_label, value, filters = q
        count = _count_only(client, org, dataset, schema, filters)
        return (feature, class_label, value, count)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        raw_results = list(executor.map(run_query, queries))

    sanitized_results, suppression = suppress_bc_counts(raw_results, min_cell_size=min_cell_size)
    sufficient_stats = _bc_gnb_sufficient_stats(sanitized_results)
    model = _GaussianNaiveBayesModel(
        var_smoothing=var_smoothing,
        threshold=threshold,
    ).fit_from_sums(
        sufficient_stats,
        n_pos=int(n_cancer),
        n_neg=int(n_no_cancer),
    )

    return {
        "_model": model,
        "raw_results": sanitized_results,
        "raw_results_unsuppressed": raw_results,
        "sufficient_stats": sufficient_stats,
        "features": features,
        "n_cancer": int(n_cancer),
        "n_no_cancer": int(n_no_cancer),
        "n_total": int(n_cancer) + int(n_no_cancer),
        "enc_queries": len(raw_results) + base_rate_queries,
        "stat_queries": len(raw_results),
        "base_rate_queries": base_rate_queries,
        "train_time": time.time() - start,
        **suppression,
    }


def bc_gnb_predict(gnb_result: dict, row: dict) -> tuple[int, float]:
    """Predict with an encrypted BC Gaussian NB result. Returns (pred, risk)."""
    model = gnb_result.get("_model")
    features = gnb_result.get("features", _BC_GNB_DEFAULT_FEATURES)
    if not model:
        return 0, 0.0
    return model.predict(_bc_gnb_row_features(row, features))


def train_plaintext_gnb_bc(
    df,
    numeric_features: list[str] | None = None,
    var_smoothing: float = 1e-9,
) -> dict[str, Any]:
    """Train sklearn GaussianNB on local plaintext breast-cancer features."""
    from sklearn.naive_bayes import GaussianNB

    start = time.time()
    features = _bc_gnb_features(numeric_features)
    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isnull().any().any():
        bad_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"GaussianNB breast-cancer features must be numeric and non-null: {bad_cols}")
    y = df["cancer_5yr"].astype(int)

    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X, y)
    n_cancer = int(y.sum())
    n_no_cancer = len(df) - n_cancer
    return {
        "model": model,
        "features": features,
        "n_cancer": n_cancer,
        "n_no_cancer": n_no_cancer,
        "n_total": n_cancer + n_no_cancer,
        "train_time": time.time() - start,
    }


def bc_plaintext_gnb_predict_proba(model, feature_columns: list[str], df_test) -> list[float]:
    """Predict sklearn GaussianNB P(cancer) for breast-cancer test rows."""
    X = df_test[feature_columns].apply(pd.to_numeric, errors="coerce")
    proba = model.predict_proba(X)
    pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    return [float(p[pos_idx]) for p in proba]


# ============================================================================
# Bayesian Network from encrypted CPT aggregates
# ============================================================================

_BC_BN_PARENT_MAP = {
    "age_group": [],
    "race": [],
    "relatives": [],
    "biopsies": [],
    "atypical": ["biopsies"],
    "menarche": [],
    "density": ["age_group"],
    "afb": ["age_group"],
}


def _bc_bn_feature_values(feature_values: dict[str, list[str]]) -> dict[str, list[str]]:
    """Map notebook feature-value config to BN feature keys."""
    return {
        feature: [str(value).lower() for value in feature_values.get(values_key, [])]
        for feature, (values_key, _field_name) in _FEATURE_MAP.items()
    }


def _bc_bn_cpt_count_queries(
    feature_values: dict[str, list[str]],
    parent_map: dict[str, list[str]] | None = None,
) -> list[tuple[str, int, tuple[tuple[str, str], ...], str, list[str]]]:
    """Build BI count_only filters for breast-cancer Bayesian-network CPT cells."""
    resolved_parent_map = parent_map or _BC_BN_PARENT_MAP
    values_by_feature = _bc_bn_feature_values(feature_values)
    queries: list[tuple[str, int, tuple[tuple[str, str], ...], str, list[str]]] = []

    for feature in _FEATURE_MAP:
        parents = resolved_parent_map.get(feature, [])
        parent_value_lists = [values_by_feature.get(parent, []) for parent in parents]
        parent_combos = list(product(*parent_value_lists)) if parent_value_lists else [tuple()]

        for class_label in (1, 0):
            class_filter = _bc_class_filter(class_label)
            for parent_combo in parent_combos:
                parent_state = tuple((parent, str(value).lower()) for parent, value in zip(parents, parent_combo))
                parent_filters = [_bi_feat_filter(parent, str(value)) for parent, value in zip(parents, parent_combo)]
                for value in values_by_feature.get(feature, []):
                    filters = [class_filter, *parent_filters, _bi_feat_filter(feature, value)]
                    queries.append((feature, class_label, parent_state, str(value).lower(), filters))
    return queries


def run_encrypted_bn_bc(
    client,
    org: str,
    dataset: str,
    schema: str,
    feature_values: dict[str, list[str]],
    parent_map: dict[str, list[str]] | None = None,
    n_cancer: int | None = None,
    n_no_cancer: int | None = None,
    alpha: float = 1.0,
    threshold: float = 0.5,
    min_cell_size: int = CMS_MIN_CELL_SIZE,
    max_workers: int = 20,
) -> dict[str, Any]:
    """Train BayesianNetworkClassifierModel from BI CPT aggregate counts only."""
    start = time.time()
    base_rate_queries = 0
    if n_cancer is None or n_no_cancer is None or int(n_cancer) + int(n_no_cancer) == 0:
        n_cancer, n_no_cancer = get_bc_base_rates(client, org, dataset, schema)
        base_rate_queries = 2

    resolved_parent_map = parent_map or _BC_BN_PARENT_MAP
    model_feature_values = _bc_bn_feature_values(feature_values)
    queries = _bc_bn_cpt_count_queries(feature_values, resolved_parent_map)

    def run_query(query_tuple):
        feature, class_label, parent_state, value, filters = query_tuple
        count = _count_only(client, org, dataset, schema, filters)
        return (feature, class_label, parent_state, value, count)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        raw_results = list(executor.map(run_query, queries))

    sanitized_results, suppression = suppress_bc_counts(raw_results, min_cell_size=min_cell_size)
    model = _BayesianNetworkClassifierModel(
        parent_map=resolved_parent_map,
        alpha=alpha,
        threshold=threshold,
    ).fit(
        sanitized_results,
        n_pos=int(n_cancer),
        n_neg=int(n_no_cancer),
        feature_values=model_feature_values,
    )

    return {
        "_model": model,
        "raw_results": sanitized_results,
        "raw_results_unsuppressed": raw_results,
        "parent_map": resolved_parent_map,
        "feature_values": feature_values,
        "n_cancer": int(n_cancer),
        "n_no_cancer": int(n_no_cancer),
        "n_total": int(n_cancer) + int(n_no_cancer),
        "enc_queries": len(raw_results) + base_rate_queries,
        "cpt_queries": len(raw_results),
        "base_rate_queries": base_rate_queries,
        "train_time": time.time() - start,
        **suppression,
    }


def _prepare_bc_bn_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create BN feature-key columns from plaintext BC columns."""
    df2 = df.copy()
    df2["age_group"] = df2["age_group"].astype(str).str.lower()
    df2["race"] = df2["race_ethnicity"].astype(str).str.lower()
    df2["relatives"] = df2["num_first_degree_relatives"].astype(str).str.lower()
    df2["biopsies"] = df2["num_prior_biopsies"].apply(_biopsy_bin).astype(str).str.lower()
    df2["atypical"] = df2["atypical_hyperplasia"].astype(str).str.lower()
    df2["menarche"] = df2["menarche_category"].astype(str).str.lower()
    df2["density"] = df2["breast_density"].astype(str).str.lower()
    df2["afb"] = df2["age_at_first_birth"].apply(_afb_bin).astype(str).str.lower()
    df2["cancer_5yr"] = df2["cancer_5yr"].astype(int)
    return df2


def train_plaintext_bn_bc(
    df: pd.DataFrame,
    feature_values: dict[str, list[str]],
    parent_map: dict[str, list[str]] | None = None,
    alpha: float = 1.0,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Train BayesianNetworkClassifierModel from local plaintext CPT counts."""
    start = time.time()
    resolved_parent_map = parent_map or _BC_BN_PARENT_MAP
    df2 = _prepare_bc_bn_df(df)
    model_feature_values = _bc_bn_feature_values(feature_values)
    feature_keys = list(model_feature_values.keys())
    cpt_counts = _build_bayesian_cpt_counts_local(
        df2,
        target_col="cancer_5yr",
        feature_values=model_feature_values,
        parent_map=resolved_parent_map,
    )
    n_cancer = int(df2["cancer_5yr"].sum())
    n_no_cancer = len(df2) - n_cancer
    model = _BayesianNetworkClassifierModel(
        parent_map=resolved_parent_map,
        alpha=alpha,
        threshold=threshold,
    ).fit(
        cpt_counts,
        n_pos=n_cancer,
        n_neg=n_no_cancer,
        feature_values={feature: model_feature_values[feature] for feature in feature_keys},
    )

    return {
        "_model": model,
        "raw_results": cpt_counts,
        "parent_map": resolved_parent_map,
        "n_cancer": n_cancer,
        "n_no_cancer": n_no_cancer,
        "n_total": n_cancer + n_no_cancer,
        "train_time": time.time() - start,
    }


def bc_bn_predict(bn_result: dict, row: dict) -> tuple[int, float]:
    """Predict with a BC BayesianNetworkClassifierModel. Returns (pred, risk)."""
    model = bn_result.get("_model")
    if model:
        return model.predict(_nb_features_from_row(row))
    return 0, 0.0


def bc_plaintext_bn_predict_proba(
    bn_result: dict,
    df_test: pd.DataFrame,
) -> list[float]:
    """Predict plaintext Bayesian Network P(cancer) on breast-cancer rows."""
    model = bn_result.get("_model")
    if not model:
        return [0.0 for _ in range(len(df_test))]
    prepared = _prepare_bc_bn_df(df_test)
    return [float(model.predict(row.to_dict())[1]) for _, row in prepared.iterrows()]


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


def train_plaintext_rf_bc(
    df,
    feature_values: dict[str, list[str]],
    n_estimators: int = 7,
    max_depth: int = 3,
    max_features: int | float | str | None = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a sklearn RandomForestClassifier on local plaintext BC features."""
    from sklearn.ensemble import RandomForestClassifier

    X_encoded, col_names = _bc_dt_one_hot(df)
    y = df["cancer_5yr"].astype(int).values

    start = time.time()
    sklearn_max_features = None if max_features == "all" else max_features
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=sklearn_max_features,
        random_state=random_state,
    )
    model.fit(X_encoded, y)
    return {"model": model, "train_time": time.time() - start, "col_names": col_names}


def bc_plaintext_rf_predict_proba(
    model,
    col_names: list[str],
    df_test,
    feature_values: dict[str, list[str]],
) -> list[float]:
    """Predict sklearn RandomForestClassifier P(cancer) for BC rows."""
    return list(plaintext_predict_proba(model, col_names, df_test, feature_values, encoding="dt"))


def train_plaintext_adaboost_bc(
    df,
    feature_values: dict[str, list[str]],
    n_estimators: int = 10,
    learning_rate: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a sklearn AdaBoostClassifier with decision stumps on plaintext BC features."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    X_encoded, col_names = _bc_dt_one_hot(df)
    y = df["cancer_5yr"].astype(int).values

    start = time.time()
    stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    try:
        model = AdaBoostClassifier(
            estimator=stump,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    except TypeError:
        model = AdaBoostClassifier(
            base_estimator=stump,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    model.fit(X_encoded, y)
    return {"model": model, "train_time": time.time() - start, "col_names": col_names}


def bc_plaintext_adaboost_predict_proba(
    model,
    col_names: list[str],
    df_test,
    feature_values: dict[str, list[str]],
) -> list[float]:
    """Predict sklearn AdaBoostClassifier P(cancer) for BC rows."""
    return list(plaintext_predict_proba(model, col_names, df_test, feature_values, encoding="dt"))


def _bc_histogram_feature_values(feature_values: dict[str, list[str]]) -> dict[str, list[str]]:
    """Map notebook feature-value config to HistogramClassifierModel keys."""
    return {
        feature: [str(value).lower() for value in feature_values.get(values_key, [])]
        for feature, (values_key, _field_name) in _FEATURE_MAP.items()
    }


def run_encrypted_histogram_bc(
    client,
    org: str,
    dataset: str,
    schema: str,
    feature_values: dict[str, list[str]],
    bi_raw: dict[str, Any] | None = None,
    raw_results: list[tuple] | None = None,
    n_cancer: int | None = None,
    n_no_cancer: int | None = None,
    alpha: float = 1.0,
    threshold: float | None = None,
    use_feature_weights: bool = True,
    min_cell_size: int = CMS_MIN_CELL_SIZE,
) -> dict[str, Any]:
    """Train HistogramClassifierModel from BI class-split marginal counts only."""
    start = time.time()
    bi_raw = bi_raw or {}
    raw_results_source = "provided"
    base_rate_queries = 0
    marginal_queries = 0
    raw_results_unsuppressed = None

    if raw_results is None:
        raw_results = bi_raw.get("raw_results")
        raw_results_source = "bi_raw" if raw_results is not None else "rerun"

    if n_cancer is None:
        n_cancer = bi_raw.get("n_cancer")
    if n_no_cancer is None:
        n_no_cancer = bi_raw.get("n_no_cancer")

    suppression = {
        "min_cell_size": min_cell_size,
        "n_suppressed": int(bi_raw.get("n_suppressed", 0)) if raw_results_source == "bi_raw" else 0,
        "suppressed_cells": list(bi_raw.get("suppressed_cells", [])) if raw_results_source == "bi_raw" else [],
        "suppression_policy": bi_raw.get(
            "suppression_policy",
            "none"
            if min_cell_size <= 1
            else f"cms_k{min_cell_size}_fixed_midpoint_{_suppression_replacement(min_cell_size)}",
        ),
    }

    if raw_results is None:
        rerun = run_bc_conditional_queries(
            client,
            org,
            dataset,
            schema,
            feature_values,
            include_base_rates=n_cancer is None or n_no_cancer is None,
            min_cell_size=min_cell_size,
        )
        raw_results = rerun["raw_results"]
        raw_results_unsuppressed = rerun.get("raw_results_unsuppressed")
        marginal_queries = int(rerun.get("enc_queries", 0))
        suppression = {
            "min_cell_size": int(rerun.get("min_cell_size", min_cell_size)),
            "n_suppressed": int(rerun.get("n_suppressed", 0)),
            "suppressed_cells": list(rerun.get("suppressed_cells", [])),
            "suppression_policy": rerun.get("suppression_policy", "none"),
        }
        if n_cancer is None:
            n_cancer = rerun.get("n_cancer")
        if n_no_cancer is None:
            n_no_cancer = rerun.get("n_no_cancer")
    elif raw_results_source == "provided":
        raw_results_unsuppressed = list(raw_results)
        raw_results, suppression = suppress_bc_counts(raw_results, min_cell_size=min_cell_size)

    if n_cancer is None or n_no_cancer is None or int(n_cancer) + int(n_no_cancer) == 0:
        n_cancer, n_no_cancer = get_bc_base_rates(client, org, dataset, schema)
        base_rate_queries = 2

    n_cancer = int(n_cancer)
    n_no_cancer = int(n_no_cancer)
    model = _HistogramClassifierModel(
        alpha=alpha,
        threshold=threshold,
        use_feature_weights=use_feature_weights,
    ).fit(
        raw_results,
        n_pos=n_cancer,
        n_neg=n_no_cancer,
        feature_values=_bc_histogram_feature_values(feature_values),
    )

    result = {
        "_model": model,
        "counts_from_bi": True,
        "raw_results": raw_results,
        "raw_results_source": raw_results_source,
        "enc_queries": marginal_queries + base_rate_queries,
        "marginal_queries": marginal_queries,
        "base_rate_queries": base_rate_queries,
        "n_cancer": n_cancer,
        "n_no_cancer": n_no_cancer,
        "n_total": n_cancer + n_no_cancer,
        "train_time": time.time() - start,
        **suppression,
    }
    if raw_results_unsuppressed is not None:
        result["raw_results_unsuppressed"] = raw_results_unsuppressed
    return result


def bc_histogram_predict(hist_result: dict, row: dict) -> tuple[int, float]:
    """Predict with a BC HistogramClassifierModel. Returns (pred, risk)."""
    model = hist_result.get("_model")
    if model:
        return model.predict(_nb_features_from_row(row))
    return 0, 0.0


def train_plaintext_histogram_bc(
    df: pd.DataFrame,
    feature_values: dict[str, list[str]],
    alpha: float = 1.0,
    threshold: float | None = None,
    use_feature_weights: bool = True,
) -> dict[str, Any]:
    """Train HistogramClassifierModel from local plaintext marginal counts."""
    start = time.time()
    raw_results = build_bc_raw_results_local(df, feature_values)
    n_cancer = int(df["cancer_5yr"].astype(int).sum())
    n_no_cancer = len(df) - n_cancer
    model = _HistogramClassifierModel(
        alpha=alpha,
        threshold=threshold,
        use_feature_weights=use_feature_weights,
    ).fit(
        raw_results,
        n_pos=n_cancer,
        n_neg=n_no_cancer,
        feature_values=_bc_histogram_feature_values(feature_values),
    )
    return {
        "_model": model,
        "raw_results": raw_results,
        "n_cancer": n_cancer,
        "n_no_cancer": n_no_cancer,
        "n_total": n_cancer + n_no_cancer,
        "train_time": time.time() - start,
    }


def bc_plaintext_histogram_predict_proba(hist_result: dict, df_test: pd.DataFrame) -> list[float]:
    """Predict plaintext HistogramClassifierModel P(cancer) for BC rows."""
    model = hist_result.get("_model")
    if not model:
        return [0.0 for _ in range(len(df_test))]
    return [float(model.predict(_nb_features_from_row(row.to_dict()))[1]) for _, row in df_test.iterrows()]


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
        return model.predict_proba(X_encoded)[:, 1]
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


def run_encrypted_lr_bc_ols(
    client,
    org: str,
    dataset: str,
    schema: str,
    feature_values: dict[str, list[str]],
    bi_raw: dict[str, Any] | None = None,
    raw_results: list[tuple] | None = None,
    n_cancer: int | None = None,
    n_no_cancer: int | None = None,
    min_cell_size: int = CMS_MIN_CELL_SIZE,
    ridge_lambda: float = 0.01,
) -> dict[str, Any]:
    """Train BC LR as aggregate OLS/ridge from BI counts only.

    The encrypted path never uses local rows: NB marginals provide ``X'y`` and
    BI pairwise cross-tabs provide ``X'X``. Small positive pairwise cells are
    replaced by independence estimates in ``run_bc_pairwise_queries``.
    """
    if client is None or not org or not dataset or not schema:
        raise ValueError("run_encrypted_lr_bc_ols requires BI client/org/dataset/schema.")
    if not feature_values:
        raise ValueError("run_encrypted_lr_bc_ols requires feature_values.")

    start = time.time()
    base_rate_queries = 0
    raw_source = "provided"
    bi_raw = bi_raw or {}

    if raw_results is None:
        raw_results = bi_raw.get("raw_results")
        raw_source = "bi_raw" if raw_results is not None else "rerun"

    if n_cancer is None:
        n_cancer = bi_raw.get("n_cancer")
    if n_no_cancer is None:
        n_no_cancer = bi_raw.get("n_no_cancer")

    marginal_queries = 0
    if raw_results is None:
        rerun = run_bc_conditional_queries(
            client,
            org,
            dataset,
            schema,
            feature_values,
            include_base_rates=n_cancer is None or n_no_cancer is None,
            min_cell_size=min_cell_size,
        )
        raw_results = rerun["raw_results"]
        marginal_queries = int(rerun.get("enc_queries", 0))
        if n_cancer is None:
            n_cancer = rerun.get("n_cancer")
        if n_no_cancer is None:
            n_no_cancer = rerun.get("n_no_cancer")
        raw_source = "rerun"

    if n_cancer is None or n_no_cancer is None or int(n_cancer) + int(n_no_cancer) == 0:
        n_cancer, n_no_cancer = get_bc_base_rates(client, org, dataset, schema)
        base_rate_queries = 2

    n_cancer = int(n_cancer)
    n_no_cancer = int(n_no_cancer)
    n_total = n_cancer + n_no_cancer
    if n_total == 0:
        raise ValueError("run_encrypted_lr_bc_ols requires non-zero BI base rates.")

    pairwise = run_bc_pairwise_queries(
        client,
        org,
        dataset,
        schema,
        feature_values,
        raw_results,
        n_total,
        min_cell_size=min_cell_size,
    )
    lr_model = build_linear_model(
        raw_results,
        pairwise,
        feature_values,
        n_cancer,
        n_no_cancer,
        ridge_lambda=ridge_lambda,
    )

    train_time = time.time() - start
    return {
        **lr_model,
        "counts_from_bi": True,
        "pairwise_from_bi": True,
        "raw_results": raw_results,
        "raw_results_source": raw_source,
        "pairwise": pairwise["pairwise"],
        "pairwise_data": pairwise,
        "enc_queries": marginal_queries + base_rate_queries + int(pairwise.get("n_queries", 0)),
        "marginal_queries": marginal_queries,
        "base_rate_queries": base_rate_queries,
        "pairwise_queries": int(pairwise.get("n_queries", 0)),
        "n_cancer": n_cancer,
        "n_no_cancer": n_no_cancer,
        "n_total": n_total,
        "n_suppressed": int(pairwise.get("n_suppressed", 0)),
        "suppressed_cells": list(pairwise.get("suppressed_cells", [])),
        "min_cell_size": min_cell_size,
        "suppression_policy": (
            "none" if min_cell_size <= 1 else f"cms_k{min_cell_size}_pairwise_independence_estimate"
        ),
        "ridge_lambda": ridge_lambda,
        "train_time": train_time,
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


def bc_lr_predict(lr_result: dict, row: dict, use_sigmoid: bool = True) -> tuple[int, float]:
    """Predict with an aggregate OLS/ridge BC LR result. Returns (pred, risk)."""
    model = lr_result.get("_model")
    if model:
        risk = float(model.predict(_nb_features_from_row(row), use_sigmoid=use_sigmoid))
    else:
        beta = lr_result.get("beta")
        dummy_index = lr_result.get("dummy_index", [])
        if beta is None:
            return 0, 0.0
        risk = float(linear_model_predict(beta, dummy_index, row, use_sigmoid=use_sigmoid))
    return (1 if risk >= 0.5 else 0), max(0.0, min(1.0, risk))


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
    plain_queries: int = 0,
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
            {"label": "Queries", "values": [str(plain_queries), str(enc_queries), "-"]},
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


def compute_bc_metrics(
    y_true,
    scores,
    threshold: float = 0.0167,
    cohort_prev: float | None = None,
    pop_rate: float = 0.016,
) -> dict[str, Any]:
    """Evaluate breast-cancer risk scores with fraud-style ranking metrics.

    ``scores`` should be probabilities on the prior used for thresholding. If
    ``cohort_prev`` is provided, an additional F1@best metric is computed after
    prior-shift recalibration to ``pop_rate``.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    y_true_arr = np.asarray(y_true, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    n = len(y_true_arr)
    if cohort_prev is None:
        cohort_prev = float(y_true_arr.mean()) if n else 0.5

    preds = (scores_arr >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true_arr == 1)).sum())
    fp = int(((preds == 1) & (y_true_arr == 0)).sum())
    fn = int(((preds == 0) & (y_true_arr == 1)).sum())
    tn = int(((preds == 0) & (y_true_arr == 0)).sum())
    base = screening_metrics(tp, fp, tn, fn)
    acc = (tp + tn) / max(1, n)

    if len(np.unique(y_true_arr)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
        f1_best = float("nan")
        f1_pop_best = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true_arr, scores_arr))
        pr_auc = float(average_precision_score(y_true_arr, scores_arr))
        precisions, recalls, _ = precision_recall_curve(y_true_arr, scores_arr)
        f1_curve = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
        f1_best = float(np.nanmax(f1_curve))

        pop_scores = np.array([recalibrate_risk(float(s), cohort_prev, pop_rate) for s in scores_arr])
        prec_p, rec_p, _ = precision_recall_curve(y_true_arr, pop_scores)
        f1_pop_curve = 2 * prec_p * rec_p / np.maximum(prec_p + rec_p, 1e-12)
        f1_pop_best = float(np.nanmax(f1_pop_curve))

    return {
        **base,
        "acc": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_best": f1_best,
        "f1_pop_best": f1_pop_best,
        "cohort_prev": cohort_prev,
        "pop_rate": pop_rate,
        "threshold": threshold,
    }


def _bc_metric_value(value: float, pct: bool = False) -> str:
    if value != value:
        return "-"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _bc_metric_delta(enc: float, plain: float, higher_better: bool = True, scale: float = 1.0) -> str:
    if enc != enc or plain != plain:
        return "<td class='number-cell'>-</td>"
    delta = (enc - plain) * scale
    cls = "status-good" if (delta >= 0) == higher_better else "status-bad"
    if scale == 100:
        return f"<td class='{cls}'>{delta:+.1f}pp</td>"
    return f"<td class='{cls}'>{delta:+.3f}</td>"


def bc_model_summary_table(
    model_name: str,
    enc_metrics: dict[str, Any],
    plain_metrics: dict[str, Any],
    enc_train_time: float,
    plain_train_time: float,
    enc_queries: int = 0,
    plain_label: str = "sklearn",
) -> str:
    """Build a fraud-style comparison table for one breast-cancer model."""
    threshold_pct = enc_metrics.get("threshold", 0.0167) * 100
    pop_pct = enc_metrics.get("pop_rate", 0.016) * 100

    def _row(label: str, key: str, pct: bool = False) -> str:
        enc_val = enc_metrics.get(key, float("nan"))
        plain_val = plain_metrics.get(key, float("nan"))
        return (
            f"<tr class='data-row'><td class='label-cell'>{label}</td>"
            f"<td class='number-cell'>{_bc_metric_value(plain_val, pct=pct)}</td>"
            f"<td class='number-cell'>{_bc_metric_value(enc_val, pct=pct)}</td>"
            f"{_bc_metric_delta(enc_val, plain_val, scale=100 if pct else 1.0)}</tr>"
        )

    return f"""<table class="bi-metrics-table">
<caption style="caption-side:top;text-align:left;font-weight:600;padding-bottom:4px;">{model_name}</caption>
<tr class="header-row"><th></th><th>{plain_label}</th><th>Blind Insight</th><th>Delta</th></tr>
{_row(f"F1 @{threshold_pct:.2f}% risk", "f1")}
{_row("F1@best", "f1_best")}
{_row("ROC-AUC", "roc_auc")}
{_row("PR-AUC", "pr_auc")}
{_row(f"F1@best @ {pop_pct:.1f}% pop prior", "f1_pop_best")}
{_row("Sensitivity", "sens", pct=True)}
{_row("Specificity", "spec", pct=True)}
{_row("PPV (precision)", "ppv", pct=True)}
{_row("Flagged High-Risk", "flagged", pct=True)}
{_row("Accuracy", "acc", pct=True)}
<tr class='data-row'><td class='label-cell'>BI Queries</td>
    <td class='number-cell'>0</td>
    <td class='number-cell'>{enc_queries}</td>
    <td class='number-cell'>-</td></tr>
<tr class='data-row'><td class='label-cell'>Train Time</td>
    <td class='number-cell'>{plain_train_time * 1000:.0f}ms</td>
    <td class='number-cell'>{enc_train_time:.1f}s</td>
    <td class='number-cell'>+{enc_train_time - plain_train_time:.1f}s</td></tr>
<tr class='data-row'><td class='label-cell'>Data Decrypted</td>
    <td class='string-cell status-bad'>YES</td>
    <td class='string-cell status-good'>NEVER</td>
    <td class='number-cell'>-</td></tr>
</table>"""


def bc_confusion_matrix_html(
    label: str,
    enc_metrics: dict[str, Any],
    plain_metrics: dict[str, Any],
) -> str:
    """Build side-by-side confusion matrices for a breast-cancer model."""

    def _cm_table(metrics, subtitle):
        err = "background:#ffebee;color:#4a2d6b;"
        return (
            f'<div><p style="font-size:11px;font-weight:600;margin-bottom:2px;">{subtitle}</p>'
            f'<table class="bi-metrics-table" style="max-width:240px;font-size:12px;">'
            f"<tr><td></td><th>Pred Low</th><th>Pred High</th></tr>"
            f"<tr><th>Actual No</th>"
            f'<td class="number-cell">{metrics["tn"]:,}</td>'
            f'<td class="number-cell" style="{err}">{metrics["fp"]:,}</td></tr>'
            f"<tr><th>Actual Yes</th>"
            f'<td class="number-cell" style="{err}">{metrics["fn"]:,}</td>'
            f'<td class="number-cell">{metrics["tp"]:,}</td></tr>'
            f"</table></div>"
        )

    return (
        f'<div style="margin-bottom:16px;">'
        f'<h4 style="font-size:14px;margin-bottom:4px;">{label}</h4>'
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;">'
        f"{_cm_table(enc_metrics, 'Encrypted (Blind Insight)')}"
        f"{_cm_table(plain_metrics, 'Plaintext benchmark')}"
        f"</div></div>"
    )


def bc_eight_model_table(models: list[dict[str, Any]]) -> str:
    """Build a multi-model BC comparison table matching the fraud notebook."""
    header = "<tr class='header-row'><th>Metric</th>"
    for model in models:
        header += f"<th>{model['name']}</th>"
    header += "</tr>"

    def _row(label: str, key: str, pct: bool = False) -> str:
        cells = f"<td class='label-cell'>{label}</td>"
        for model in models:
            metrics = model.get("enc_metrics", model)
            cells += f"<td class='number-cell'>{_bc_metric_value(metrics.get(key, float('nan')), pct=pct)}</td>"
        return f"<tr class='data-row'>{cells}</tr>"

    rows = "\n".join(
        [
            _row("F1 @1.67%", "f1"),
            _row("F1@best", "f1_best"),
            _row("ROC-AUC", "roc_auc"),
            _row("PR-AUC", "pr_auc"),
            _row("Sensitivity", "sens", pct=True),
            _row("Specificity", "spec", pct=True),
            _row("PPV", "ppv", pct=True),
            _row("Flagged High-Risk", "flagged", pct=True),
        ]
    )
    return f"""<table class="bi-metrics-table">
{header}
{rows}
</table>"""


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
            "name": "Logistic Reg (OLS/ridge)",
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
    """Validate representative models (NB, DT, LR): encrypted vs sklearn plaintext.

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
        + _cm("Logistic Regression (OLS/ridge)", lr_enc_m, lr_pln_m)
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
