from __future__ import annotations

import pandas as pd

from blind_ml.healthcare import (
    bc_adaboost_predict,
    bc_plaintext_adaboost_predict_proba,
    bc_plaintext_rf_predict_proba,
    bc_rf_predict,
    run_encrypted_adaboost_bc,
    run_encrypted_rf_bc,
    train_plaintext_adaboost_bc,
    train_plaintext_rf_bc,
)


class MockBIClient:
    def __init__(self, counts: dict[tuple[str, ...], int]):
        self.counts = counts
        self.calls: list[tuple[str, ...]] = []

    def query(self, **kwargs):
        filters = tuple(kwargs.get("filters", []))
        self.calls.append(filters)
        return {"count": self.counts.get(filters, 0)}


def _mini_feature_values() -> dict[str, list[str]]:
    return {
        "age_groups": ["40_49", "50_59"],
        "race_values": ["white", "black"],
        "relatives_values": ["0", "1"],
        "biopsies_values": ["0", "3_plus"],
        "atypical_values": ["no", "yes"],
        "menarche_values": ["12_13", "14_plus"],
        "density_values": ["1", "4"],
        "afb_values": ["nulliparous", "30_plus"],
    }


def _complete_raw_results(age_left_pos: int = 20, age_left_neg: int = 80) -> list[tuple[str, int, str, int]]:
    age_right_pos = 100 - age_left_pos
    age_right_neg = 100 - age_left_neg
    raw_results = [
        ("age_group", 1, "40_49", age_left_pos),
        ("age_group", 0, "40_49", age_left_neg),
        ("age_group", 1, "50_59", age_right_pos),
        ("age_group", 0, "50_59", age_right_neg),
    ]
    for feature, raw_values in {
        "race": ["white", "black"],
        "relatives": ["0", "1"],
        "biopsies": ["0", "3_plus"],
        "atypical": ["no", "yes"],
        "menarche": ["12_13", "14_plus"],
        "density": ["1", "4"],
        "afb": ["nulliparous", "30_plus"],
    }.items():
        for value in raw_values:
            count = 100 if value in {"white", "0", "no", "12_13", "1", "nulliparous"} else 0
            raw_results.extend([(feature, 1, value, count), (feature, 0, value, count)])
    return raw_results


def _row() -> dict:
    return {
        "age_group": "50_59",
        "race_ethnicity": "white",
        "num_first_degree_relatives": 0,
        "num_prior_biopsies": 0,
        "atypical_hyperplasia": "no",
        "menarche_category": "12_13",
        "breast_density": 1,
        "age_at_first_birth": 0,
    }


def test_run_encrypted_rf_bc_trains_from_aggregate_counts_and_reuses_dt_cache():
    dt_result = {
        "feature_values": _mini_feature_values(),
        "raw_results": _complete_raw_results(),
        "n_cancer": 100,
        "n_no_cancer": 100,
        "query_cache": {("cancer_5yr:1", "age_group:40_49"): 20},
    }

    result = run_encrypted_rf_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        dt_result=dt_result,
        n_estimators=2,
        max_depth=1,
        max_features="all",
        k_min=11,
    )

    assert result["counts_from_bi"] is True
    assert result["raw_results_source"] == "dt_result"
    assert result["reused_cache_entries"] == 1
    assert result["additional_rf_queries"] == 0
    assert result["min_cell_size"] == 11
    pred, risk = bc_rf_predict(result, _row())
    assert pred in (0, 1)
    assert 0.0 <= risk <= 1.0


def test_run_encrypted_rf_bc_k_min_blocks_unsafe_small_positive_split():
    result = run_encrypted_rf_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        feature_values=_mini_feature_values(),
        raw_results=_complete_raw_results(age_left_pos=5, age_left_neg=90),
        n_cancer=100,
        n_no_cancer=100,
        n_estimators=1,
        max_depth=1,
        max_features="all",
        k_min=11,
    )

    assert result["trees"][0]["type"] == "leaf"


def test_run_encrypted_adaboost_bc_trains_from_aggregate_counts_and_reuses_rf_cache():
    rf_result = {
        "feature_values": _mini_feature_values(),
        "raw_results": _complete_raw_results(),
        "n_cancer": 100,
        "n_no_cancer": 100,
        "query_cache": {("cancer_5yr:1", "age_group:40_49"): 20},
    }

    result = run_encrypted_adaboost_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        rf_result=rf_result,
        n_estimators=1,
        k_min=11,
    )

    assert result["counts_from_bi"] is True
    assert result["raw_results_source"] == "rf_result"
    assert result["reused_cache_entries"] == 1
    assert result["additional_adaboost_queries"] == 0
    assert result["min_cell_size"] == 11
    pred, risk = bc_adaboost_predict(result, _row())
    assert pred in (0, 1)
    assert 0.0 <= risk <= 1.0


def test_run_encrypted_adaboost_bc_k_min_blocks_unsafe_stump():
    result = run_encrypted_adaboost_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        feature_values=_mini_feature_values(),
        raw_results=_complete_raw_results(age_left_pos=5, age_left_neg=90),
        n_cancer=100,
        n_no_cancer=100,
        n_estimators=1,
        k_min=11,
    )

    assert result["stumps"] == []


def test_plaintext_rf_and_adaboost_benchmark_helpers():
    df = pd.DataFrame(
        {
            "age_group": ["40_49", "50_59", "50_59", "40_49"],
            "race_ethnicity": ["white", "white", "black", "black"],
            "num_first_degree_relatives": [0, 1, 0, 1],
            "num_prior_biopsies": [0, 3, 0, 3],
            "atypical_hyperplasia": ["no", "yes", "no", "yes"],
            "menarche_category": ["12_13", "14_plus", "12_13", "14_plus"],
            "breast_density": [1, 4, 1, 4],
            "age_at_first_birth": [0, 31, 0, 31],
            "cancer_5yr": [0, 1, 0, 1],
        }
    )

    rf = train_plaintext_rf_bc(df, _mini_feature_values(), n_estimators=2, max_depth=1, max_features="all")
    rf_probs = bc_plaintext_rf_predict_proba(rf["model"], rf["col_names"], df, _mini_feature_values())
    boost = train_plaintext_adaboost_bc(df, _mini_feature_values(), n_estimators=2)
    boost_probs = bc_plaintext_adaboost_predict_proba(boost["model"], boost["col_names"], df, _mini_feature_values())

    assert len(rf_probs) == len(df)
    assert len(boost_probs) == len(df)
    assert all(0.0 <= p <= 1.0 for p in rf_probs)
    assert all(0.0 <= p <= 1.0 for p in boost_probs)
