from __future__ import annotations

import numpy as np
import pandas as pd

from blind_ml.healthcare import (
    CMS_MIN_CELL_SIZE,
    bc_eight_model_table,
    bc_histogram_predict,
    bc_lr_predict,
    bc_plaintext_histogram_predict_proba,
    run_bc_pairwise_queries,
    run_encrypted_histogram_bc,
    run_encrypted_lr_bc_ols,
    run_model_comparison,
    train_plaintext_histogram_bc,
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
        "menarche_values": ["14_plus", "12_13"],
        "density_values": ["1", "4"],
        "afb_values": ["nulliparous", "30_plus"],
    }


def _raw_results(values: dict[str, list[str]]) -> list[tuple[str, int, str, int]]:
    mapping = {
        "age_group": "age_groups",
        "race": "race_values",
        "relatives": "relatives_values",
        "biopsies": "biopsies_values",
        "atypical": "atypical_values",
        "menarche": "menarche_values",
        "density": "density_values",
        "afb": "afb_values",
    }
    rows: list[tuple[str, int, str, int]] = []
    for feature, value_key in mapping.items():
        for value in values[value_key]:
            if value in {"50_59", "black", "1", "3_plus", "yes", "12_13", "4", "30_plus"}:
                rows.extend([(feature, 1, value, 30), (feature, 0, value, 10)])
            else:
                rows.extend([(feature, 1, value, 20), (feature, 0, value, 40)])
    return rows


def _bc_row() -> dict:
    return {
        "age_group": "50_59",
        "race_ethnicity": "black",
        "num_first_degree_relatives": 1,
        "num_prior_biopsies": 4,
        "atypical_hyperplasia": "yes",
        "menarche_category": "12_13",
        "breast_density": 4,
        "age_at_first_birth": 31,
        "age": 55,
        "cancer_5yr": 1,
        "is_cancer": 1,
    }


def test_bc_pairwise_queries_cover_feature_value_grid_and_independence_suppression():
    values = _mini_feature_values()
    raw_results = _raw_results(values)
    counts = {
        ("age_group:50_59", "race_ethnicity:black"): 3,
        ("num_prior_biopsies:3~7", "age_at_first_birth:30~47"): 13,
    }
    client = MockBIClient(counts)

    result = run_bc_pairwise_queries(
        client,
        "org",
        "dataset",
        "schema",
        values,
        raw_results,
        n_total=100,
        min_cell_size=CMS_MIN_CELL_SIZE,
    )

    assert result["n_queries"] == 28 * 4
    assert len(result["pairwise"]) == 28 * 4
    assert ("num_prior_biopsies:3~7", "age_at_first_birth:30~47") in client.calls
    assert result["pairwise"][("age_group", "50_59", "race", "black")] == 16.0
    assert result["n_suppressed"] == 1


def test_run_encrypted_lr_bc_ols_uses_bi_pairwise_counts_without_local_df():
    values = _mini_feature_values()
    raw_results = _raw_results(values)
    result = run_encrypted_lr_bc_ols(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        values,
        bi_raw={"raw_results": raw_results, "n_cancer": 50, "n_no_cancer": 50},
        ridge_lambda=0.01,
    )

    assert result["counts_from_bi"] is True
    assert result["pairwise_from_bi"] is True
    assert result["pairwise_queries"] == 28 * 4
    assert result["min_cell_size"] == CMS_MIN_CELL_SIZE
    assert result["ridge_lambda"] == 0.01
    assert result["pairwise"]
    assert result["enc_queries"] == result["pairwise_queries"]


def test_bc_lr_predict_maps_raw_rows_through_binned_nb_features():
    pred, risk = bc_lr_predict(
        {"beta": np.array([-8.0, 16.0]), "dummy_index": [("biopsies", "3_plus")]},
        _bc_row(),
        use_sigmoid=True,
    )

    assert pred == 1
    assert risk > 0.99


def test_run_encrypted_histogram_bc_fits_from_bi_marginals_and_sanitizes_small_cells():
    values = _mini_feature_values()
    raw_results = [
        ("biopsies", 1, "3_plus", 3),
        ("biopsies", 0, "3_plus", 20),
        ("biopsies", 1, "0", 20),
        ("biopsies", 0, "0", 40),
    ]

    result = run_encrypted_histogram_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        values,
        raw_results=raw_results,
        n_cancer=25,
        n_no_cancer=60,
        min_cell_size=CMS_MIN_CELL_SIZE,
    )

    assert result["counts_from_bi"] is True
    assert result["raw_results_source"] == "provided"
    assert ("biopsies", 1, "3_plus", 5) in result["raw_results"]
    assert result["raw_results_unsuppressed"] == raw_results
    assert result["n_suppressed"] == 1
    assert result["min_cell_size"] == CMS_MIN_CELL_SIZE
    assert result["_model"].histograms["biopsies"]["3_plus"]["n_pos"] == 5.0


def test_bc_histogram_predict_maps_binned_features():
    values = _mini_feature_values()
    result = run_encrypted_histogram_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        values,
        raw_results=[("biopsies", 1, "3_plus", 30), ("biopsies", 0, "3_plus", 0)],
        n_cancer=30,
        n_no_cancer=30,
        min_cell_size=CMS_MIN_CELL_SIZE,
    )

    pred, risk = bc_histogram_predict(result, _bc_row())

    assert pred == 1
    assert risk > 0.9


def test_plaintext_histogram_bc_uses_local_data_only():
    df = pd.DataFrame(
        [
            {**_bc_row(), "cancer_5yr": 1},
            {**_bc_row(), "cancer_5yr": 1},
            {**_bc_row(), "num_prior_biopsies": 0, "cancer_5yr": 0},
        ]
    )

    result = train_plaintext_histogram_bc(df, _mini_feature_values())
    probs = bc_plaintext_histogram_predict_proba(result, df)

    assert result["n_cancer"] == 2
    assert result["n_no_cancer"] == 1
    assert result["raw_results"]
    assert len(probs) == len(df)
    assert all(0.0 <= prob <= 1.0 for prob in probs)


def test_bc_eight_model_table_includes_all_eight_model_names():
    names = [
        "Naive Bayes",
        "Gaussian NB",
        "Bayesian Network",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "Logistic Regression",
        "Histogram",
    ]
    html = bc_eight_model_table([{"name": name, "enc_metrics": {"f1": 0.1}} for name in names])

    for name in names:
        assert name in html


def test_new_encrypted_results_expose_min_cell_size_where_suppression_applies():
    values = _mini_feature_values()
    raw_results = _raw_results(values)
    lr = run_encrypted_lr_bc_ols(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        values,
        bi_raw={"raw_results": raw_results, "n_cancer": 50, "n_no_cancer": 50},
    )
    hist = run_encrypted_histogram_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        values,
        raw_results=raw_results,
        n_cancer=50,
        n_no_cancer=50,
    )

    assert lr["min_cell_size"] == CMS_MIN_CELL_SIZE
    assert hist["min_cell_size"] == CMS_MIN_CELL_SIZE


def test_bcrat_bcsc_comparison_still_runs():
    df = pd.DataFrame([_bc_row(), {**_bc_row(), "patient_id": "2", "cancer_5yr": 0, "is_cancer": 0}])
    p = {"biopsies": {1: {"3_plus": 0.8}, 0: {"3_plus": 0.2}}}

    result = run_model_comparison(df, 0.5, 0.5, p)

    assert "BCRAT (Gail)" in result["summary_html"]
    assert "BCSC" in result["summary_html"]
    assert len(result["comp_df"]) == 2
