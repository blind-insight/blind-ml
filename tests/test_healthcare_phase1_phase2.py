from __future__ import annotations

import pandas as pd

from blind_ml.healthcare import (
    CMS_MIN_CELL_SIZE,
    bc_gnb_predict,
    bc_model_summary_table,
    bc_plaintext_gnb_predict_proba,
    bc_confusion_matrix_html,
    compute_bc_metrics,
    get_bc_feature_values,
    run_encrypted_gnb_bc,
    suppress_bc_counts,
    train_plaintext_gnb_bc,
)


class MockBIClient:
    def __init__(self, counts: dict[tuple[str, ...], int]):
        self.counts = counts
        self.calls: list[tuple[str, ...]] = []

    def query(self, **kwargs):
        filters = tuple(kwargs.get("filters", []))
        self.calls.append(filters)
        return {"count": self.counts.get(filters, 0)}


def test_get_bc_feature_values_static_domains():
    values = get_bc_feature_values()

    assert values["age_groups"] == ["40_49", "50_59", "60_69", "70_74"]
    assert values["biopsies_values"] == ["0", "1", "2", "3_plus"]
    assert values["afb_values"] == ["nulliparous", "under_20", "20_24", "25_29", "30_plus"]
    assert "white" in values["race_values"]
    assert values["density_values"] == ["1", "2", "3", "4"]


def test_suppress_bc_counts_keeps_zero_replaces_small_preserves_large():
    sanitized, meta = suppress_bc_counts(
        [
            ("density", 1, "1", 0),
            ("density", 1, "2", 3),
            ("density", 1, "3", CMS_MIN_CELL_SIZE),
        ]
    )

    assert sanitized == [
        ("density", 1, "1", 0),
        ("density", 1, "2", 5),
        ("density", 1, "3", CMS_MIN_CELL_SIZE),
    ]
    assert meta["n_suppressed"] == 1
    assert meta["min_cell_size"] == CMS_MIN_CELL_SIZE
    assert meta["suppressed_cells"] == ["density=2|class=1"]


def test_run_encrypted_gnb_bc_trains_from_mocked_aggregate_counts():
    counts = {
        ("cancer_5yr:1",): 32,
        ("cancer_5yr:0",): 35,
        ("cancer_5yr:1", "breast_density:1"): 0,
        ("cancer_5yr:1", "breast_density:2"): 3,
        ("cancer_5yr:1", "breast_density:3"): 12,
        ("cancer_5yr:1", "breast_density:4"): 15,
        ("cancer_5yr:0", "breast_density:1"): 20,
        ("cancer_5yr:0", "breast_density:2"): 8,
        ("cancer_5yr:0", "breast_density:3"): 10,
        ("cancer_5yr:0", "breast_density:4"): 2,
    }
    client = MockBIClient(counts)

    result = run_encrypted_gnb_bc(
        client,
        "org",
        "dataset",
        "schema",
        numeric_features=["breast_density"],
        max_workers=1,
    )

    assert result["n_cancer"] == 32
    assert result["n_no_cancer"] == 35
    assert result["n_suppressed"] == 4
    assert result["base_rate_queries"] == 2
    assert ("breast_density", 1, "2", 5) in result["raw_results"]
    assert ("breast_density", 1, "2", 3) in result["raw_results_unsuppressed"]
    assert result["sufficient_stats"]

    pred, risk = bc_gnb_predict(result, {"breast_density": 4})
    assert pred in (0, 1)
    assert 0.0 <= risk <= 1.0


def test_plaintext_gnb_bc_benchmark_helpers():
    df = pd.DataFrame(
        {
            "breast_density": [1, 2, 3, 4],
            "cancer_5yr": [0, 0, 1, 1],
        }
    )

    model = train_plaintext_gnb_bc(df, numeric_features=["breast_density"])
    probs = bc_plaintext_gnb_predict_proba(model["model"], model["features"], df)

    assert model["n_cancer"] == 2
    assert model["n_no_cancer"] == 2
    assert len(probs) == len(df)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_bc_metrics_and_summary_tables_include_ranking_metrics():
    enc = compute_bc_metrics(
        [0, 0, 1, 1],
        [0.01, 0.02, 0.03, 0.04],
        threshold=0.0167,
        cohort_prev=0.5,
        pop_rate=0.016,
    )
    plain = compute_bc_metrics(
        [0, 0, 1, 1],
        [0.01, 0.03, 0.02, 0.04],
        threshold=0.0167,
        cohort_prev=0.5,
        pop_rate=0.016,
    )

    assert enc["roc_auc"] == 1.0
    assert enc["pr_auc"] == 1.0
    assert enc["tp"] == 2
    assert enc["fp"] == 1
    assert "ROC-AUC" in bc_model_summary_table("Model", enc, plain, 1.0, 0.1, enc_queries=12)
    assert "Pred High" in bc_confusion_matrix_html("Model", enc, plain)
