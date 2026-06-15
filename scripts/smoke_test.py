#!/usr/bin/env python3
"""Repo smoketest — run from repo root: python3 scripts/smoke_test.py"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FRAUD_NOTEBOOK_SYMBOLS = [
    "load_env",
    "get_fraud_demo_config",
    "load_training_data",
    "load_test_data",
    "discover_feature_values",
    "run_bi_training",
    "train_plaintext_nb",
    "naive_bayes_predict",
    "naive_bayes_predict_proba",
    "compute_fraud_metrics",
    "FRAUD_PRODUCTION_PRIOR",
    "data_table",
    "training_summary_table",
    "run_encrypted_gnb_fraud",
    "train_plaintext_gnb_fraud",
    "fraud_gnb_predict",
    "fraud_plaintext_gnb_predict_proba",
    "run_encrypted_bn_fraud",
    "train_plaintext_bn_fraud",
    "fraud_bn_predict",
    "fraud_plaintext_bn_predict_proba",
    "run_encrypted_dt_fraud",
    "fraud_dt_predict",
    "fraud_dt_describe",
    "run_encrypted_rf_fraud",
    "fraud_rf_predict",
    "fraud_rf_describe",
    "train_plaintext_rf_fraud",
    "fraud_plaintext_rf_predict_proba",
    "run_encrypted_adaboost_fraud",
    "fraud_adaboost_predict",
    "fraud_adaboost_describe",
    "train_plaintext_adaboost_fraud",
    "fraud_plaintext_adaboost_predict_proba",
    "train_plaintext_dt_fraud",
    "fraud_plaintext_predict_proba",
    "fraud_model_summary_table",
    "compute_fraud_metrics",
    "naive_bayes_predict_proba",
    "FRAUD_PRODUCTION_PRIOR",
    "build_raw_results_local",
    "fraud_confusion_matrix_html",
    "compute_fraud_pairwise_local",
    "build_fraud_linear_model",
    "fraud_lr_predict",
    "refine_with_irls",
    "train_plaintext_lr",
    "build_plaintext_row",
    "fraud_three_model_table",
    "run_realtime_demo",
    "run_test_validation",
    "scaling_calculator_html",
]

BC_NOTEBOOK_SYMBOLS = [
    "load_env",
    "get_bc_demo_config",
    "load_bc_training_data",
    "load_bc_test_data",
    "discover_feature_values",
    "run_bc_conditional_queries",
    "build_bc_model",
    "train_plaintext_bc_nb",
    "bc_training_summary_table",
    "run_bc_full_validation",
    "run_model_comparison",
    "sample_comparison_table",
    "run_bc_realtime_demo",
    "run_encrypted_dt",
    "encrypted_dt_describe",
    "train_plaintext_bc_dt",
    "evaluate_bc_dt_nb_models",
    "bc_dt_summary_table",
    "bc_three_model_comparison_table",
    "build_bc_raw_results_local",
    "train_evaluate_bc_lr_models",
    "bc_lr_summary_table",
    "build_three_model_rows",
    "CMS_MIN_CELL_SIZE",
]


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        raise


def _require_symbols(module_name: str, symbols: list[str]) -> None:
    mod = importlib.import_module(module_name)
    missing = [s for s in symbols if not hasattr(mod, s)]
    if missing:
        raise AttributeError(f"{module_name} missing: {', '.join(missing)}")


def main() -> int:
    print("blind-ml smoketest\n")

    check("blind_ml package", lambda: importlib.import_module("blind_ml"))
    check(
        "model exports",
        lambda: _require_symbols(
            "blind_ml",
            [
                "NaiveBayesModel",
                "DecisionTreeModel",
                "LogisticRegressionModel",
                "GaussianNaiveBayesModel",
                "BayesianNetworkClassifierModel",
                "HistogramClassifierModel",
                "RandomForestModel",
                "AdaBoostStumpModel",
            ],
        ),
    )
    check("client", lambda: _require_symbols("blind_ml.client", ["BlindInsightClient"]))
    check("demo_helpers", lambda: _require_symbols("blind_ml.demo_helpers", ["data_table"]))
    check("healthcare", lambda: _require_symbols("blind_ml.healthcare", ["run_bc_full_validation"]))

    check("fraud notebook symbols", lambda: _require_symbols("blind_ml.demo_helpers", FRAUD_NOTEBOOK_SYMBOLS))
    check("breast_cancer notebook symbols", lambda: _require_symbols("blind_ml.healthcare", BC_NOTEBOOK_SYMBOLS))

    def demo_configs():
        dh = importlib.import_module("blind_ml.demo_helpers")
        hc = importlib.import_module("blind_ml.healthcare")
        assert dh.get_fraud_demo_config()["dataset"]
        assert hc.get_bc_demo_config()["schema"] == "train"
        assert hc.load_env is dh.load_env

    check("demo configs + load_env re-export", demo_configs)

    def client_defaults():
        client_mod = importlib.import_module("blind_ml.client")
        c = client_mod.BlindInsightClient()
        assert "localhost" in c.proxy_url or "blindinsight" in c.proxy_url

    check("BlindInsightClient()", client_defaults)

    def schemas():
        json.loads((REPO_ROOT / "schemas" / "fraud.json").read_text())
        json.loads((REPO_ROOT / "schemas" / "breast_cancer.json").read_text())

    check("JSON schemas", schemas)

    def upload_batches_dir():
        batches = REPO_ROOT / "demo_data" / "upload_batches"
        if not batches.is_dir():
            raise FileNotFoundError(batches)

    check("demo_data/upload_batches/ exists", upload_batches_dir)

    def model_smoke():

        from blind_ml import (
            AdaBoostStumpModel,
            BayesianNetworkClassifierModel,
            DecisionTreeModel,
            GaussianNaiveBayesModel,
            HistogramClassifierModel,
            NaiveBayesModel,
            RandomForestModel,
        )

        m = NaiveBayesModel()
        assert m is not None

        gnb = GaussianNaiveBayesModel().fit(
            [
                ("score", 1, 3, 10.0, 1.0),
                ("score", 0, 3, 0.0, 1.0),
            ],
            n_pos=3,
            n_neg=3,
        )
        pred, risk = gnb.predict({"score": 9.0})
        assert pred == 1
        assert risk > 0.5
        bn = BayesianNetworkClassifierModel(parent_map={"shape": ["color"]}).fit(
            [
                ("color", 1, tuple(), "red", 8),
                ("color", 0, tuple(), "red", 2),
                ("color", 1, tuple(), "blue", 1),
                ("color", 0, tuple(), "blue", 9),
                ("shape", 1, (("color", "red"),), "round", 7),
                ("shape", 0, (("color", "red"),), "round", 1),
                ("shape", 1, (("color", "blue"),), "round", 1),
                ("shape", 0, (("color", "blue"),), "round", 8),
            ],
            n_pos=9,
            n_neg=11,
            feature_values={"color": ["red", "blue"], "shape": ["round", "square"]},
        )
        pred, risk = bn.predict({"color": "red", "shape": "round"})
        assert pred == 1
        assert risk > 0.5

        hist = HistogramClassifierModel().fit(
            [
                ("color", 1, "red", 8),
                ("color", 0, "red", 2),
                ("color", 1, "blue", 1),
                ("color", 0, "blue", 9),
            ],
            n_pos=9,
            n_neg=11,
        )
        pred, risk = hist.predict({"color": "red"})
        assert pred == 1
        assert risk > 0.5

        rows = [
            {"color": "red", "shape": "round", "y": 1},
            {"color": "red", "shape": "square", "y": 1},
            {"color": "blue", "shape": "round", "y": 0},
            {"color": "blue", "shape": "square", "y": 0},
        ]

        def count_fn(path, feature, value, cls):
            total = 0
            for row in rows:
                if row["y"] != cls:
                    continue
                if any((str(row[f]).lower() == v) != branch for f, v, branch in path):
                    continue
                if str(row[feature]).lower() == value:
                    total += 1
            return total

        dt = DecisionTreeModel(max_depth=2).fit_from_counts(
            count_fn=count_fn,
            feature_values={"color": ["red", "blue"], "shape": ["round", "square"]},
            n_pos=2,
            n_neg=2,
        )
        pred, risk = dt.predict({"color": "red", "shape": "round"})
        assert pred == 1
        assert risk > 0.5

        rf = RandomForestModel(n_estimators=3, max_depth=2, max_features="all", random_state=42).fit_from_counts(
            count_fn=count_fn,
            feature_values={"color": ["red", "blue"], "shape": ["round", "square"]},
            n_pos=2,
            n_neg=2,
        )
        pred, risk = rf.predict({"color": "red", "shape": "round"})
        assert pred == 1
        assert risk > 0.5

        boost = AdaBoostStumpModel(n_estimators=3).fit_from_counts(
            count_fn=count_fn,
            feature_values={"color": ["red", "blue"], "shape": ["round", "square"]},
            n_pos=2,
            n_neg=2,
        )
        pred, risk = boost.predict({"color": "red", "shape": "round"})
        assert pred == 1
        assert risk > 0.5

    check(
        "NaiveBayesModel(), GaussianNaiveBayesModel(), BayesianNetworkClassifierModel(), HistogramClassifierModel(), DecisionTreeModel(), RandomForestModel(), AdaBoostStumpModel()",
        model_smoke,
    )

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(1)
