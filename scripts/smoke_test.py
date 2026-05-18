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
    "data_table",
    "training_summary_table",
    "run_encrypted_dt_fraud",
    "fraud_dt_predict",
    "train_plaintext_dt_fraud",
    "fraud_plaintext_predict_proba",
    "fraud_model_summary_table",
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
            ["NaiveBayesModel", "DecisionTreeModel", "LogisticRegressionModel"],
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
        from blind_ml import NaiveBayesModel

        m = NaiveBayesModel()
        assert m is not None

    check("NaiveBayesModel()", model_smoke)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(1)
