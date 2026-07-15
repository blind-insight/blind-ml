"""Tests for scripts/check_plaintext_leaks.py guardrails."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from check_plaintext_leaks import check_functions, check_models

_TMP_DIR = REPO_ROOT / ".test_tmp"
_counter = 0


def _write_tmp(source: str) -> Path:
    global _counter
    _TMP_DIR.mkdir(exist_ok=True)
    _counter += 1
    p = _TMP_DIR / f"_test_{_counter}.py"
    p.write_text(textwrap.dedent(source))
    return p


@pytest.fixture(autouse=True, scope="session")
def _cleanup_tmp():
    yield
    import shutil

    if _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR)


# ---------------------------------------------------------------------------
# Guardrail 1 — model fit methods
# ---------------------------------------------------------------------------


class TestGuardrail1:
    def test_sklearn_in_fit_method(self):
        """fit_ting() imports sklearn and calls .fit(X, Y) — must be caught."""
        path = _write_tmp("""\
            class RandomForestModel:
                def fit_ting(self, X, Y):
                    from sklearn.ensemble import RandomForestClassifier
                    clf = RandomForestClassifier()
                    clf.fit(X, Y)
                    return clf
        """)
        violations = check_models([path])
        assert len(violations) == 1
        assert "fit_ting()" in violations[0]

    def test_logistic_regression_fit_local(self):
        """fit_local() uses sklearn LogisticRegression — must be caught."""
        path = _write_tmp("""\
            class SomeModel:
                def fit_local(self, records, labels):
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression()
                    model.fit(records.to_numpy(), labels)
                    return model
        """)
        violations = check_models([path])
        assert len(violations) == 1
        assert "fit_local()" in violations[0]

    def test_read_csv_in_fit(self):
        """fit_from_plaintext() loads a CSV and trains sklearn — must be caught."""
        path = _write_tmp("""\
            class SomeModel:
                def fit_from_plaintext(self):
                    from pandas import read_csv
                    from sklearn.ensemble import RandomForestClassifier
                    records = read_csv("training.csv")
                    features = records.drop(columns=["target"])
                    labels = records["target"]
                    model = RandomForestClassifier()
                    model.fit(features, labels)
                    return model
        """)
        violations = check_models([path])
        assert len(violations) == 1
        assert "fit_from_plaintext()" in violations[0]

    def test_dataframe_param_caught(self):
        """fit() with a df param — original detection still works."""
        path = _write_tmp("""\
            class TreeModel:
                def fit(self, df, feature_cols, target_col):
                    import pandas as pd
                    X = pd.get_dummies(df[feature_cols])
                    return X
        """)
        violations = check_models([path])
        assert len(violations) == 1

    def test_dataframe_annotation_caught(self):
        """fit() with DataFrame type annotation — still caught."""
        path = _write_tmp("""\
            import pandas as pd
            class TreeModel:
                def fit(self, data: pd.DataFrame):
                    pass
        """)
        violations = check_models([path])
        assert len(violations) == 1

    def test_count_based_fit_clean(self):
        """fit_from_counts() using aggregate counts — must NOT be flagged."""
        path = _write_tmp("""\
            class DecisionTreeModel:
                def fit_from_counts(self, count_fn, feature_values, n_pos, n_neg):
                    candidates = []
                    for feature in feature_values:
                        for value in feature_values[feature]:
                            candidates.append((feature, value))
                    return candidates
        """)
        violations = check_models([path])
        assert violations == []

    def test_fit_from_sums_clean(self):
        """fit_from_sums() using aggregate stats — must NOT be flagged."""
        path = _write_tmp("""\
            class GaussianNaiveBayesModel:
                def fit_from_sums(self, sufficient_stats, n_pos=None, n_neg=None):
                    summaries = []
                    for feature_key, class_label, count, value_sum, squared_sum in sufficient_stats:
                        n = int(count)
                        mean = float(value_sum) / max(n, 1)
                        summaries.append((feature_key, int(class_label), n, mean, 0.0))
                    return self.fit(summaries, n_pos=n_pos, n_neg=n_neg)
        """)
        violations = check_models([path])
        assert violations == []

    def test_sklearn_in_docstring_not_flagged(self):
        """Docstring mentioning sklearn must NOT trigger a violation."""
        path = _write_tmp("""\
            class GaussianNaiveBayesModel:
                def fit(self, gaussian_stats, n_pos=None, n_neg=None):
                    \"\"\"Fit from summaries matching sklearn's GaussianNB convention.\"\"\"
                    return {"stats": gaussian_stats}
        """)
        violations = check_models([path])
        assert violations == []

    def test_allowlisted_method_skipped(self):
        """refine_irls is allowlisted — must NOT be flagged."""
        path = _write_tmp("""\
            class SomeModel:
                def refine_irls(self, df, weights):
                    import pandas as pd
                    return pd.get_dummies(df)
        """)
        violations = check_models([path])
        assert violations == []

    def test_standalone_function_ignored(self):
        """Guardrail 1 only checks methods inside classes."""
        path = _write_tmp("""\
            def fit_something(X, Y):
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier()
                clf.fit(X, Y)
                return clf
        """)
        violations = check_models([path])
        assert violations == []


# ---------------------------------------------------------------------------
# Guardrail 2 — encrypted-zone functions
# ---------------------------------------------------------------------------


class TestGuardrail2:
    def test_raw_array_params_caught(self):
        """run_encrypted_* with X, y params — must be caught."""
        path = _write_tmp("""\
            def run_encrypted_nn_fraud(X, y):
                model = _build_fraud_nn_count_provider(X, y)
                return {"_model": model}
        """)
        violations = check_functions([path])
        assert len(violations) == 1
        assert "param X, y" in violations[0]

    def test_features_labels_params_caught(self):
        """run_encrypted_* with features/labels params — must be caught."""
        path = _write_tmp("""\
            def run_encrypted_lr(features, labels, client=None):
                pass
        """)
        violations = check_functions([path])
        assert len(violations) == 1
        assert "features" in violations[0]

    def test_sklearn_in_encrypted_body(self):
        """run_encrypted_* importing sklearn directly — must be caught."""
        path = _write_tmp("""\
            def run_encrypted_train(client, org, dataset, schema):
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
                return model
        """)
        violations = check_functions([path])
        assert len(violations) == 1
        assert "sklearn" in violations[0]

    def test_read_csv_in_encrypted_body(self):
        """run_encrypted_* loading a CSV — must be caught."""
        path = _write_tmp("""\
            def run_encrypted_training(client):
                import pandas as pd
                df = pd.read_csv("training.csv")
                return df
        """)
        violations = check_functions([path])
        assert len(violations) == 1

    def test_bare_read_csv_in_encrypted_body(self):
        """run_encrypted_* using read_csv without pd prefix — must be caught."""
        path = _write_tmp("""\
            def run_encrypted_training(client):
                from pandas import read_csv
                df = read_csv("training.csv")
                return df
        """)
        violations = check_functions([path])
        assert len(violations) == 1

    def test_fit_with_df_in_encrypted_body(self):
        """run_encrypted_* calling .fit(df...) — must be caught."""
        path = _write_tmp("""\
            def run_encrypted_dt(df_local, client):
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier()
                model.fit(df_local, labels)
                return model
        """)
        violations = check_functions([path])
        assert len(violations) >= 1

    def test_clean_encrypted_function(self):
        """run_encrypted_* using BI queries — must NOT be flagged."""
        path = _write_tmp("""\
            def run_encrypted_dt_fraud(feature_values, client, org, dataset, schema):
                n_high, n_low = get_bi_base_rates(client, org, dataset, schema)
                count_fn = _build_fraud_dt_count_provider(
                    client=client, org=org, dataset=dataset, schema=schema,
                    feature_values=feature_values)
                dt = DecisionTreeModel()
                dt.fit_from_counts(count_fn=count_fn, feature_values=feature_values,
                                   n_pos=n_high, n_neg=n_low)
                return {"_model": dt}
        """)
        violations = check_functions([path])
        assert violations == []

    def test_clean_encrypted_with_model_fit(self):
        """run_encrypted_* calling model.fit() with aggregate results — must NOT be flagged."""
        path = _write_tmp("""\
            def run_encrypted_bn_fraud(client, org, dataset, schema, feature_values):
                n_high, n_low = get_bi_base_rates(client, org, dataset, schema)
                raw_results = []
                model = BayesianNetworkClassifierModel(
                    alpha=1.0,
                    threshold=0.5,
                ).fit(
                    raw_results,
                    n_pos=int(n_high),
                    n_neg=int(n_low),
                    feature_values=feature_values,
                )
                return {"_model": model}
        """)
        violations = check_functions([path])
        assert violations == []

    def test_non_encrypted_function_ignored(self):
        """Helper function not in encrypted zone — Guardrail 2 skips it."""
        path = _write_tmp("""\
            def _build_fraud_nn_count_provider(X, y):
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(hidden_layer_sizes=(100, 100))
                model.fit(X, y)
                return model
        """)
        violations = check_functions([path])
        assert violations == []

    def test_train_plaintext_excluded(self):
        """train_plaintext_* functions are explicitly excluded."""
        path = _write_tmp("""\
            def train_plaintext_dt(df, feature_cols, target_col):
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier()
                model.fit(df[feature_cols], df[target_col])
                return model
        """)
        violations = check_functions([path])
        assert violations == []

    def test_excluded_function_skipped(self):
        """Excluded validation functions are not flagged."""
        path = _write_tmp("""\
            def run_test_validation(df, model):
                from sklearn.metrics import accuracy_score
                return accuracy_score(df["target"], model.predict(df))
        """)
        violations = check_functions([path])
        assert violations == []

    def test_run_bi_prefix_scanned(self):
        """run_bi_* functions are in the encrypted zone."""
        path = _write_tmp("""\
            def run_bi_training(X, y, client):
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
                model.fit(X, y)
                return model
        """)
        violations = check_functions([path])
        assert len(violations) == 1
