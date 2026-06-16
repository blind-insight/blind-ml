"""
blind_ml -- Machine Learning toolkit for Blind Insight.

Generic train/predict for Naive Bayes, Decision Trees, and Logistic
Regression using encrypted aggregate counts or local data mirrors.
No domain knowledge -- works with any categorical dataset.

Usage::

    from blind_ml import NaiveBayesModel, DecisionTreeModel, LogisticRegressionModel

    nb = NaiveBayesModel().fit(marginal_counts, n_pos, n_neg)
    pred, risk = nb.predict(row_features)

    dt = DecisionTreeModel(max_depth=3).fit(df, feature_cols, target_col)
    pred, risk = dt.predict(row_dict)

    lr = LogisticRegressionModel(ridge_lambda=1e-6)
    lr.fit_from_counts(marginals, pos_counts, pairwise, dummy_idx, ...)
    lr.refine_irls(X, y)
    prob = lr.predict(row_features)
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# IMPURITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def gini(n_pos: int, n_neg: int) -> float:
    total = n_pos + n_neg
    if total == 0:
        return 0.0
    p = n_pos / total
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def entropy(n_pos: int, n_neg: int) -> float:
    total = n_pos + n_neg
    if total == 0:
        return 0.0
    probs = [c / total for c in (n_pos, n_neg) if c > 0]
    return -sum(p * math.log2(p) for p in probs)


# ═══════════════════════════════════════════════════════════════════════════════
# NAIVE BAYES
# ═══════════════════════════════════════════════════════════════════════════════


class NaiveBayesModel:
    """Categorical Naive Bayes trained from aggregate marginal counts.

    Works identically whether counts come from encrypted BI aggregate
    queries or from a local plaintext DataFrame.
    """

    def __init__(self) -> None:
        self.P_pos: float = 0.5
        self.P_neg: float = 0.5
        self.tables: dict[str, dict[int, dict[str, float]]] = {}
        self.feature_keys: list[str] = []
        self.train_time: float = 0.0

    def fit(
        self,
        marginal_counts: list[tuple[str, int, str, int]],
        n_pos: int,
        n_neg: int,
        feature_values: dict[str, list[str]] | None = None,
    ) -> NaiveBayesModel:
        """Fit probability tables from aggregate counts with Laplace smoothing.

        Parameters
        ----------
        marginal_counts : list of (feature_key, class_label, value, count)
        n_pos, n_neg : class totals
        feature_values : {feature_key: [values]} for Laplace denominator;
                         inferred from counts if omitted.
        """
        start = time.time()
        n_total = n_pos + n_neg
        self.P_pos = n_pos / n_total if n_total > 0 else 0.5
        self.P_neg = n_neg / n_total if n_total > 0 else 0.5

        feat_vals: dict[str, set] = {}
        for fk, _cls, val, _count in marginal_counts:
            feat_vals.setdefault(fk, set()).add(val.lower())
        self.feature_keys = sorted(feat_vals.keys())

        n_vals: dict[str, int] = {}
        for fk in self.feature_keys:
            if feature_values:
                n_vals[fk] = len(feature_values.get(fk, feat_vals.get(fk, set())))
            else:
                n_vals[fk] = len(feat_vals.get(fk, set()))

        self.tables = {fk: {1: {}, 0: {}} for fk in self.feature_keys}
        for fk, cls, val, count in marginal_counts:
            n_class = n_pos if cls == 1 else n_neg
            nv = n_vals.get(fk, 1)
            self.tables[fk][cls][val.lower()] = (count + 1) / (n_class + nv)

        self.train_time = time.time() - start
        return self

    def predict(self, row_features: dict[str, str]) -> tuple[int, float]:
        """Return (predicted_class, posterior_risk).

        Parameters
        ----------
        row_features : {feature_key: value} with keys matching those in fit().
        """
        eps = 1e-10
        log_pos = math.log(self.P_pos + eps)
        log_neg = math.log(self.P_neg + eps)

        for fk in self.feature_keys:
            val = str(row_features.get(fk, "")).lower()
            if fk in self.tables:
                log_pos += math.log(max(self.tables[fk][1].get(val, 0.1), eps))
                log_neg += math.log(max(self.tables[fk][0].get(val, 0.1), eps))

        max_log = max(log_pos, log_neg)
        p_pos = math.exp(log_pos - max_log)
        p_neg = math.exp(log_neg - max_log)
        risk = p_pos / (p_pos + p_neg)
        pred = 1 if log_pos > log_neg else 0
        return pred, risk

    def predict_class(self, row_features: dict[str, str]) -> int:
        return self.predict(row_features)[0]

    def predict_risk(self, row_features: dict[str, str]) -> float:
        return self.predict(row_features)[1]


# ═══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN NAIVE BAYES
# ═══════════════════════════════════════════════════════════════════════════════


class GaussianNaiveBayesModel:
    """Gaussian Naive Bayes trained from class-conditional numeric summaries.

    Each feature is modeled as normally distributed within each class using
    ``count``, ``mean``, and population ``variance``. These summaries can come
    from local plaintext data or encrypted aggregate queries.
    """

    def __init__(
        self,
        var_smoothing: float = 1e-9,
        threshold: float = 0.5,
    ) -> None:
        self.var_smoothing = var_smoothing
        self.threshold = threshold
        self.P_pos: float = 0.5
        self.P_neg: float = 0.5
        self.stats: dict[str, dict[int, dict[str, float]]] = {}
        self.feature_keys: list[str] = []
        self.epsilon_: float = 1e-12
        self.train_time: float = 0.0

    def fit(
        self,
        gaussian_stats: list[tuple[str, int, int, float, float]],
        n_pos: int | None = None,
        n_neg: int | None = None,
        global_variance: float | None = None,
    ) -> GaussianNaiveBayesModel:
        """Fit from class-conditional Gaussian summaries.

        Parameters
        ----------
        gaussian_stats : list of (feature_key, class_label, count, mean, variance)
            ``variance`` must be the population variance for that feature within
            the class, matching sklearn's GaussianNB convention.
        n_pos, n_neg : optional class totals. If omitted, inferred from summary
            counts by taking the largest count seen for each class.
        global_variance : optional maximum overall feature variance for sklearn-
            style smoothing. If omitted, max class-conditional variance is used.
        """
        start = time.time()
        if not gaussian_stats:
            raise ValueError("gaussian_stats must contain at least one feature summary")

        grouped: dict[str, dict[int, dict[str, float]]] = {}
        inferred_counts = {1: 0, 0: 0}
        max_class_variance = 0.0

        for feature_key, class_label, count, mean, variance in gaussian_stats:
            cls = int(class_label)
            if cls not in (0, 1):
                raise ValueError("GaussianNaiveBayesModel supports binary class labels 0 and 1")
            n = int(count)
            var = max(float(variance), 0.0)
            grouped.setdefault(feature_key, {})[cls] = {
                "count": float(n),
                "mean": float(mean),
                "var": var,
            }
            inferred_counts[cls] = max(inferred_counts[cls], n)
            max_class_variance = max(max_class_variance, var)

        if n_pos is None:
            n_pos = inferred_counts[1]
        if n_neg is None:
            n_neg = inferred_counts[0]

        n_total = int(n_pos) + int(n_neg)
        self.P_pos = int(n_pos) / n_total if n_total > 0 else 0.5
        self.P_neg = int(n_neg) / n_total if n_total > 0 else 0.5

        smoothing_source = max(float(global_variance or 0.0), max_class_variance)
        self.epsilon_ = max(self.var_smoothing * smoothing_source, 1e-12)
        self.feature_keys = sorted(grouped.keys())
        self.stats = {feature_key: {} for feature_key in self.feature_keys}

        for feature_key in self.feature_keys:
            for cls in (0, 1):
                if cls not in grouped[feature_key]:
                    continue
                class_stats = grouped[feature_key][cls]
                self.stats[feature_key][cls] = {
                    "count": class_stats["count"],
                    "mean": class_stats["mean"],
                    "var": max(class_stats["var"] + self.epsilon_, 1e-12),
                }

        self.train_time = time.time() - start
        return self

    def fit_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_col: str,
    ) -> GaussianNaiveBayesModel:
        """Fit from a plaintext DataFrame of numeric features."""
        if not feature_columns:
            raise ValueError("feature_columns must contain at least one feature")

        X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        if X.isnull().any().any():
            bad_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"GaussianNaiveBayesModel requires numeric, non-null features: {bad_cols}")

        y = df[target_col].astype(int)
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("GaussianNaiveBayesModel requires binary target labels 0 and 1")

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        global_variance = float(np.var(X.values.astype(np.float64), axis=0).max())

        summaries: list[tuple[str, int, int, float, float]] = []
        for feature_key in feature_columns:
            values = X[feature_key].values.astype(np.float64)
            for cls in (1, 0):
                class_values = values[(y == cls).values]
                count = len(class_values)
                mean = float(class_values.mean()) if count else 0.0
                variance = float(class_values.var()) if count else 0.0
                summaries.append((feature_key, cls, count, mean, variance))

        return self.fit(summaries, n_pos=n_pos, n_neg=n_neg, global_variance=global_variance)

    def fit_from_sums(
        self,
        sufficient_stats: list[tuple[str, int, int, float, float]],
        n_pos: int | None = None,
        n_neg: int | None = None,
    ) -> GaussianNaiveBayesModel:
        """Fit from (feature_key, class_label, count, sum, sum_of_squares)."""
        summaries: list[tuple[str, int, int, float, float]] = []
        feature_totals: dict[str, dict[str, float]] = {}

        for feature_key, class_label, count, value_sum, squared_sum in sufficient_stats:
            n = int(count)
            if n > 0:
                mean = float(value_sum) / n
                variance = max(float(squared_sum) / n - mean * mean, 0.0)
            else:
                mean = 0.0
                variance = 0.0
            totals = feature_totals.setdefault(feature_key, {"count": 0.0, "sum": 0.0, "sum_sq": 0.0})
            totals["count"] += n
            totals["sum"] += float(value_sum)
            totals["sum_sq"] += float(squared_sum)
            summaries.append((feature_key, int(class_label), n, mean, variance))

        global_variance = 0.0
        for totals in feature_totals.values():
            n = totals["count"]
            if n > 0:
                mean = totals["sum"] / n
                global_variance = max(global_variance, max(totals["sum_sq"] / n - mean * mean, 0.0))

        return self.fit(summaries, n_pos=n_pos, n_neg=n_neg, global_variance=global_variance)

    def predict(self, row_features: dict[str, Any]) -> tuple[int, float]:
        """Return (predicted_class, posterior_risk) for one numeric row."""
        eps = 1e-12
        log_pos = math.log(self.P_pos + eps)
        log_neg = math.log(self.P_neg + eps)

        for feature_key in self.feature_keys:
            raw_value = row_features.get(feature_key)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            for cls, log_name in ((1, "pos"), (0, "neg")):
                class_stats = self.stats.get(feature_key, {}).get(cls)
                if not class_stats:
                    continue
                mean = class_stats["mean"]
                variance = class_stats["var"]
                log_likelihood = -0.5 * (math.log(2.0 * math.pi * variance) + ((value - mean) ** 2) / variance)
                if log_name == "pos":
                    log_pos += log_likelihood
                else:
                    log_neg += log_likelihood

        max_log = max(log_pos, log_neg)
        p_pos = math.exp(log_pos - max_log)
        p_neg = math.exp(log_neg - max_log)
        risk = p_pos / (p_pos + p_neg)
        pred = 1 if risk >= self.threshold else 0
        return pred, risk

    def predict_class(self, row_features: dict[str, Any]) -> int:
        return self.predict(row_features)[0]

    def predict_risk(self, row_features: dict[str, Any]) -> float:
        return self.predict(row_features)[1]

    def predict_batch(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]


GaussianNaiveBayes = GaussianNaiveBayesModel


# ═══════════════════════════════════════════════════════════════════════════════
# BAYESIAN NETWORK CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════


class BayesianNetworkClassifierModel:
    """Discrete Bayesian-network classifier trained from conditional counts.

    The target class is implicit and acts as a parent of every feature.  The
    optional ``parent_map`` adds directed feature-to-feature dependencies, so
    prediction uses:

    ``P(class | x) ∝ P(class) × Π P(feature | class, feature_parents)``.

    The conditional-probability tables can be built from encrypted aggregate
    counts because every CPT cell is just a filtered count.
    """

    def __init__(
        self,
        parent_map: dict[str, list[str]] | None = None,
        alpha: float = 1.0,
        threshold: float = 0.5,
        backoff_to_marginal: bool = True,
    ) -> None:
        self.parent_map = {feature: list(parents) for feature, parents in (parent_map or {}).items()}
        self.alpha = alpha
        self.threshold = threshold
        self.backoff_to_marginal = backoff_to_marginal
        self.P_pos: float = 0.5
        self.P_neg: float = 0.5
        self.feature_keys: list[str] = []
        self.feature_values: dict[str, list[str]] = {}
        self.cpts: dict[str, dict[int, dict[tuple[tuple[str, str], ...], dict[str, float]]]] = {}
        self.marginals: dict[str, dict[int, dict[str, float]]] = {}
        self.default_probs: dict[str, dict[int, float]] = {}
        self.train_time: float = 0.0

    @staticmethod
    def _norm_value(value: Any) -> str:
        return str(value).lower()

    def _normalize_parent_state(
        self,
        feature_key: str,
        parent_values: dict[str, Any] | list[tuple[str, Any]] | tuple[tuple[str, Any], ...] | None,
    ) -> tuple[tuple[str, str], ...]:
        parent_order = self.parent_map.get(feature_key, [])
        if not parent_order:
            return tuple()

        if parent_values is None:
            provided: dict[str, Any] = {}
        elif isinstance(parent_values, dict):
            provided = parent_values
        else:
            provided = dict(parent_values)

        return tuple((parent, self._norm_value(provided.get(parent, ""))) for parent in parent_order)

    def _validate_graph(self, feature_keys: list[str]) -> None:
        feature_set = set(feature_keys)
        for feature, parents in self.parent_map.items():
            if feature not in feature_set:
                raise ValueError(f"parent_map contains unknown feature {feature!r}")
            unknown = [parent for parent in parents if parent not in feature_set]
            if unknown:
                raise ValueError(f"parent_map[{feature!r}] contains unknown parents: {unknown}")
            if feature in parents:
                raise ValueError(f"parent_map[{feature!r}] cannot include itself as a parent")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(feature: str) -> None:
            if feature in visited:
                return
            if feature in visiting:
                raise ValueError("parent_map must be acyclic")
            visiting.add(feature)
            for parent in self.parent_map.get(feature, []):
                visit(parent)
            visiting.remove(feature)
            visited.add(feature)

        for feature in feature_keys:
            visit(feature)

    def fit(
        self,
        cpt_counts: list[tuple[str, int, Any, str, int]],
        n_pos: int,
        n_neg: int,
        feature_values: dict[str, list[str]],
    ) -> BayesianNetworkClassifierModel:
        """Fit CPTs from conditional count tuples.

        Parameters
        ----------
        cpt_counts : list of ``(feature_key, class_label, parent_values, value, count)``
            ``parent_values`` may be a dict or tuple/list of ``(parent, value)``
            pairs matching ``parent_map[feature_key]``.
        n_pos, n_neg : class totals
        feature_values : ``{feature_key: [possible_values]}``
        """
        start = time.time()
        if not feature_values:
            raise ValueError("feature_values must contain at least one feature")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")

        self.feature_keys = list(feature_values.keys())
        self.feature_values = {
            feature_key: [self._norm_value(value) for value in values] for feature_key, values in feature_values.items()
        }
        for feature_key in self.feature_keys:
            self.parent_map.setdefault(feature_key, [])
        self._validate_graph(self.feature_keys)

        n_total = int(n_pos) + int(n_neg)
        self.P_pos = int(n_pos) / n_total if n_total > 0 else 0.5
        self.P_neg = int(n_neg) / n_total if n_total > 0 else 0.5

        counts: dict[str, dict[int, dict[tuple[tuple[str, str], ...], dict[str, int]]]] = {}
        marginal_counts: dict[str, dict[int, dict[str, int]]] = {}
        for feature_key, class_label, parent_values, raw_value, raw_count in cpt_counts:
            feature = str(feature_key)
            if feature not in self.feature_values:
                continue
            cls = int(class_label)
            if cls not in (0, 1):
                raise ValueError("BayesianNetworkClassifierModel supports binary class labels 0 and 1")
            value = self._norm_value(raw_value)
            count = int(raw_count)
            parent_state = self._normalize_parent_state(feature, parent_values)
            counts.setdefault(feature, {}).setdefault(cls, {}).setdefault(parent_state, {})[value] = (
                counts.setdefault(feature, {}).setdefault(cls, {}).setdefault(parent_state, {}).get(value, 0) + count
            )
            marginal_counts.setdefault(feature, {}).setdefault(cls, {})[value] = (
                marginal_counts.setdefault(feature, {}).setdefault(cls, {}).get(value, 0) + count
            )

        self.cpts = {feature: {1: {}, 0: {}} for feature in self.feature_keys}
        self.marginals = {feature: {1: {}, 0: {}} for feature in self.feature_keys}
        self.default_probs = {feature: {1: 0.0, 0: 0.0} for feature in self.feature_keys}

        for feature in self.feature_keys:
            values = self.feature_values[feature]
            n_values = max(1, len(values))
            for cls in (1, 0):
                class_counts = marginal_counts.get(feature, {}).get(cls, {})
                class_total = sum(class_counts.values())
                marginal_denom = class_total + self.alpha * n_values
                self.default_probs[feature][cls] = (
                    (self.alpha / marginal_denom) if marginal_denom > 0 else 1.0 / n_values
                )
                for value in values:
                    self.marginals[feature][cls][value] = (
                        (class_counts.get(value, 0) + self.alpha) / marginal_denom
                        if marginal_denom > 0
                        else 1.0 / n_values
                    )

                for parent_state, state_counts in counts.get(feature, {}).get(cls, {}).items():
                    parent_total = sum(state_counts.values())
                    denom = parent_total + self.alpha * n_values
                    self.cpts[feature][cls][parent_state] = {}
                    for value in values:
                        self.cpts[feature][cls][parent_state][value] = (
                            (state_counts.get(value, 0) + self.alpha) / denom if denom > 0 else 1.0 / n_values
                        )

        self.train_time = time.time() - start
        return self

    def fit_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_col: str,
        feature_values: dict[str, list[str]] | None = None,
    ) -> BayesianNetworkClassifierModel:
        """Fit CPTs from a plaintext categorical DataFrame."""
        if not feature_columns:
            raise ValueError("feature_columns must contain at least one feature")

        working = df.copy()
        for feature in feature_columns:
            working[feature] = working[feature].astype(str).str.lower()
        y = working[target_col].astype(int)
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("BayesianNetworkClassifierModel requires binary target labels 0 and 1")

        if feature_values is None:
            feature_values = {
                feature: sorted(working[feature].astype(str).str.lower().unique().tolist())
                for feature in feature_columns
            }
        else:
            feature_values = {
                feature: [self._norm_value(value) for value in feature_values.get(feature, [])]
                for feature in feature_columns
            }

        for feature in feature_columns:
            self.parent_map.setdefault(feature, [])
        self._validate_graph(feature_columns)

        cpt_counts = build_bayesian_cpt_counts_local(
            working,
            target_col=target_col,
            feature_values=feature_values,
            parent_map=self.parent_map,
        )

        return self.fit(
            cpt_counts,
            n_pos=int((y == 1).sum()),
            n_neg=int((y == 0).sum()),
            feature_values=feature_values,
        )

    def predict(self, row_features: dict[str, Any]) -> tuple[int, float]:
        """Return ``(predicted_class, posterior_risk)`` for one row."""
        eps = 1e-12
        log_scores = {
            1: math.log(self.P_pos + eps),
            0: math.log(self.P_neg + eps),
        }

        for feature in self.feature_keys:
            value = self._norm_value(row_features.get(feature, ""))
            parent_values = {parent: row_features.get(parent, "") for parent in self.parent_map.get(feature, [])}
            parent_state = self._normalize_parent_state(feature, parent_values)
            for cls in (1, 0):
                probs = self.cpts.get(feature, {}).get(cls, {}).get(parent_state)
                if probs is None and self.backoff_to_marginal:
                    probs = self.marginals.get(feature, {}).get(cls, {})
                prob = (probs or {}).get(value, self.default_probs.get(feature, {}).get(cls, eps))
                log_scores[cls] += math.log(max(prob, eps))

        max_log = max(log_scores.values())
        p_pos = math.exp(log_scores[1] - max_log)
        p_neg = math.exp(log_scores[0] - max_log)
        risk = p_pos / (p_pos + p_neg)
        return (1 if risk >= self.threshold else 0), risk

    def predict_class(self, row_features: dict[str, Any]) -> int:
        return self.predict(row_features)[0]

    def predict_risk(self, row_features: dict[str, Any]) -> float:
        return self.predict(row_features)[1]

    def predict_batch(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]


BayesianNetwork = BayesianNetworkClassifierModel


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION TREE  (binary CART, matches sklearn DecisionTreeClassifier)
# ═══════════════════════════════════════════════════════════════════════════════


class DecisionTreeModel:
    """Binary CART decision tree on one-hot-encoded categorical features.

    Optional *k_min* suppresses any split where either child has fewer
    than *k_min* positive samples (useful for cell-suppression policies).
    """

    def __init__(
        self,
        max_depth: int = 3,
        criterion: str = "gini",
        k_min: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.criterion = criterion
        self.k_min = k_min
        self.tree: dict | None = None
        self.col_names: list[str] = []
        self._col_set: set = set()
        self.feature_columns: list[str] = []
        self.train_time: float = 0.0

    @staticmethod
    def _norm_value(value: Any) -> str:
        return str(value).lower()

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_col: str,
    ) -> DecisionTreeModel:
        """One-hot encode *feature_columns* and build a binary tree.

        Parameters
        ----------
        df : DataFrame with feature and target columns
        feature_columns : categorical column names to split on
        target_col : binary 0/1 target column
        """
        start = time.time()
        self.feature_columns = list(feature_columns)

        X = df[feature_columns].copy()
        for col in feature_columns:
            X[col] = X[col].astype(str)
        X_encoded = pd.get_dummies(X, columns=feature_columns, drop_first=False)
        self.col_names = X_encoded.columns.tolist()
        self._col_set = set(self.col_names)

        y = df[target_col].values.astype(int)
        X_arr = X_encoded.values.astype(np.float64)

        imp_fn = gini if self.criterion == "gini" else entropy
        _k = self.k_min
        _md = self.max_depth

        def _build(indices: np.ndarray, depth: int) -> dict:
            n = len(indices)
            n_pos = int(y[indices].sum())
            n_neg = n - n_pos
            risk = n_pos / max(1, n)

            if depth >= _md or n == 0 or n_pos == 0 or n_neg == 0:
                return {"type": "leaf", "risk": risk, "n_pos": n_pos, "n_neg": n_neg, "n": n}

            base_imp = imp_fn(n_pos, n_neg)
            best_gain, best_ci = 0.0, -1
            y_sub = y[indices]
            X_sub = X_arr[indices]

            for ci in range(X_sub.shape[1]):
                left_mask = X_sub[:, ci] == 1
                left_n = int(left_mask.sum())
                right_n = n - left_n
                if left_n == 0 or right_n == 0:
                    continue
                left_pos = int((left_mask & (y_sub == 1)).sum())
                right_pos = n_pos - left_pos
                if _k > 0 and (0 < left_pos < _k or 0 < right_pos < _k):
                    continue
                wg = (left_n / n) * imp_fn(left_pos, left_n - left_pos) + (right_n / n) * imp_fn(
                    right_pos, n_neg - (left_n - left_pos)
                )
                g = base_imp - wg
                if g > best_gain:
                    best_gain, best_ci = g, ci

            if best_ci < 0:
                return {"type": "leaf", "risk": risk, "n_pos": n_pos, "n_neg": n_neg, "n": n}

            mask = X_arr[indices, best_ci] == 1
            return {
                "type": "split",
                "col_idx": best_ci,
                "col_name": self.col_names[best_ci],
                "left": _build(indices[mask], depth + 1),
                "right": _build(indices[~mask], depth + 1),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n": n,
            }

        self.tree = _build(np.arange(len(df)), 0)
        self.train_time = time.time() - start
        return self

    def fit_from_counts(
        self,
        count_fn: Callable[[tuple[tuple[str, str, bool], ...], str, str, int], int],
        feature_values: dict[str, list[str]],
        n_pos: int,
        n_neg: int,
    ) -> DecisionTreeModel:
        """Build a binary CART tree from aggregate conditional counts.

        Parameters
        ----------
        count_fn : callable
            ``count_fn(path, feature_key, value, class_label)`` must return the
            count of rows matching ``path`` AND ``feature_key == value`` AND the
            binary class label. ``path`` is a tuple of
            ``(feature_key, value, branch)`` entries where ``branch=True`` means
            the previous split took the equality/left branch and
            ``branch=False`` means it took the not-equal/right branch.
        feature_values : {feature_key: [values]}
            Candidate categorical values for one-hot CART splits.
        n_pos, n_neg : class totals at the root node.
        """
        start = time.time()
        if not feature_values:
            raise ValueError("feature_values must contain at least one feature")

        self.feature_columns = list(feature_values.keys())
        normalized_values: dict[str, list[str]] = {}
        for feature, values in feature_values.items():
            seen: set[str] = set()
            normalized_values[feature] = []
            for raw_value in values:
                value = self._norm_value(raw_value)
                if value in seen:
                    continue
                seen.add(value)
                normalized_values[feature].append(value)

        candidates: list[tuple[str, str, int, str]] = []
        self.col_names = []
        for feature in self.feature_columns:
            for value in normalized_values[feature]:
                col_name = f"{feature}_{value}"
                col_idx = len(self.col_names)
                self.col_names.append(col_name)
                candidates.append((feature, value, col_idx, col_name))
        self._col_set = set(self.col_names)

        imp_fn = gini if self.criterion == "gini" else entropy
        _k = self.k_min
        _md = self.max_depth
        count_cache: dict[tuple[tuple[tuple[str, str, bool], ...], str, str, int], int] = {}

        def _count(
            path: tuple[tuple[str, str, bool], ...],
            feature: str,
            value: str,
            cls: int,
        ) -> int:
            key = (path, feature, value, int(cls))
            if key not in count_cache:
                count_cache[key] = int(count_fn(path, feature, value, int(cls)))
            return count_cache[key]

        def _leaf(n_pos_node: int, n_neg_node: int) -> dict:
            n = n_pos_node + n_neg_node
            risk = n_pos_node / max(1, n)
            return {"type": "leaf", "risk": risk, "n_pos": n_pos_node, "n_neg": n_neg_node, "n": n}

        def _build(
            path: tuple[tuple[str, str, bool], ...],
            depth: int,
            n_pos_node: int,
            n_neg_node: int,
        ) -> dict:
            n = n_pos_node + n_neg_node
            if depth >= _md or n == 0 or n_pos_node == 0 or n_neg_node == 0:
                return _leaf(n_pos_node, n_neg_node)

            base_imp = imp_fn(n_pos_node, n_neg_node)
            best: tuple[float, str, str, int, str, int, int, int, int] | None = None

            for feature, value, col_idx, col_name in candidates:
                left_pos = _count(path, feature, value, 1)
                left_neg = _count(path, feature, value, 0)
                if left_pos < 0 or left_neg < 0:
                    raise ValueError("count_fn returned a negative count")
                if left_pos > n_pos_node or left_neg > n_neg_node:
                    raise ValueError(
                        f"count_fn returned a split count larger than the current node total for {feature}={value!r}"
                    )

                left_n = left_pos + left_neg
                right_pos = n_pos_node - left_pos
                right_neg = n_neg_node - left_neg
                right_n = right_pos + right_neg
                if left_n == 0 or right_n == 0:
                    continue
                if _k > 0 and (0 < left_pos < _k or 0 < right_pos < _k):
                    continue

                weighted_imp = (left_n / n) * imp_fn(left_pos, left_neg) + (right_n / n) * imp_fn(right_pos, right_neg)
                gain = base_imp - weighted_imp
                if best is None or gain > best[0]:
                    best = (gain, feature, value, col_idx, col_name, left_pos, left_neg, right_pos, right_neg)

            if best is None or best[0] <= 0:
                return _leaf(n_pos_node, n_neg_node)

            _gain, feature, value, col_idx, col_name, left_pos, left_neg, right_pos, right_neg = best
            left_path = path + ((feature, value, True),)
            right_path = path + ((feature, value, False),)
            return {
                "type": "split",
                "col_idx": col_idx,
                "col_name": col_name,
                "left": _build(left_path, depth + 1, left_pos, left_neg),
                "right": _build(right_path, depth + 1, right_pos, right_neg),
                "n_pos": n_pos_node,
                "n_neg": n_neg_node,
                "n": n,
                "gain": best[0],
            }

        self.tree = _build(tuple(), 0, int(n_pos), int(n_neg))
        self.train_time = time.time() - start
        return self

    def predict(self, row_dict: dict[str, Any]) -> tuple[int, float]:
        """Return (predicted_class, risk) for one row."""
        if not self.tree:
            return 0, 0.0

        active: set = set()
        for feat in self.feature_columns:
            raw_value = row_dict.get(feat, "")
            cname = f"{feat}_{raw_value}"
            cname_norm = f"{feat}_{self._norm_value(raw_value)}"
            if cname in self._col_set:
                active.add(cname)
            elif cname_norm in self._col_set:
                active.add(cname_norm)

        def _walk(node: dict) -> float:
            if node["type"] == "leaf":
                return node["risk"]
            return _walk(node["left"]) if node["col_name"] in active else _walk(node["right"])

        risk = _walk(self.tree)
        return (1 if risk >= 0.5 else 0), risk

    def predict_batch(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST  (ensemble of aggregate-count decision trees)
# ═══════════════════════════════════════════════════════════════════════════════


class RandomForestModel:
    """Random forest over aggregate-count decision trees.

    Each tree is a ``DecisionTreeModel`` trained through ``fit_from_counts`` on
    a random subset of feature keys. This keeps the model generic: it depends on
    a count provider, not on Blind Insight, DataFrames, or demo-specific fields.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 3,
        criterion: str = "gini",
        k_min: int = 0,
        max_features: int | float | str | None = "sqrt",
        random_state: int | None = None,
        threshold: float = 0.5,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.criterion = criterion
        self.k_min = k_min
        self.max_features = max_features
        self.random_state = random_state
        self.threshold = threshold
        self.estimators_: list[DecisionTreeModel] = []
        self.feature_subsets_: list[list[str]] = []
        self.feature_columns: list[str] = []
        self.feature_values: dict[str, list[str]] = {}
        self.train_time: float = 0.0

    @staticmethod
    def _norm_value(value: Any) -> str:
        return str(value).lower()

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None or mf == "all":
            return n_features
        if mf == "sqrt":
            return max(1, int(math.ceil(math.sqrt(n_features))))
        if mf == "log2":
            return max(1, int(math.ceil(math.log2(max(2, n_features)))))
        if isinstance(mf, float):
            if not 0 < mf <= 1:
                raise ValueError("float max_features must be in (0, 1]")
            return max(1, int(math.ceil(mf * n_features)))
        if isinstance(mf, int):
            if mf < 1:
                raise ValueError("integer max_features must be >= 1")
            return min(mf, n_features)
        raise ValueError("max_features must be None, 'all', 'sqrt', 'log2', int, or float")

    def fit_from_counts(
        self,
        count_fn: Callable[[tuple[tuple[str, str, bool], ...], str, str, int], int],
        feature_values: dict[str, list[str]],
        n_pos: int,
        n_neg: int,
    ) -> RandomForestModel:
        """Train an ensemble of count-backed decision trees.

        Parameters match ``DecisionTreeModel.fit_from_counts``. Randomness only
        controls feature-subset selection; all counts still come from the
        supplied aggregate count function.
        """
        start = time.time()
        if not feature_values:
            raise ValueError("feature_values must contain at least one feature")

        self.feature_columns = list(feature_values.keys())
        self.feature_values = {
            feature: [self._norm_value(value) for value in values] for feature, values in feature_values.items()
        }
        n_features = len(self.feature_columns)
        n_subset = self._resolve_max_features(n_features)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.feature_subsets_ = []

        for i in range(self.n_estimators):
            if i == 0 and n_subset < n_features:
                # Include one full-view tree so the ensemble remains stable on
                # sparse categorical demos where one feature may dominate.
                subset = list(self.feature_columns)
            elif n_subset >= n_features:
                subset = list(self.feature_columns)
                rng.shuffle(subset)
            else:
                subset = rng.choice(self.feature_columns, size=n_subset, replace=False).tolist()

            subset_values = {feature: self.feature_values[feature] for feature in subset}
            tree = DecisionTreeModel(
                max_depth=self.max_depth,
                criterion=self.criterion,
                k_min=self.k_min,
            ).fit_from_counts(
                count_fn=count_fn,
                feature_values=subset_values,
                n_pos=n_pos,
                n_neg=n_neg,
            )
            self.estimators_.append(tree)
            self.feature_subsets_.append(subset)

        self.train_time = time.time() - start
        return self

    def predict(self, row_features: dict[str, Any]) -> tuple[int, float]:
        """Return ``(predicted_class, mean_tree_risk)``."""
        if not self.estimators_:
            return 0, 0.0
        risks = [tree.predict(row_features)[1] for tree in self.estimators_]
        risk = float(sum(risks) / len(risks))
        return (1 if risk >= self.threshold else 0), risk

    def predict_batch(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]


# ═══════════════════════════════════════════════════════════════════════════════
# ADABOOST  (aggregate-count decision stumps)
# ═══════════════════════════════════════════════════════════════════════════════


class AdaBoostStumpModel:
    """AdaBoost over one-hot categorical decision stumps from aggregate counts.

    The model is generic: ``fit_from_counts`` only needs class totals, candidate
    feature values, and a count provider with the same contract as
    ``DecisionTreeModel.fit_from_counts``. Sample weights are represented as
    class-specific multipliers over regions induced by the stumps already
    selected, so no row-level data is required.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 1.0,
        k_min: int = 0,
        threshold: float = 0.5,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.k_min = int(k_min)
        self.threshold = float(threshold)
        self.stumps_: list[dict[str, Any]] = []
        self.feature_columns: list[str] = []
        self.feature_values: dict[str, list[str]] = {}
        self.col_names: list[str] = []
        self._col_set: set[str] = set()
        self.train_time: float = 0.0

    @staticmethod
    def _norm_value(value: Any) -> str:
        return str(value).lower()

    def fit_from_counts(
        self,
        count_fn: Callable[[tuple[tuple[str, str, bool], ...], str, str, int], int],
        feature_values: dict[str, list[str]],
        n_pos: int,
        n_neg: int,
    ) -> AdaBoostStumpModel:
        """Train weighted decision stumps from aggregate conditional counts."""
        start = time.time()
        if not feature_values:
            raise ValueError("feature_values must contain at least one feature")
        if n_pos < 0 or n_neg < 0:
            raise ValueError("class totals must be non-negative")
        n_total = int(n_pos) + int(n_neg)
        if n_total <= 0:
            raise ValueError("class totals must be non-zero")

        self.feature_columns = list(feature_values.keys())
        self.feature_values = {}
        candidates: list[tuple[str, str, int, str]] = []
        self.col_names = []
        for feature in self.feature_columns:
            seen: set[str] = set()
            self.feature_values[feature] = []
            for raw_value in feature_values[feature]:
                value = self._norm_value(raw_value)
                if value in seen:
                    continue
                seen.add(value)
                self.feature_values[feature].append(value)
                col_name = f"{feature}_{value}"
                candidates.append((feature, value, len(self.col_names), col_name))
                self.col_names.append(col_name)
        self._col_set = set(self.col_names)
        self.stumps_ = []

        count_cache: dict[tuple[tuple[tuple[str, str, bool], ...], str, str, int], int] = {}

        def _count(
            path: tuple[tuple[str, str, bool], ...],
            feature: str,
            value: str,
            cls: int,
        ) -> int:
            key = (path, feature, value, int(cls))
            if key not in count_cache:
                count_cache[key] = int(count_fn(path, feature, value, int(cls)))
            return count_cache[key]

        regions: list[dict[str, Any]] = [
            {
                "path": tuple(),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "w_pos": 1.0 / n_total,
                "w_neg": 1.0 / n_total,
            }
        ]
        used_candidates: set[tuple[str, str]] = set()
        eps = 1e-12

        for _round_idx in range(self.n_estimators):
            best: dict[str, Any] | None = None
            total_weight = sum(r["n_pos"] * r["w_pos"] + r["n_neg"] * r["w_neg"] for r in regions)
            if total_weight <= eps:
                break

            for feature, value, col_idx, col_name in candidates:
                if (feature, value) in used_candidates:
                    continue

                details: list[tuple[dict[str, Any], int, int, int, int]] = []
                left_pos_w = left_neg_w = right_pos_w = right_neg_w = 0.0
                left_pos_raw = left_neg_raw = right_pos_raw = right_neg_raw = 0

                for region in regions:
                    path = region["path"]
                    left_pos = _count(path, feature, value, 1)
                    left_neg = _count(path, feature, value, 0)
                    if left_pos < 0 or left_neg < 0:
                        raise ValueError("count_fn returned a negative count")
                    if left_pos > region["n_pos"] or left_neg > region["n_neg"]:
                        raise ValueError(
                            "count_fn returned a split count larger than the current region total "
                            f"for {feature}={value!r}"
                        )

                    right_pos = region["n_pos"] - left_pos
                    right_neg = region["n_neg"] - left_neg
                    details.append((region, left_pos, left_neg, right_pos, right_neg))

                    left_pos_raw += left_pos
                    left_neg_raw += left_neg
                    right_pos_raw += right_pos
                    right_neg_raw += right_neg
                    left_pos_w += region["w_pos"] * left_pos
                    left_neg_w += region["w_neg"] * left_neg
                    right_pos_w += region["w_pos"] * right_pos
                    right_neg_w += region["w_neg"] * right_neg

                left_n = left_pos_raw + left_neg_raw
                right_n = right_pos_raw + right_neg_raw
                if left_n == 0 or right_n == 0:
                    continue
                if self.k_min > 0 and (0 < left_pos_raw < self.k_min or 0 < right_pos_raw < self.k_min):
                    continue

                left_pred = 1 if left_pos_w >= left_neg_w else 0
                right_pred = 1 if right_pos_w >= right_neg_w else 0
                if left_pred == right_pred:
                    continue

                error = (left_neg_w if left_pred == 1 else left_pos_w) + (
                    right_neg_w if right_pred == 1 else right_pos_w
                )
                error /= total_weight
                if best is None or error < best["error"]:
                    best = {
                        "feature": feature,
                        "value": value,
                        "col_idx": col_idx,
                        "col_name": col_name,
                        "left_pred": left_pred,
                        "right_pred": right_pred,
                        "left_risk": left_pos_raw / max(1, left_n),
                        "right_risk": right_pos_raw / max(1, right_n),
                        "left_n": left_n,
                        "right_n": right_n,
                        "left_pos": left_pos_raw,
                        "left_neg": left_neg_raw,
                        "right_pos": right_pos_raw,
                        "right_neg": right_neg_raw,
                        "error": float(error),
                        "details": details,
                    }

            if best is None or best["error"] >= 0.5:
                break

            raw_error = best["error"]
            clipped_error = min(max(raw_error, eps), 1.0 - eps)
            alpha = self.learning_rate * 0.5 * math.log((1.0 - clipped_error) / clipped_error)
            stump = {k: v for k, v in best.items() if k != "details"}
            stump["alpha"] = float(alpha)
            self.stumps_.append(stump)
            used_candidates.add((best["feature"], best["value"]))

            new_regions: list[dict[str, Any]] = []
            for region, left_pos, left_neg, right_pos, right_neg in best["details"]:
                for branch, pred, pos_count, neg_count in (
                    (True, best["left_pred"], left_pos, left_neg),
                    (False, best["right_pred"], right_pos, right_neg),
                ):
                    if pos_count + neg_count == 0:
                        continue
                    pos_factor = math.exp(-alpha) if pred == 1 else math.exp(alpha)
                    neg_factor = math.exp(-alpha) if pred == 0 else math.exp(alpha)
                    new_regions.append(
                        {
                            "path": region["path"] + ((best["feature"], best["value"], branch),),
                            "n_pos": pos_count,
                            "n_neg": neg_count,
                            "w_pos": region["w_pos"] * pos_factor,
                            "w_neg": region["w_neg"] * neg_factor,
                        }
                    )

            norm = sum(r["n_pos"] * r["w_pos"] + r["n_neg"] * r["w_neg"] for r in new_regions)
            if norm <= eps:
                break
            for region in new_regions:
                region["w_pos"] /= norm
                region["w_neg"] /= norm
            regions = new_regions

            if raw_error <= eps:
                break

        self.train_time = time.time() - start
        return self

    def predict(self, row_features: dict[str, Any]) -> tuple[int, float]:
        """Return ``(predicted_class, boosted_risk)`` for one row."""
        if not self.stumps_:
            return 0, 0.0

        active: set[str] = set()
        for feature in self.feature_columns:
            raw_value = row_features.get(feature, "")
            cname = f"{feature}_{raw_value}"
            cname_norm = f"{feature}_{self._norm_value(raw_value)}"
            if cname in self._col_set:
                active.add(cname)
            elif cname_norm in self._col_set:
                active.add(cname_norm)

        score = 0.0
        for stump in self.stumps_:
            pred = stump["left_pred"] if stump["col_name"] in active else stump["right_pred"]
            score += stump["alpha"] * (1.0 if pred == 1 else -1.0)

        margin = 2.0 * score
        if margin >= 0:
            risk = 1.0 / (1.0 + math.exp(-margin))
        else:
            exp_margin = math.exp(margin)
            risk = exp_margin / (1.0 + exp_margin)
        return (1 if risk >= self.threshold else 0), float(risk)

    def predict_batch(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]


# ═══════════════════════════════════════════════════════════════════════════════
# HISTOGRAM CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════


class HistogramClassifierModel:
    """Categorical histogram classifier trained from aggregate marginal counts.

    The model stores one posterior risk bucket per ``feature=value``:
    ``P(positive | feature=value)``.  Prediction averages the bucket risks for
    the row's observed feature values.  Unlike Naive Bayes, this does not
    multiply conditionals or assume feature independence; each feature casts a
    direct risk vote from its encrypted count histogram.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        threshold: float | None = None,
        use_feature_weights: bool = True,
    ) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.use_feature_weights = use_feature_weights
        self.P_pos: float = 0.5
        self.P_neg: float = 0.5
        self.histograms: dict[str, dict[str, dict[str, float]]] = {}
        self.feature_keys: list[str] = []
        self.feature_weights: dict[str, float] = {}
        self.train_time: float = 0.0

    def fit(
        self,
        marginal_counts: list[tuple[str, int, str, int]],
        n_pos: int,
        n_neg: int,
        feature_values: dict[str, list[str]] | None = None,
    ) -> HistogramClassifierModel:
        """Fit lookup histograms from class-split aggregate counts.

        Parameters
        ----------
        marginal_counts : list of (feature_key, class_label, value, count)
        n_pos, n_neg : class totals
        feature_values : optional {feature_key: [values]} so zero-count buckets
                         can still receive smoothed probabilities.
        """
        start = time.time()
        n_total = n_pos + n_neg
        self.P_pos = n_pos / n_total if n_total > 0 else 0.5
        self.P_neg = n_neg / n_total if n_total > 0 else 0.5

        counts: dict[str, dict[str, dict[int, int]]] = {}
        for feature_key, class_label, raw_value, count in marginal_counts:
            value = str(raw_value).lower()
            class_counts = counts.setdefault(feature_key, {}).setdefault(value, {1: 0, 0: 0})
            class_counts[int(class_label)] = class_counts.get(int(class_label), 0) + int(count)

        if feature_values:
            for feature_key in counts:
                for raw_value in feature_values.get(feature_key, []):
                    value = str(raw_value).lower()
                    counts[feature_key].setdefault(value, {1: 0, 0: 0})

        self.feature_keys = sorted(counts.keys())
        self.histograms = {feature_key: {} for feature_key in self.feature_keys}
        self.feature_weights = {}

        for feature_key in self.feature_keys:
            feature_support = sum(
                class_counts.get(1, 0) + class_counts.get(0, 0) for class_counts in counts[feature_key].values()
            )
            weighted_discrimination = 0.0

            for value, class_counts in counts[feature_key].items():
                pos_count = class_counts.get(1, 0)
                neg_count = class_counts.get(0, 0)
                support = pos_count + neg_count
                risk = (pos_count + self.alpha) / (support + 2.0 * self.alpha)
                self.histograms[feature_key][value] = {
                    "risk": risk,
                    "support": float(support),
                    "n_pos": float(pos_count),
                    "n_neg": float(neg_count),
                }
                if feature_support > 0:
                    weighted_discrimination += (support / feature_support) * abs(risk - self.P_pos)

            if self.use_feature_weights:
                self.feature_weights[feature_key] = 1.0 + weighted_discrimination
            else:
                self.feature_weights[feature_key] = 1.0

        if self.threshold is None:
            self.threshold = self.P_pos

        self.train_time = time.time() - start
        return self

    def predict(self, row_features: dict[str, str]) -> tuple[int, float]:
        """Return (predicted_class, averaged_histogram_risk)."""
        weighted_risk = 0.0
        total_weight = 0.0

        for feature_key in self.feature_keys:
            value = str(row_features.get(feature_key, "")).lower()
            bucket = self.histograms.get(feature_key, {}).get(value)
            risk = bucket["risk"] if bucket else self.P_pos
            weight = self.feature_weights.get(feature_key, 1.0)
            weighted_risk += weight * risk
            total_weight += weight

        risk = weighted_risk / total_weight if total_weight > 0 else self.P_pos
        threshold = self.threshold if self.threshold is not None else self.P_pos
        pred = 1 if risk >= threshold else 0
        return pred, risk

    def predict_class(self, row_features: dict[str, str]) -> int:
        return self.predict(row_features)[0]

    def predict_risk(self, row_features: dict[str, str]) -> float:
        return self.predict(row_features)[1]

    def feature_importance(self) -> list[tuple[str, float]]:
        """Return features ordered by histogram discrimination weight."""
        return sorted(self.feature_weights.items(), key=lambda item: item[1], reverse=True)


HistogramClassifier = HistogramClassifierModel

# ═══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION  (OLS from aggregate counts + optional IRLS refinement)
# ═══════════════════════════════════════════════════════════════════════════════


class LogisticRegressionModel:
    """OLS / ridge logistic regression reconstructed from encrypted counts.

    Two-phase workflow:
      1. ``fit_from_counts`` -- OLS beta from marginal + pairwise counts
      2. ``refine_irls``     -- optional Newton-Raphson on local data mirror
    """

    def __init__(self, ridge_lambda: float = 0.0) -> None:
        self.ridge_lambda = ridge_lambda
        self.beta: np.ndarray | None = None
        self.dummy_index: list[tuple[str, str]] = []
        self.train_time: float = 0.0

    def fit_from_counts(
        self,
        marginals: dict[tuple[str, str], int],
        pos_counts: dict[tuple[str, str], int],
        pairwise: dict[tuple[str, str, str, str], float],
        dummy_index: list[tuple[str, str]],
        n_pos: int,
        n_neg: int,
        feat_order: list[str],
        class_weight: str | None = None,
    ) -> LogisticRegressionModel:
        """Build OLS beta = (X'WX + λI)⁻¹ X'Wy from aggregate counts.

        Parameters
        ----------
        marginals : {(feature_key, value): total_count}
        pos_counts : {(feature_key, value): positive_class_count}
        pairwise : {(feat_a, val_a, feat_b, val_b): joint_count}
        dummy_index : ordered (feature_key, value) dummy variables
        n_pos, n_neg : class totals
        feat_order : ordered feature keys (determines pairwise key direction)
        class_weight : None or ``"balanced"`` (sklearn convention)
        """
        start = time.time()
        self.dummy_index = list(dummy_index)
        n_total = n_pos + n_neg

        if class_weight == "balanced" and n_pos > 0 and n_neg > 0:
            w_pos = n_total / (2.0 * n_pos)
            w_neg = n_total / (2.0 * n_neg)
        else:
            w_pos = w_neg = 1.0

        p = len(dummy_index) + 1
        fo = {f: i for i, f in enumerate(feat_order)}

        XtWX = np.zeros((p, p))
        XtWX[0, 0] = w_pos * n_pos + w_neg * n_neg

        for i, (fi, vi) in enumerate(dummy_index):
            ci_total = marginals.get((fi, vi), 0)
            ci_pos = pos_counts.get((fi, vi), 0)
            ci_w = w_pos * ci_pos + w_neg * (ci_total - ci_pos)
            XtWX[0, i + 1] = ci_w
            XtWX[i + 1, 0] = ci_w
            XtWX[i + 1, i + 1] = ci_w

            for j in range(i + 1, len(dummy_index)):
                fj, vj = dummy_index[j]
                if fi == fj:
                    continue
                key = (fi, vi, fj, vj) if fo.get(fi, 0) < fo.get(fj, 0) else (fj, vj, fi, vi)
                cij = pairwise.get(key, 0)
                w_avg = (w_pos + w_neg) / 2.0
                XtWX[i + 1, j + 1] = cij * w_avg
                XtWX[j + 1, i + 1] = cij * w_avg

        if self.ridge_lambda > 0:
            XtWX += self.ridge_lambda * np.eye(p)

        XtWy = np.zeros(p)
        XtWy[0] = w_pos * n_pos
        for i, (fi, vi) in enumerate(dummy_index):
            XtWy[i + 1] = w_pos * pos_counts.get((fi, vi), 0)

        self.beta, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
        self.train_time = time.time() - start
        return self

    def refine_irls(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 25,
        tol: float = 1e-6,
    ) -> LogisticRegressionModel:
        """Refine beta via IRLS (Newton-Raphson for logistic regression).

        Parameters
        ----------
        X : (n, p) design matrix with intercept in column 0
        y : (n,) binary target
        """
        if self.beta is None:
            raise ValueError("Call fit_from_counts first")

        start = time.time()
        p = X.shape[1]
        beta = self.beta.copy()

        for _ in range(max_iter):
            z = np.clip(X @ beta, -500, 500)
            mu = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-10, 1 - 1e-10)
            w = mu * (1.0 - mu)
            wr = z + (y - mu) / w

            XtWX = (X.T * w) @ X
            if self.ridge_lambda > 0:
                XtWX += self.ridge_lambda * np.eye(p)

            beta_new = np.linalg.solve(XtWX, (X.T * w) @ wr)
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        self.beta = beta
        self.train_time += time.time() - start
        return self

    def predict(
        self,
        row_features: dict[str, str],
        use_sigmoid: bool = True,
    ) -> float:
        """Return P(positive) for a single row.

        Parameters
        ----------
        row_features : {feature_key: value} matching dummy_index keys.
        """
        if self.beta is None:
            return 0.0

        x = np.zeros(len(self.dummy_index) + 1)
        x[0] = 1.0
        for i, (fk, val) in enumerate(self.dummy_index):
            if str(row_features.get(fk, "")).lower() == val:
                x[i + 1] = 1.0

        z = float(x @ self.beta)
        if use_sigmoid:
            z = max(-500, min(500, z))
            return 1.0 / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, z))


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def build_marginals_local(
    df: pd.DataFrame,
    target_col: str,
    feature_config: list[tuple[str, str, list[str]]],
) -> list[tuple[str, int, str, int]]:
    """Build NB-format marginal counts from a local DataFrame.

    Parameters
    ----------
    target_col : binary 0/1 column name
    feature_config : list of ``(feature_key, column_name, values)``
        *feature_key* appears in the output tuples,
        *column_name* is the actual DataFrame column,
        *values* is the list of possible string values.
    """
    results: list[tuple[str, int, str, int]] = []
    y = df[target_col].values.astype(int)

    for fk, col, vals in feature_config:
        series = df[col].astype(str).str.lower()
        for v in vals:
            mask = (series == v.lower()).values
            results.append((fk, 1, v, int((mask & (y == 1)).sum())))
            results.append((fk, 0, v, int((mask & (y == 0)).sum())))
    return results


def extract_marginals(
    raw_results: list[tuple[str, int, str, int]],
) -> dict[tuple[str, str], int]:
    """Sum class-split counts into totals per (feature_key, value)."""
    totals: dict[tuple[str, str], int] = {}
    for fk, _cls, val, count in raw_results:
        key = (fk, val.lower())
        totals[key] = totals.get(key, 0) + count
    return totals


def extract_pos_counts(
    raw_results: list[tuple[str, int, str, int]],
) -> dict[tuple[str, str], int]:
    """Extract positive-class counts per (feature_key, value)."""
    counts: dict[tuple[str, str], int] = {}
    for fk, cls, val, count in raw_results:
        if cls == 1:
            counts[(fk, val.lower())] = count
    return counts


def compute_pairwise_local(
    df: pd.DataFrame,
    feature_columns: list[str],
    feature_values: dict[str, list[str]],
    marginals: dict[tuple[str, str], int] | None = None,
    n_total: int | None = None,
    min_cell_size: int = 0,
) -> dict[str, Any]:
    """Compute pairwise cross-tabulation counts from local data.

    Parameters
    ----------
    feature_columns : ordered column names (also used as feature keys)
    feature_values : {column_name: [possible_values]}
    min_cell_size : cells with ``0 < count < min_cell_size`` are replaced
                    by independence estimates ``(marginal_a * marginal_b) / n``.
    """
    pw: dict[tuple[str, str, str, str], float] = {}
    n_suppressed = 0
    suppressed_cells: list[str] = []

    for i, fa in enumerate(feature_columns):
        for fb in feature_columns[i + 1 :]:
            ct = pd.crosstab(df[fa], df[fb])
            for va in ct.index:
                for vb in ct.columns:
                    count: float = int(ct.loc[va, vb])
                    if min_cell_size > 0 and 0 < count < min_cell_size:
                        if marginals and n_total:
                            ca = marginals.get((fa, str(va)), 0)
                            cb = marginals.get((fb, str(vb)), 0)
                            count = (ca * cb / n_total) if n_total > 0 else 0
                        n_suppressed += 1
                        suppressed_cells.append(f"{fa}={va} x {fb}={vb}")
                    pw[(fa, str(va), fb, str(vb))] = count

    return {"pairwise": pw, "n_suppressed": n_suppressed, "suppressed_cells": suppressed_cells}


def build_dummy_index(
    feature_columns: list[str],
    feature_values: dict[str, list[str]],
    reference: dict[str, str] | None = None,
    drop: str = "last",
) -> list[tuple[str, str]]:
    """Build ordered dummy-variable index for linear models.

    Parameters
    ----------
    drop : ``"last"`` drops the last value per feature;
           ``"reference"`` drops the value specified in *reference*.
    """
    index: list[tuple[str, str]] = []
    for fk in feature_columns:
        vals = feature_values.get(fk, [])
        for v in vals:
            v_low = v.lower()
            if drop == "last" and v == vals[-1]:
                continue
            if drop == "reference" and reference and v_low == reference.get(fk, "").lower():
                continue
            index.append((fk, v_low))
    return index


def build_design_matrix(
    df: pd.DataFrame,
    dummy_index: list[tuple[str, str]],
    col_map: dict[str, str] | None = None,
) -> np.ndarray:
    """Build ``(n, p)`` design matrix with intercept in column 0.

    Parameters
    ----------
    col_map : optional ``{feature_key: column_name}``; if *None*,
              feature_key is used as the column name directly.
    """
    n = len(df)
    p = len(dummy_index) + 1
    X = np.zeros((n, p))
    X[:, 0] = 1.0

    feature_keys = sorted(set(fk for fk, _ in dummy_index))
    _cm = col_map or {}

    for fk in feature_keys:
        col = _cm.get(fk, fk)
        series = df[col].astype(str).str.lower()
        for i, (di_fk, di_val) in enumerate(dummy_index):
            if di_fk == fk:
                X[:, i + 1] = (series == di_val).astype(float).values
    return X


def platt_scale(
    risk_scores: list[float],
    y_true: list[int],
) -> tuple[float, float]:
    """Fit Platt scaling: ``P(y=1|s) = 1/(1+exp(-(a*s+b)))``.

    Returns ``(a, b)`` coefficients.
    """
    from scipy.optimize import minimize

    scores = np.array(risk_scores, dtype=np.float64)
    labels = np.array(y_true, dtype=np.float64)

    def _nll(params):
        a, b = params
        z = np.clip(a * scores + b, -500, 500)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-10, 1 - 1e-10)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    result = minimize(_nll, [1.0, 0.0], method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def apply_platt(risk: float, a: float, b: float) -> float:
    """Apply Platt scaling to a single risk score."""
    z = max(-500, min(500, a * risk + b))
    return 1.0 / (1.0 + math.exp(-z))


def build_bayesian_cpt_counts_local(
    df: pd.DataFrame,
    target_col: str,
    feature_values: dict[str, list[str]],
    parent_map: dict[str, list[str]] | None = None,
    col_map: dict[str, str] | None = None,
) -> list[tuple[str, int, tuple[tuple[str, str], ...], str, int]]:
    """Build Bayesian-network CPT counts from a local categorical DataFrame.

    Returns ``(feature_key, class_label, parent_state, value, count)`` tuples
    compatible with ``BayesianNetworkClassifierModel.fit``.  The output includes
    explicit zero-count CPT cells so plaintext and encrypted aggregate workflows
    have the same table shape.
    """
    parent_map = {feature: list(parents) for feature, parents in (parent_map or {}).items()}
    col_map = col_map or {}
    values = {
        feature: [str(value).lower() for value in feature_possible_values]
        for feature, feature_possible_values in feature_values.items()
    }

    working = df.copy()
    for feature in values:
        column = col_map.get(feature, feature)
        working[column] = working[column].astype(str).str.lower()
    working[target_col] = working[target_col].astype(int)

    results: list[tuple[str, int, tuple[tuple[str, str], ...], str, int]] = []
    for feature, feature_possible_values in values.items():
        parents = parent_map.get(feature, [])
        feature_column = col_map.get(feature, feature)
        group_columns = [target_col, *(col_map.get(parent, parent) for parent in parents), feature_column]
        grouped = working.groupby(group_columns, dropna=False).size().to_dict()
        parent_value_lists = [values[parent] for parent in parents]
        parent_combos = list(product(*parent_value_lists)) if parent_value_lists else [tuple()]

        for class_label in (1, 0):
            for parent_combo in parent_combos:
                parent_state = tuple((parent, parent_value) for parent, parent_value in zip(parents, parent_combo))
                for value in feature_possible_values:
                    group_key = (class_label, *parent_combo, value)
                    count = int(grouped.get(group_key, 0))
                    results.append((feature, class_label, parent_state, value, count))
    return results
