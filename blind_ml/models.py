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

    def fit_with_bi_root(
        self,
        raw_results: list[tuple[str, int, str, int]],
        feat_type_to_column: dict[str, str],
        df: pd.DataFrame,
        feature_columns: list[str],
        target_col: str,
        n_pos: int,
        n_neg: int,
    ) -> DecisionTreeModel:
        """Build a decision tree whose ROOT split is chosen from BI marginal counts.

        The root one-hot dummy column is selected by computing Gini gain across
        every (feature, value) pair in ``raw_results`` using only the encrypted
        aggregate counts. Deeper splits are computed from the local DataFrame
        mirror — see APPROACH.md, "root split from BI marginals, deeper splits
        from local cross-tabs."

        Parameters
        ----------
        raw_results : list of ``(feat_type, class_label, value, count)`` tuples
            as returned by ``run_bi_training`` — the encrypted aggregate counts.
        feat_type_to_column : maps the ``feat_type`` strings used in raw_results
            (e.g. ``"fraud"``) to the actual DataFrame column names (e.g.
            ``"fraud_type"``).
        df : local plaintext mirror; used only for deeper splits.
        feature_columns : feature columns to one-hot encode (in df).
        target_col : binary 0/1 target column.
        n_pos, n_neg : class totals from BI (``get_bi_base_rates``). The root
            split's Gini gain is computed against these totals, not the local
            DataFrame's class counts.
        """
        start = time.time()
        self.feature_columns = list(feature_columns)

        # One-hot encode the local mirror — same encoding ``fit`` uses so
        # predict() and column names stay consistent.
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

        # ── Step 1: pick the ROOT split from BI marginal counts ────────────
        # raw_results contains, for each (feat_type, class, value) triple, the
        # encrypted aggregate count. Build a lookup: counts[col_name][class] = count.
        # col_name follows pd.get_dummies' convention: f"{column}_{value}".
        counts: dict[str, dict[int, int]] = {}
        for feat_type, cls, val, count in raw_results:
            col = feat_type_to_column.get(feat_type)
            if col is None:
                continue
            col_name = f"{col}_{val}"
            counts.setdefault(col_name, {0: 0, 1: 0})[int(cls)] = int(count)

        base_imp_root = imp_fn(n_pos, n_neg)
        best_gain_bi = 0.0
        best_col_name: str | None = None
        for col_name, by_class in counts.items():
            if col_name not in self._col_set:
                # raw_results references a value not present in local one-hot
                # columns — skip rather than fabricate a split direction.
                continue
            left_pos = by_class.get(1, 0)
            left_neg = by_class.get(0, 0)
            left_n = left_pos + left_neg
            right_pos = n_pos - left_pos
            right_neg = n_neg - left_neg
            right_n = right_pos + right_neg
            if left_n == 0 or right_n == 0:
                continue
            if _k > 0 and (0 < left_pos < _k or 0 < right_pos < _k):
                continue
            n_total = left_n + right_n
            wg = (left_n / n_total) * imp_fn(left_pos, left_neg) + (right_n / n_total) * imp_fn(right_pos, right_neg)
            g = base_imp_root - wg
            if g > best_gain_bi:
                best_gain_bi = g
                best_col_name = col_name

        if best_col_name is None:
            # No usable BI split — fall back to local fit so the demo still
            # produces a tree, but flag it on the tree dict for transparency.
            self.fit(df, feature_columns, target_col)
            if self.tree is not None:
                self.tree["bi_root"] = False
            self.train_time = time.time() - start
            return self

        best_ci = self.col_names.index(best_col_name)

        # ── Step 2: apply the BI-chosen split to the local data ────────────
        all_indices = np.arange(len(df))
        mask = X_arr[all_indices, best_ci] == 1

        # ── Step 3: build child subtrees from local data (depths 1..max) ───
        def _build(indices: np.ndarray, depth: int) -> dict:
            n = len(indices)
            n_pos_node = int(y[indices].sum())
            n_neg_node = n - n_pos_node
            risk = n_pos_node / max(1, n)
            if depth >= _md or n == 0 or n_pos_node == 0 or n_neg_node == 0:
                return {
                    "type": "leaf",
                    "risk": risk,
                    "n_pos": n_pos_node,
                    "n_neg": n_neg_node,
                    "n": n,
                }
            base_imp = imp_fn(n_pos_node, n_neg_node)
            best_gain, best_ci_child = 0.0, -1
            y_sub = y[indices]
            X_sub = X_arr[indices]
            for ci in range(X_sub.shape[1]):
                left_mask = X_sub[:, ci] == 1
                left_n_c = int(left_mask.sum())
                right_n_c = n - left_n_c
                if left_n_c == 0 or right_n_c == 0:
                    continue
                left_pos_c = int((left_mask & (y_sub == 1)).sum())
                right_pos_c = n_pos_node - left_pos_c
                if _k > 0 and (0 < left_pos_c < _k or 0 < right_pos_c < _k):
                    continue
                wg_c = (left_n_c / n) * imp_fn(left_pos_c, left_n_c - left_pos_c) + (right_n_c / n) * imp_fn(
                    right_pos_c, n_neg_node - (left_n_c - left_pos_c)
                )
                g_c = base_imp - wg_c
                if g_c > best_gain:
                    best_gain, best_ci_child = g_c, ci
            if best_ci_child < 0:
                return {
                    "type": "leaf",
                    "risk": risk,
                    "n_pos": n_pos_node,
                    "n_neg": n_neg_node,
                    "n": n,
                }
            mask_c = X_arr[indices, best_ci_child] == 1
            return {
                "type": "split",
                "col_idx": best_ci_child,
                "col_name": self.col_names[best_ci_child],
                "left": _build(indices[mask_c], depth + 1),
                "right": _build(indices[~mask_c], depth + 1),
                "n_pos": n_pos_node,
                "n_neg": n_neg_node,
                "n": n,
            }

        # Use BI base rates (n_pos, n_neg) on the root node's counts so the
        # tree's reported totals reflect the encrypted dataset, not local.
        self.tree = {
            "type": "split",
            "col_idx": best_ci,
            "col_name": best_col_name,
            "left": _build(all_indices[mask], 1),
            "right": _build(all_indices[~mask], 1),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n": n_pos + n_neg,
            "bi_root": True,
            "bi_root_gain": best_gain_bi,
        }
        self.train_time = time.time() - start
        return self

    def predict(self, row_dict: dict[str, Any]) -> tuple[int, float]:
        """Return (predicted_class, risk) for one row."""
        if not self.tree:
            return 0, 0.0

        active: set = set()
        for feat in self.feature_columns:
            cname = f"{feat}_{row_dict.get(feat, '')}"
            if cname in self._col_set:
                active.add(cname)

        def _walk(node: dict) -> float:
            if node["type"] == "leaf":
                return node["risk"]
            return _walk(node["left"]) if node["col_name"] in active else _walk(node["right"])

        risk = _walk(self.tree)
        return (1 if risk >= 0.5 else 0), risk

    def predict_batch(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]


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
