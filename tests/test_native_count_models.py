import math

import numpy as np

from blind_ml import (
    DecisionTreeModel,
    EntropyDecisionTreeModel,
    GradientBoostedTreesModel,
    RidgeRegressionModel,
)

ROWS = [
    {"color": "red", "shape": "round", "y": 1},
    {"color": "red", "shape": "square", "y": 1},
    {"color": "red", "shape": "round", "y": 1},
    {"color": "blue", "shape": "round", "y": 0},
    {"color": "blue", "shape": "square", "y": 0},
    {"color": "blue", "shape": "square", "y": 0},
]


def _matches_path(row, path):
    for feature, value, branch in path:
        if (str(row[feature]).lower() == value) != branch:
            return False
    return True


def _split_count_fn(path, feature, value, cls):
    return sum(
        1
        for row in ROWS
        if row["y"] == cls and _matches_path(row, path) and str(row[feature]).lower() == value
    )


def _path_count_fn(path, cls):
    return sum(1 for row in ROWS if row["y"] == cls and _matches_path(row, path))


def test_entropy_decision_tree_trains_from_counts():
    tree = EntropyDecisionTreeModel(max_depth=2).fit_from_counts(
        count_fn=_split_count_fn,
        feature_values={"color": ["red", "blue"], "shape": ["round", "square"]},
        n_pos=3,
        n_neg=3,
    )

    assert tree.criterion == "entropy"
    assert tree.tree["type"] == "split"
    assert tree.tree["col_name"] == "color_red"
    assert tree.predict({"color": "red", "shape": "round"})[0] == 1
    assert tree.predict({"color": "blue", "shape": "round"})[0] == 0


def test_decision_tree_rejects_unknown_criterion():
    try:
        DecisionTreeModel(criterion="gain_ratio").fit_from_counts(
            count_fn=_split_count_fn,
            feature_values={"color": ["red", "blue"]},
            n_pos=3,
            n_neg=3,
        )
    except ValueError as exc:
        assert "criterion" in str(exc)
    else:
        raise AssertionError("unknown criterion should fail")


def test_ridge_matches_manual_count_solution_and_tunes_lambda():
    marginals = {("color", "red"): 3, ("color", "blue"): 3}
    pos_counts = {("color", "red"): 3, ("color", "blue"): 0}
    pairwise = {}
    dummy_index = [("color", "red")]

    model = RidgeRegressionModel(ridge_lambda=1.0).fit_from_counts(
        marginals=marginals,
        target_counts=pos_counts,
        pairwise=pairwise,
        dummy_index=dummy_index,
        n_pos=3,
        n_neg=3,
        feat_order=["color"],
    )

    xtx = np.array([[6.0, 3.0], [3.0, 3.0]])
    xty = np.array([3.0, 3.0])
    penalty = np.array([[0.0, 0.0], [0.0, 1.0]])
    expected = np.linalg.solve(xtx + penalty, xty)
    assert np.allclose(model.beta, expected)
    assert model.predict({"color": "red"}) > model.predict({"color": "blue"})

    def holdout_count_fn(state, cls):
        return sum(
            1
            for row in ROWS
            if row["y"] == cls and all(str(row[feature]).lower() == value for feature, value in state)
        )

    tuned = RidgeRegressionModel().tune_lambda_from_counts(
        lambda_grid=[0.0, 1.0, 10.0],
        holdout_count_fn=holdout_count_fn,
        feature_values={"color": ["red", "blue"]},
        marginals=marginals,
        target_counts=pos_counts,
        pairwise=pairwise,
        dummy_index=dummy_index,
        n_pos=3,
        n_neg=3,
        feat_order=["color"],
    )
    assert math.isclose(tuned.ridge_lambda, 0.0)


def test_gradient_boosted_trees_train_from_path_counts():
    model = GradientBoostedTreesModel(
        n_estimators=2,
        learning_rate=1.0,
        max_depth=1,
        max_score_regions=16,
    ).fit_from_counts(
        path_count_fn=_path_count_fn,
        feature_values={"color": ["red", "blue"], "shape": ["round", "square"]},
        n_pos=3,
        n_neg=3,
    )

    assert model.trees_
    assert model.predict({"color": "red", "shape": "round"})[1] > 0.5
    assert model.predict({"color": "blue", "shape": "round"})[1] < 0.5

    adapted = GradientBoostedTreesModel.path_count_from_split_count_fn(_split_count_fn, n_pos=3, n_neg=3)
    assert adapted((("color", "red", False),), 0) == 3
