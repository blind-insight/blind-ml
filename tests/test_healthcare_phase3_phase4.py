from __future__ import annotations

import pandas as pd

from blind_ml.healthcare import (
    _BC_BN_PARENT_MAP,
    _bc_bn_cpt_count_queries,
    _bc_bn_feature_values,
    _build_bc_dt_count_provider,
    bc_bn_predict,
    bc_dt_predict,
    bc_plaintext_bn_predict_proba,
    run_encrypted_bn_bc,
    run_encrypted_dt_bc,
    train_plaintext_bn_bc,
)


class MockBIClient:
    def __init__(self, counts: dict[tuple[str, ...], int]):
        self.counts = counts
        self.aggregate_counts: dict[tuple[str, tuple[str, ...]], int] = {}
        self.calls: list[tuple[str, ...]] = []
        self.aggregate_calls: list[tuple[str, tuple[str, ...]]] = []

    def query(self, **kwargs):
        filters = tuple(kwargs.get("filters", []))
        self.calls.append(filters)
        return {"count": self.counts.get(filters, 0)}

    def aggregate(self, **kwargs):
        agg_filter = kwargs.get("agg_filter", "")
        extra_filters = tuple(kwargs.get("extra_filters", []))
        self.aggregate_calls.append((agg_filter, extra_filters))
        value = self.aggregate_counts.get((agg_filter, extra_filters), 0)
        return {"records": [{"data": {"value": value}}]}


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


def _key(filters: list[str]) -> tuple[str, ...]:
    return tuple(sorted(filters))


def test_bc_bn_parent_map_keys_match_feature_values():
    values = _bc_bn_feature_values(_mini_feature_values())

    assert set(_BC_BN_PARENT_MAP) == set(values)
    for feature, parents in _BC_BN_PARENT_MAP.items():
        assert feature in values
        assert all(parent in values for parent in parents)


def test_bc_bn_cpt_queries_include_parent_combinations_and_binned_filters():
    queries = _bc_bn_cpt_count_queries(_mini_feature_values())

    atypical = [q for q in queries if q[0] == "atypical" and q[1] == 1 and q[3] == "yes"]
    assert any(q[2] == (("biopsies", "3_plus"),) for q in atypical)
    assert any("num_prior_biopsies:3~7" in q[4] for q in atypical)

    afb = [q for q in queries if q[0] == "afb" and q[1] == 0 and q[3] == "30_plus"]
    assert any(q[2] == (("age_group", "50_59"),) for q in afb)
    assert any("age_at_first_birth:30~47" in q[4] for q in afb)


def test_run_encrypted_bn_bc_trains_from_mocked_aggregate_counts_and_suppresses_cells():
    values = _mini_feature_values()
    counts = {
        ("cancer_5yr:1",): 40,
        ("cancer_5yr:0",): 60,
        ("cancer_5yr:1", "age_group:40_49"): 3,
        ("cancer_5yr:0", "age_group:40_49"): 20,
    }
    client = MockBIClient(counts)

    result = run_encrypted_bn_bc(
        client,
        "org",
        "dataset",
        "schema",
        values,
        max_workers=1,
    )

    assert result["n_cancer"] == 40
    assert result["n_no_cancer"] == 60
    assert result["n_suppressed"] == 1
    assert ("age_group", 1, tuple(), "40_49", 5) in result["raw_results"]
    assert ("age_group", 1, tuple(), "40_49", 3) in result["raw_results_unsuppressed"]
    assert result["enc_queries"] == result["cpt_queries"] + 2

    pred, risk = bc_bn_predict(
        result,
        {
            "age_group": "40_49",
            "race_ethnicity": "white",
            "num_first_degree_relatives": 0,
            "num_prior_biopsies": 0,
            "atypical_hyperplasia": "no",
            "menarche_category": "12_13",
            "breast_density": 1,
            "age_at_first_birth": 0,
        },
    )
    assert pred in (0, 1)
    assert 0.0 <= risk <= 1.0


def test_plaintext_bn_bc_benchmark_helpers():
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

    model = train_plaintext_bn_bc(df, _mini_feature_values())
    probs = bc_plaintext_bn_predict_proba(model, df)

    assert model["n_cancer"] == 2
    assert model["n_no_cancer"] == 2
    assert len(probs) == len(df)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_bc_dt_count_provider_equality_and_not_equal_paths():
    values = _mini_feature_values()
    counts = {
        ("cancer_5yr:1", "race_ethnicity:white"): 30,
        ("cancer_5yr:1", "age_group:40_49", "race_ethnicity:white"): 7,
    }
    client = MockBIClient(counts)
    count_fn, query_count, _, query_cache, fallback_count = _build_bc_dt_count_provider(
        client,
        "org",
        "dataset",
        "schema",
        values,
        raw_results=None,
        n_cancer=30,
        n_no_cancer=0,
    )

    total = count_fn(tuple(), "race", "white", 1)
    not_age_40 = count_fn((("age_group", "40_49", False),), "race", "white", 1)

    assert total == 30
    assert not_age_40 == 23
    assert query_count() == 3
    assert fallback_count() == 0
    assert _key(["cancer_5yr:1", "race_ethnicity:white"]) in query_cache
    assert client.calls == [
        ("cancer_5yr:1", "race_ethnicity:white"),
        ("cancer_5yr:1", "age_group:40_49", "race_ethnicity:white"),
        ("cancer_5yr:1", "age_group:40_49"),
    ]


def test_bc_dt_count_provider_falls_back_to_class_aggregate_for_deep_paths():
    class FailingDeepPathClient(MockBIClient):
        def query(self, **kwargs):
            filters = tuple(kwargs.get("filters", []))
            self.calls.append(filters)
            if filters == (
                "cancer_5yr:1",
                "age_group:40_49",
                "num_first_degree_relatives:0",
                "num_prior_biopsies:0",
            ):
                raise RuntimeError("422")
            return {"count": self.counts.get(filters, 0)}

    values = _mini_feature_values()
    client = FailingDeepPathClient({
        ("cancer_5yr:1", "age_group:40_49"): 13,
        ("cancer_5yr:1", "age_group:40_49", "num_first_degree_relatives:0"): 13,
    })
    client.aggregate_counts[
        (
            "cancer_5yr:count(1)",
            ("age_group:40_49", "num_first_degree_relatives:0", "num_prior_biopsies:0"),
        )
    ] = 13
    count_fn, query_count, _, _, fallback_count = _build_bc_dt_count_provider(
        client,
        "org",
        "dataset",
        "schema",
        values,
        raw_results=None,
        n_cancer=100,
        n_no_cancer=100,
    )

    count = count_fn(
        (("age_group", "40_49", True), ("relatives", "0", True)),
        "biopsies",
        "0",
        1,
    )

    assert count == 13
    assert query_count() == 3
    assert fallback_count() == 1
    assert client.aggregate_calls == [
        (
            "cancer_5yr:count(1)",
            ("age_group:40_49", "num_first_degree_relatives:0", "num_prior_biopsies:0"),
        )
    ]


def test_bc_dt_count_provider_caps_deeper_counts_to_path_bound():
    values = _mini_feature_values()
    values["relatives_values"] = ["0", "1", "5"]
    counts = {
        ("cancer_5yr:1", "num_first_degree_relatives:5"): 10,
        ("cancer_5yr:1", "age_group:40_49", "num_first_degree_relatives:5"): 2,
    }
    client = MockBIClient(counts)
    count_fn, _, _, _, _ = _build_bc_dt_count_provider(
        client,
        "org",
        "dataset",
        "schema",
        values,
        raw_results=[("age_group", 1, "40_49", 95)],
        n_cancer=100,
        n_no_cancer=100,
    )

    count = count_fn((("age_group", "40_49", False),), "relatives", "5", 1)

    assert count == 5


def test_run_encrypted_dt_bc_trains_from_aggregate_counts_without_df_local():
    values = _mini_feature_values()
    raw_results = [
        ("age_group", 1, "40_49", 20),
        ("age_group", 0, "40_49", 80),
        ("age_group", 1, "50_59", 80),
        ("age_group", 0, "50_59", 20),
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

    result = run_encrypted_dt_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        values,
        raw_results=raw_results,
        n_cancer=100,
        n_no_cancer=100,
        max_depth=1,
        k_min=11,
    )

    assert result["counts_from_bi"] is True
    assert result["root_from_bi"] is True
    assert result["additional_dt_queries"] == 0
    assert result["tree"]["type"] == "split"
    pred, risk = bc_dt_predict(
        result,
        {
            "age_group": "50_59",
            "race_ethnicity": "white",
            "num_first_degree_relatives": 0,
            "num_prior_biopsies": 0,
            "atypical_hyperplasia": "no",
            "menarche_category": "12_13",
            "breast_density": 1,
            "age_at_first_birth": 0,
        },
    )
    assert pred in (0, 1)
    assert 0.0 <= risk <= 1.0


def test_run_encrypted_dt_bc_k_min_blocks_small_positive_split():
    values = _mini_feature_values()
    raw_results = [
        ("age_group", 1, "40_49", 5),
        ("age_group", 0, "40_49", 90),
        ("age_group", 1, "50_59", 95),
        ("age_group", 0, "50_59", 10),
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

    result = run_encrypted_dt_bc(
        MockBIClient({}),
        "org",
        "dataset",
        "schema",
        values,
        raw_results=raw_results,
        n_cancer=100,
        n_no_cancer=100,
        max_depth=1,
        k_min=11,
    )

    assert result["tree"]["type"] == "leaf"
    assert result["min_cell_size"] == 11
