from sara_engine.nn.local_manifold_memory import (
    LocalManifoldTransitionMemory,
    evaluate_local_manifold_transition_memory,
    nearest_trajectories,
    overlap_ratio,
    predict_next_ids_from_nearest,
)


def test_local_manifold_memory_selects_sparse_nearest_trajectory() -> None:
    trajectory_graph = [
        {
            "trajectory_id": "stable-path",
            "source_ids": [1, 2, 3],
            "next_ids": [10, 11],
            "causal_edges": [{"from": "a", "to": "b", "support": 0.9}],
            "event_cost_proxy": 0.2,
        },
        {
            "trajectory_id": "near-distractor",
            "source_ids": [1, 2, 9],
            "next_ids": [90],
            "causal_edges": [{"from": "a", "to": "risk", "support": 0.7}],
            "event_cost_proxy": 0.1,
        },
    ]

    nearest = nearest_trajectories([1, 2, 3], trajectory_graph, limit=1)
    predicted = predict_next_ids_from_nearest(nearest, trajectory_graph)

    assert overlap_ratio([1, 2, 3], [1, 2, 3]) == 1.0
    assert nearest[0]["trajectory_id"] == "stable-path"
    assert nearest[0]["scanned_trajectory_count"] == 2
    assert predicted == [10, 11]


def test_local_manifold_memory_evaluates_case_coverage_without_dense_search() -> None:
    trajectory_graph = [
        {
            "trajectory_id": "path-a",
            "source_ids": [1, 2, 3],
            "next_ids": [10, 11],
            "causal_edges": [{"from": "a", "to": "b", "support": 0.9}],
            "event_cost_proxy": 0.2,
        },
        {
            "trajectory_id": "path-b",
            "source_ids": [4, 5, 6],
            "next_ids": [20, 21],
            "causal_edges": [{"from": "b", "to": "c", "support": 0.88}],
            "event_cost_proxy": 0.18,
        },
    ]

    report = evaluate_local_manifold_transition_memory(
        query_ids=[1, 2, 3],
        withheld_expected_ids=[10, 11],
        trajectory_graph=trajectory_graph,
        case_specs=[
            {
                "case_id": "case-a",
                "query_ids": [1, 2, 3],
                "expected_trajectory_id": "path-a",
                "expected_next_ids": [10, 11],
            },
            {
                "case_id": "case-b",
                "query_ids": [4, 5, 6],
                "expected_trajectory_id": "path-b",
                "expected_next_ids": [20, 21],
            },
        ],
    )

    assert report["observed_only"] is True
    assert report["trajectory_top_match_ratio"] == 1.0
    assert report["average_case_recall"] == 1.0
    assert report["metrics"]["manifold_trajectory_case_coverage"] == 1.0
    assert report["metrics"]["manifold_average_case_recall"] == 1.0
    assert report["metrics"]["manifold_scan_budget_integrity"] == 1.0
    assert report["metrics"]["manifold_indexed_candidate_integrity"] == 1.0
    assert report["metrics"]["manifold_index_scan_reduction"] == 1.0
    assert report["max_scanned_trajectory_count"] == 1
    assert report["dense_scan_baseline_count"] == 2
    assert report["indexed_scan_reduction_ratio"] == 0.5
    assert report["source_event_index_size"] == 6
    assert report["indexed_candidate_case_ratio"] == 1.0
    assert all(case["scan_budget_ok"] for case in report["case_results"])
    assert all(case["indexed_candidate_ok"] for case in report["case_results"])


def test_local_manifold_transition_memory_supports_bounded_continual_updates() -> None:
    memory = LocalManifoldTransitionMemory(capacity=2)
    memory.add_trajectory(
        "older-path",
        source_events=[7, 8],
        next_events=[70],
        causal_edges=[{"from": "old", "to": "path", "support": 0.9}],
        event_cost_proxy=0.3,
    )
    memory.add_trajectory(
        "stable-path",
        source_events=[1, 2, 3],
        next_events=[10, 11],
        causal_edges=[{"from": "a", "to": "b", "support": 0.92}],
        event_cost_proxy=0.2,
    )
    memory.add_trajectory(
        "new-path",
        source_events=[4, 5, 6],
        next_events=[20, 21],
        causal_edges=[{"from": "b", "to": "c", "support": 0.88}],
        event_cost_proxy=0.18,
    )

    graph = memory.trajectory_graph()
    nearest = memory.query([1, 2, 3], limit=1)
    report = memory.evaluate(
        query_events=[1, 2, 3],
        withheld_expected_events=[10, 11],
        case_specs=[
            {
                "case_id": "stable",
                "query_events": [1, 2, 3],
                "expected_trajectory_id": "stable-path",
                "expected_next_events": [10, 11],
            },
            {
                "case_id": "new",
                "query_events": [4, 5, 6],
                "expected_trajectory_id": "new-path",
                "expected_next_events": [20, 21],
            },
        ],
    )

    assert [item["trajectory_id"] for item in graph] == ["stable-path", "new-path"]
    assert nearest[0]["trajectory_id"] == "stable-path"
    assert memory.predict_next_ids([4, 5, 6], limit=1) == [20, 21]
    assert memory.source_event_index() == {1: ["stable-path"], 2: ["stable-path"], 3: ["stable-path"], 4: ["new-path"], 5: ["new-path"], 6: ["new-path"]}
    assert report["trajectory_top_match_ratio"] == 1.0
    assert report["metrics"]["manifold_average_case_recall"] == 1.0
    assert report["metrics"]["manifold_scan_budget_integrity"] == 1.0
    assert report["metrics"]["manifold_indexed_candidate_integrity"] == 1.0
    assert report["metrics"]["manifold_index_scan_reduction"] == 1.0


def test_local_manifold_transition_memory_is_exposed_from_nn_package() -> None:
    import sara_engine.nn as nn

    memory = nn.LocalManifoldTransitionMemory(capacity=1)
    memory.add_trajectory(
        "public-api-path",
        source_events=[1],
        next_events=[2],
        causal_edges=[{"from": "x", "to": "y", "support": 0.9}],
    )

    assert memory.predict_next_ids([1], limit=1) == [2]


def test_local_manifold_transition_memory_returns_isolated_trajectory_copies() -> None:
    memory = LocalManifoldTransitionMemory(capacity=2)
    added = memory.add_trajectory(
        "isolated-path",
        source_events=[1, 2],
        next_events=[10],
        correction_events=[11],
        causal_edges=[{"from": "a", "to": "b", "support": 0.9}],
    )
    graph = memory.trajectory_graph()
    report = memory.evaluate(
        query_events=[1, 2],
        withheld_expected_events=[10],
        case_specs=[
            {
                "case_id": "isolated",
                "query_events": [1, 2],
                "expected_trajectory_id": "isolated-path",
                "expected_next_events": [10],
            },
        ],
    )

    added["source_ids"].append(99)
    added["causal_edges"][0]["support"] = 0.1
    graph[0]["next_ids"].append(999)
    graph[0]["causal_edges"][0]["support"] = 0.1
    report["trajectory_graph"][0]["source_ids"].append(100)

    fresh_graph = memory.trajectory_graph()
    assert fresh_graph[0]["source_ids"] == [1, 2]
    assert fresh_graph[0]["next_ids"] == [10]
    assert fresh_graph[0]["correction_ids"] == [11]
    assert fresh_graph[0]["causal_edges"][0]["support"] == 0.9
    assert memory.predict_next_ids([1, 2], limit=1) == [10]


def test_local_manifold_transition_memory_respects_explicit_scan_budget() -> None:
    memory = LocalManifoldTransitionMemory(capacity=4)
    memory.add_trajectory("path-a", source_events=[1, 2], next_events=[10])
    memory.add_trajectory("path-b", source_events=[3, 4], next_events=[20])
    memory.add_trajectory("path-c", source_events=[5, 6], next_events=[30])

    nearest = memory.query([5, 6], limit=2, max_scan=2)
    predicted = memory.predict_next_ids([5, 6], limit=2, max_scan=2)

    assert len(nearest) == 1
    assert nearest[0]["trajectory_id"] == "path-c"
    assert nearest[0]["indexed_candidate"] is True
    assert nearest[0]["scanned_trajectory_count"] == 1
    assert predicted == [30]


def test_local_manifold_transition_memory_does_not_predict_for_empty_query() -> None:
    memory = LocalManifoldTransitionMemory(capacity=2)
    memory.add_trajectory("path-a", source_events=[1, 2], next_events=[10])
    memory.add_trajectory("path-b", source_events=[3, 4], next_events=[20])

    nearest = memory.query([], limit=1)
    predicted = memory.predict_next_ids([], limit=1)

    assert nearest == []
    assert predicted == []


def test_local_manifold_transition_memory_does_not_predict_for_unknown_query() -> None:
    memory = LocalManifoldTransitionMemory(capacity=2)
    memory.add_trajectory("path-a", source_events=[1, 2], next_events=[10])
    memory.add_trajectory("path-b", source_events=[3, 4], next_events=[20])

    nearest = memory.query([99, 100], limit=1)
    predicted = memory.predict_next_ids([99, 100], limit=1)
    report = memory.evaluate(
        query_events=[99, 100],
        withheld_expected_events=[999],
        scan_budget=2,
        case_specs=[
            {
                "case_id": "unknown",
                "query_events": [99, 100],
                "expected_trajectory_id": "missing-path",
                "expected_next_events": [999],
            },
        ],
    )

    assert nearest == []
    assert predicted == []
    assert report["candidate_miss"] is True
    assert report["candidate_miss_count"] == 2
    assert report["metrics"]["manifold_candidate_miss_guard"] == 1.0
    assert report["metrics"]["withheld_trajectory_recall"] == 0.0
    assert report["case_results"][0]["candidate_miss"] is True
    assert report["case_results"][0]["candidate_count"] == 0


def test_local_manifold_transition_memory_prefers_refreshed_tie_candidate() -> None:
    memory = LocalManifoldTransitionMemory(capacity=4)
    memory.add_trajectory(
        "older-tie",
        source_events=[1, 2],
        next_events=[10],
        event_cost_proxy=0.01,
    )
    memory.add_trajectory(
        "refreshed-tie",
        source_events=[1, 2],
        next_events=[20],
        event_cost_proxy=0.50,
    )
    memory.add_trajectory(
        "refreshed-tie",
        source_events=[1, 2],
        next_events=[20],
        event_cost_proxy=0.50,
    )

    nearest = memory.query([1, 2], limit=2)

    assert nearest[0]["trajectory_id"] == "refreshed-tie"
    assert nearest[0]["refresh_count"] == 2
    assert nearest[0]["update_step"] == 3
    assert memory.predict_next_ids([1, 2], limit=1) == [20]


def test_local_manifold_transition_memory_preserves_recall_under_capacity_pressure() -> None:
    memory = LocalManifoldTransitionMemory(capacity=10)
    for index in range(8):
        base = 100 + index * 10
        memory.add_trajectory(
            f"distractor-{index}",
            source_events=[base, base + 1],
            next_events=[base + 2],
            causal_edges=[{"from": f"d{index}", "to": "unused", "support": 0.9}],
            event_cost_proxy=0.25,
        )
    memory.add_trajectory(
        "late-critical-path",
        source_events=[7, 8, 9],
        next_events=[70, 71],
        causal_edges=[{"from": "critical", "to": "retained", "support": 0.94}],
        event_cost_proxy=0.12,
    )

    report = memory.evaluate(
        query_events=[7, 8, 9],
        withheld_expected_events=[70, 71],
        scan_budget=2,
        case_specs=[
            {
                "case_id": "late-critical",
                "query_events": [7, 8, 9],
                "expected_trajectory_id": "late-critical-path",
                "expected_next_events": [70, 71],
            },
        ],
    )

    assert report["dense_scan_baseline_count"] == 9
    assert report["max_scanned_trajectory_count"] == 1
    assert report["indexed_scan_reduction_ratio"] > 0.85
    assert report["trajectory_top_match_ratio"] == 1.0
    assert report["average_case_recall"] == 1.0
    assert report["metrics"]["manifold_index_scan_reduction"] == 1.0


def test_local_manifold_transition_memory_replay_refresh_survives_eviction() -> None:
    memory = LocalManifoldTransitionMemory(capacity=3)
    memory.add_trajectory("anchor-path", source_events=[1, 2], next_events=[10])
    memory.add_trajectory("distractor-a", source_events=[20, 21], next_events=[22])
    memory.add_trajectory("distractor-b", source_events=[30, 31], next_events=[32])

    memory.add_trajectory("anchor-path", source_events=[1, 2], next_events=[10])
    memory.add_trajectory("distractor-c", source_events=[40, 41], next_events=[42])
    memory.add_trajectory("distractor-d", source_events=[50, 51], next_events=[52])

    graph = memory.trajectory_graph()
    graph_ids = [item["trajectory_id"] for item in graph]
    anchor = next(item for item in graph if item["trajectory_id"] == "anchor-path")
    report = memory.evaluate(
        query_events=[1, 2],
        withheld_expected_events=[10],
        scan_budget=2,
        case_specs=[
            {
                "case_id": "refreshed-anchor",
                "query_events": [1, 2],
                "expected_trajectory_id": "anchor-path",
                "expected_next_events": [10],
            },
        ],
    )

    assert graph_ids == ["anchor-path", "distractor-c", "distractor-d"]
    assert anchor["refresh_count"] == 2
    assert anchor["update_step"] == 4
    assert report["trajectory_top_match_ratio"] == 1.0
    assert report["average_case_recall"] == 1.0
    assert memory.predict_next_ids([1, 2], limit=1, max_scan=2) == [10]
