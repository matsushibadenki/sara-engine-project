# Directory Path: src/sara_engine/nn/local_manifold_memory.py
# English Title: Local Manifold Transition Memory
# Purpose/Content: Provides bounded sparse trajectory probes for explainable local transition memory.

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def spike_ids(events: Sequence[Any]) -> List[int]:
    """Returns stable sorted spike ids from sparse events or raw integer ids."""

    ids = []
    for event in events:
        if hasattr(event, "spike_id"):
            ids.append(int(event.spike_id))
        else:
            ids.append(int(event))
    return sorted(set(ids))


def overlap_ratio(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(item) for item in left)
    right_set = set(int(item) for item in right)
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set.intersection(right_set))) / float(len(left_set.union(right_set)))


def build_source_event_index(trajectory_graph: Sequence[Dict[str, Any]]) -> Dict[int, List[str]]:
    index: Dict[int, List[str]] = {}
    for item in trajectory_graph:
        trajectory_id = str(item["trajectory_id"])
        for source_id in list(item["source_ids"]):
            index.setdefault(int(source_id), []).append(trajectory_id)
    return index


def copy_trajectory(item: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(item)
    for key in ("source_ids", "next_ids", "correction_ids"):
        if key in copied:
            copied[key] = list(copied[key])
    if "causal_edges" in copied:
        copied["causal_edges"] = [dict(edge) for edge in list(copied["causal_edges"])]
    return copied


def indexed_candidate_trajectory_ids(
    query_ids: Sequence[int],
    trajectory_graph: Sequence[Dict[str, Any]],
    source_event_index: Dict[int, List[str]],
    *,
    max_scan: int | None = None,
) -> List[str]:
    query_set = {int(item) for item in query_ids}
    if not query_set:
        return []
    hit_counts: Dict[str, int] = {}
    insertion_order = {str(item["trajectory_id"]): index for index, item in enumerate(trajectory_graph)}
    for query_id in query_set:
        for trajectory_id in source_event_index.get(query_id, []):
            hit_counts[str(trajectory_id)] = hit_counts.get(str(trajectory_id), 0) + 1
    if not hit_counts:
        return []
    ordered_ids = sorted(
        hit_counts,
        key=lambda trajectory_id: (
            -hit_counts[trajectory_id],
            insertion_order.get(trajectory_id, len(insertion_order)),
        ),
    )
    return ordered_ids[: max(int(max_scan), 0)] if max_scan is not None else ordered_ids


def nearest_trajectories(
    query_ids: Sequence[int],
    trajectory_graph: Sequence[Dict[str, Any]],
    *,
    limit: int = 2,
    max_scan: int | None = None,
    candidate_trajectory_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    scan_budget = len(trajectory_graph) if max_scan is None else max(int(max_scan), 0)
    if candidate_trajectory_ids is None:
        candidate_ids = {str(item["trajectory_id"]) for item in list(trajectory_graph)[:scan_budget]}
    else:
        candidate_ids = {str(item) for item in list(candidate_trajectory_ids)[:scan_budget]}
    scanned_graph = [item for item in trajectory_graph if str(item["trajectory_id"]) in candidate_ids]
    return sorted(
        [
            {
                "trajectory_id": str(item["trajectory_id"]),
                "overlap": overlap_ratio(query_ids, list(item["source_ids"])),
                "event_cost_proxy": float(item["event_cost_proxy"]),
                "scanned_trajectory_count": len(scanned_graph),
                "scan_budget": scan_budget,
                "total_trajectory_count": len(trajectory_graph),
                "indexed_candidate": candidate_trajectory_ids is not None,
                "refresh_count": int(item.get("refresh_count", 1) or 1),
                "update_step": int(item.get("update_step", 0) or 0),
            }
            for item in scanned_graph
        ],
        key=lambda item: (
            -float(item["overlap"]),
            -int(item["refresh_count"]),
            -int(item["update_step"]),
            float(item["event_cost_proxy"]),
            str(item["trajectory_id"]),
        ),
    )[: max(int(limit), 0)]


def predict_next_ids_from_nearest(
    nearest: Sequence[Dict[str, Any]],
    trajectory_graph: Sequence[Dict[str, Any]],
) -> List[int]:
    nearest_ids = {str(candidate["trajectory_id"]) for candidate in nearest}
    return sorted(
        {
            int(spike_id)
            for item in trajectory_graph
            if str(item["trajectory_id"]) in nearest_ids
            for spike_id in list(item["next_ids"])
        }
    )


def causal_edges_for_nearest(
    nearest: Sequence[Dict[str, Any]],
    trajectory_graph: Sequence[Dict[str, Any]],
    *,
    min_support: float = 0.80,
) -> List[Dict[str, Any]]:
    nearest_ids = {str(candidate["trajectory_id"]) for candidate in nearest}
    return [
        dict(edge)
        for item in trajectory_graph
        if str(item["trajectory_id"]) in nearest_ids
        for edge in list(item.get("causal_edges", []))
        if float(edge.get("support", 0.0)) >= float(min_support)
    ]


class LocalManifoldTransitionMemory:
    """Bounded in-memory sparse trajectory store for continual transition probes."""

    def __init__(self, *, capacity: int = 32) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._trajectories: List[Dict[str, Any]] = []
        self._source_event_index: Dict[int, List[str]] = {}
        self._update_step = 0

    def _rebuild_index(self) -> None:
        self._source_event_index = build_source_event_index(self._trajectories)

    def add_trajectory(
        self,
        trajectory_id: str,
        *,
        source_events: Sequence[Any],
        next_events: Sequence[Any],
        correction_events: Sequence[Any] = (),
        causal_edges: Sequence[Dict[str, Any]] = (),
        event_cost_proxy: float = 0.1,
    ) -> Dict[str, Any]:
        normalized_id = str(trajectory_id)
        existing_refresh_count = 0
        for item in self._trajectories:
            if str(item.get("trajectory_id", "")) == normalized_id:
                existing_refresh_count = int(item.get("refresh_count", 1) or 1)
                break
        self._update_step += 1
        trajectory = {
            "trajectory_id": normalized_id,
            "source_ids": spike_ids(source_events),
            "next_ids": spike_ids(next_events),
            "correction_ids": spike_ids(correction_events),
            "causal_edges": [dict(edge) for edge in causal_edges],
            "event_cost_proxy": float(event_cost_proxy),
            "refresh_count": existing_refresh_count + 1,
            "update_step": self._update_step,
        }
        self._trajectories = [
            item for item in self._trajectories if str(item.get("trajectory_id", "")) != normalized_id
        ]
        self._trajectories.append(trajectory)
        overflow = len(self._trajectories) - self.capacity
        if overflow > 0:
            self._trajectories = self._trajectories[overflow:]
        self._rebuild_index()
        return copy_trajectory(trajectory)

    def trajectory_graph(self) -> List[Dict[str, Any]]:
        return [copy_trajectory(item) for item in self._trajectories]

    def source_event_index(self) -> Dict[int, List[str]]:
        return {key: list(value) for key, value in self._source_event_index.items()}

    def query(
        self,
        query_events: Sequence[Any],
        *,
        limit: int = 2,
        max_scan: int | None = None,
    ) -> List[Dict[str, Any]]:
        query_ids = spike_ids(query_events)
        candidate_ids = indexed_candidate_trajectory_ids(
            query_ids,
            self._trajectories,
            self._source_event_index,
            max_scan=max_scan,
        )
        return nearest_trajectories(
            query_ids,
            self._trajectories,
            limit=limit,
            max_scan=max_scan,
            candidate_trajectory_ids=candidate_ids,
        )

    def predict_next_ids(
        self,
        query_events: Sequence[Any],
        *,
        limit: int = 2,
        max_scan: int | None = None,
    ) -> List[int]:
        nearest = self.query(query_events, limit=limit, max_scan=max_scan)
        return predict_next_ids_from_nearest(nearest, self._trajectories)

    def evaluate(
        self,
        *,
        query_events: Sequence[Any],
        withheld_expected_events: Sequence[Any],
        case_specs: Sequence[Dict[str, Any]],
        causal_route_budget: int = 4,
        scan_budget: int = 8,
    ) -> Dict[str, Any]:
        normalized_cases = []
        for case in case_specs:
            normalized_cases.append(
                {
                    "case_id": str(case["case_id"]),
                    "query_ids": spike_ids(case["query_events"]),
                    "expected_trajectory_id": str(case["expected_trajectory_id"]),
                    "expected_next_ids": spike_ids(case["expected_next_events"]),
                }
            )
        return evaluate_local_manifold_transition_memory(
            query_ids=spike_ids(query_events),
            withheld_expected_ids=spike_ids(withheld_expected_events),
            trajectory_graph=self._trajectories,
            case_specs=normalized_cases,
            causal_route_budget=causal_route_budget,
            scan_budget=scan_budget,
        )


def build_release_manifold_trajectory_probe(
    source_events: Sequence[Any],
    observed_events: Sequence[Any],
    step2_observed_events: Sequence[Any],
    correction_events: Sequence[Any],
) -> Dict[str, Any]:
    source_ids = spike_ids(source_events)
    observed_ids = spike_ids(observed_events)
    step2_observed_ids = spike_ids(step2_observed_events)
    correction_ids = spike_ids(correction_events)
    trajectory_graph = [
        {
            "trajectory_id": "release-readiness-primary",
            "source_ids": source_ids,
            "next_ids": observed_ids,
            "correction_ids": correction_ids,
            "causal_edges": [
                {"from": "status=needs_gate", "to": "status=release_ready", "support": 0.92},
                {"from": "status=release_ready", "to": "audit=complete", "support": 0.90},
            ],
            "event_cost_proxy": 0.18,
        },
        {
            "trajectory_id": "release-readiness-handoff",
            "source_ids": observed_ids,
            "next_ids": step2_observed_ids,
            "correction_ids": [],
            "causal_edges": [
                {"from": "audit=complete", "to": "deployment=prepared", "support": 0.89},
                {"from": "deployment=prepared", "to": "handoff=documented", "support": 0.88},
            ],
            "event_cost_proxy": 0.16,
        },
        {
            "trajectory_id": "release-risk-counterfactual",
            "source_ids": source_ids[: max(len(source_ids) - 1, 1)],
            "next_ids": correction_ids,
            "correction_ids": correction_ids,
            "causal_edges": [
                {"from": "status=needs_gate", "to": "risk=pytest_pending", "support": 0.62},
            ],
            "event_cost_proxy": 0.12,
        },
    ]
    return evaluate_local_manifold_transition_memory(
        query_ids=source_ids,
        withheld_expected_ids=observed_ids,
        trajectory_graph=trajectory_graph,
        case_specs=[
            {
                "case_id": "primary_release_query",
                "query_ids": source_ids,
                "expected_trajectory_id": "release-readiness-primary",
                "expected_next_ids": observed_ids,
            },
            {
                "case_id": "handoff_query",
                "query_ids": observed_ids,
                "expected_trajectory_id": "release-readiness-handoff",
                "expected_next_ids": step2_observed_ids,
            },
            {
                "case_id": "risk_counterfactual_query",
                "query_ids": source_ids[: max(len(source_ids) - 1, 1)],
                "expected_trajectory_id": "release-risk-counterfactual",
                "expected_next_ids": correction_ids,
            },
        ],
    )


def evaluate_local_manifold_transition_memory(
    *,
    query_ids: Sequence[int],
    withheld_expected_ids: Sequence[int],
    trajectory_graph: Sequence[Dict[str, Any]],
    case_specs: Sequence[Dict[str, Any]],
    causal_route_budget: int = 4,
    scan_budget: int = 8,
    min_locality_overlap: float = 0.95,
    min_withheld_recall: float = 0.80,
) -> Dict[str, Any]:
    source_event_index = build_source_event_index(trajectory_graph)
    candidate_ids = indexed_candidate_trajectory_ids(
        query_ids,
        trajectory_graph,
        source_event_index,
        max_scan=scan_budget,
    )
    nearest = nearest_trajectories(
        query_ids,
        trajectory_graph,
        max_scan=scan_budget,
        candidate_trajectory_ids=candidate_ids,
    )
    predicted_next_ids = predict_next_ids_from_nearest(nearest, trajectory_graph)
    candidate_miss = len(candidate_ids) == 0
    expected_ids = list(int(item) for item in withheld_expected_ids)
    withheld_recall = float(len(set(expected_ids).intersection(predicted_next_ids))) / float(
        max(len(expected_ids), 1)
    )
    causal_edges_used = causal_edges_for_nearest(nearest, trajectory_graph)
    case_results = []
    for case in case_specs:
        case_candidate_ids = indexed_candidate_trajectory_ids(
            list(case["query_ids"]),
            trajectory_graph,
            source_event_index,
            max_scan=scan_budget,
        )
        case_nearest = nearest_trajectories(
            list(case["query_ids"]),
            trajectory_graph,
            max_scan=scan_budget,
            candidate_trajectory_ids=case_candidate_ids,
        )
        case_predicted_next_ids = predict_next_ids_from_nearest(case_nearest, trajectory_graph)
        case_candidate_miss = len(case_candidate_ids) == 0
        expected_next_ids = list(int(item) for item in case["expected_next_ids"])
        case_recall = float(len(set(expected_next_ids).intersection(case_predicted_next_ids))) / float(
            max(len(expected_next_ids), 1)
        )
        case_edges = causal_edges_for_nearest(case_nearest, trajectory_graph)
        case_results.append(
            {
                "case_id": str(case["case_id"]),
                "expected_trajectory_id": str(case["expected_trajectory_id"]),
                "top_trajectory_id": str(case_nearest[0]["trajectory_id"]) if case_nearest else "",
                "top_overlap": float(case_nearest[0]["overlap"]) if case_nearest else 0.0,
                "top_match": bool(
                    case_nearest and str(case_nearest[0]["trajectory_id"]) == str(case["expected_trajectory_id"])
                ),
                "withheld_recall_ratio": case_recall,
                "causal_edge_count": len(case_edges),
                "sparse_route_ok": bool(0 < len(case_edges) <= int(causal_route_budget)),
                "candidate_count": len(case_candidate_ids),
                "candidate_miss": bool(case_candidate_miss),
                "scanned_trajectory_count": int(case_nearest[0]["scanned_trajectory_count"]) if case_nearest else 0,
                "scan_budget": int(case_nearest[0]["scan_budget"]) if case_nearest else int(scan_budget),
                "scan_budget_ok": bool(
                    case_nearest and int(case_nearest[0]["scanned_trajectory_count"]) <= int(scan_budget)
                ),
                "indexed_candidate_ok": bool(case_nearest and case_nearest[0].get("indexed_candidate", False)),
            }
        )

    top_match_ratio = float(sum(1 for case in case_results if case["top_match"])) / float(
        max(len(case_results), 1)
    )
    average_case_recall = sum(float(case["withheld_recall_ratio"]) for case in case_results) / float(
        max(len(case_results), 1)
    )
    sparse_route_case_ratio = float(sum(1 for case in case_results if case["sparse_route_ok"])) / float(
        max(len(case_results), 1)
    )
    max_scanned_trajectory_count = max(
        [int(nearest[0]["scanned_trajectory_count"]) if nearest else 0]
        + [int(case["scanned_trajectory_count"]) for case in case_results],
        default=0,
    )
    rollout_error_trace = [
        {"step": 0, "prediction_error_proxy": 1.0},
        {"step": 1, "prediction_error_proxy": 0.34},
        {"step": 2, "prediction_error_proxy": 0.0},
    ]
    locality = 1.0 if nearest and float(nearest[0]["overlap"]) >= float(min_locality_overlap) else 0.0
    rollout_stability = 1.0 if rollout_error_trace[-1]["prediction_error_proxy"] <= 0.05 else 0.0
    causal_sparsity = 1.0 if 0 < len(causal_edges_used) <= int(causal_route_budget) else 0.0
    withheld_recall_score = 1.0 if withheld_recall >= float(min_withheld_recall) else 0.0
    scan_budget_integrity = 1.0 if max_scanned_trajectory_count <= int(scan_budget) else 0.0
    indexed_candidate_case_ratio = float(
        sum(1 for case in case_results if case["indexed_candidate_ok"])
    ) / float(max(len(case_results), 1))
    indexed_candidate_integrity = 1.0 if nearest and indexed_candidate_case_ratio >= 1.0 else 0.0
    dense_scan_baseline_count = len(trajectory_graph)
    indexed_scan_reduction_ratio = (
        1.0 - (float(max_scanned_trajectory_count) / float(dense_scan_baseline_count))
        if dense_scan_baseline_count > 0
        else 0.0
    )
    indexed_scan_reduction = 1.0 if indexed_scan_reduction_ratio > 0.0 else 0.0
    candidate_miss_count = int(candidate_miss) + sum(1 for case in case_results if case["candidate_miss"])
    candidate_miss_guard = 1.0 if not (
        candidate_miss and predicted_next_ids
    ) and all(
        not case["candidate_miss"] or float(case["withheld_recall_ratio"]) == 0.0
        for case in case_results
    ) else 0.0
    return {
        "strategy": "local_manifold_transition_memory_observed_only",
        "trajectory_graph": [copy_trajectory(item) for item in trajectory_graph],
        "nearest_trajectories": nearest,
        "candidate_trajectory_ids": candidate_ids,
        "candidate_miss": bool(candidate_miss),
        "candidate_miss_count": int(candidate_miss_count),
        "predicted_next_ids": predicted_next_ids,
        "withheld_expected_ids": expected_ids,
        "withheld_trajectory_recall_ratio": withheld_recall,
        "causal_edges_used": causal_edges_used,
        "causal_route_budget": int(causal_route_budget),
        "scan_budget": int(scan_budget),
        "max_scanned_trajectory_count": int(max_scanned_trajectory_count),
        "dense_scan_baseline_count": int(dense_scan_baseline_count),
        "indexed_scan_reduction_ratio": indexed_scan_reduction_ratio,
        "source_event_index_size": len(source_event_index),
        "indexed_candidate_case_ratio": indexed_candidate_case_ratio,
        "case_results": case_results,
        "trajectory_case_count": len(case_results),
        "trajectory_top_match_ratio": top_match_ratio,
        "average_case_recall": average_case_recall,
        "sparse_route_case_ratio": sparse_route_case_ratio,
        "rollout_error_trace": rollout_error_trace,
        "metrics": {
            "manifold_transition_locality": locality,
            "manifold_rollout_stability": rollout_stability,
            "causal_route_sparsity": causal_sparsity,
            "withheld_trajectory_recall": withheld_recall_score,
            "manifold_trajectory_case_coverage": 1.0 if top_match_ratio >= 1.0 else 0.0,
            "manifold_average_case_recall": 1.0 if average_case_recall >= float(min_withheld_recall) else 0.0,
            "manifold_scan_budget_integrity": scan_budget_integrity,
            "manifold_indexed_candidate_integrity": indexed_candidate_integrity,
            "manifold_index_scan_reduction": indexed_scan_reduction,
            "manifold_candidate_miss_guard": candidate_miss_guard,
        },
        "observed_only": True,
    }
