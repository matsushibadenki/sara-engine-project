import json
import time
from typing import Any, Dict, List, Optional

from ..utils.project_paths import ensure_parent_directory


def flatten_phase3_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}

    for component_name, component_report in component_reports.items():
        if not isinstance(component_report, dict):
            continue
        metrics = component_report.get("metrics", {})
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                try:
                    flattened[f"{component_name}.{metric_name}"] = float(value)
                except (TypeError, ValueError):
                    continue
        try:
            flattened[f"{component_name}.overall_score"] = float(
                component_report.get("overall_score", 0.0)
            )
        except (TypeError, ValueError):
            pass

    focus_summary = report.get("focus_summary", {})
    if isinstance(focus_summary, dict):
        for focus_name, focus_report in focus_summary.items():
            if not isinstance(focus_report, dict):
                continue
            try:
                flattened[f"focus.{focus_name}.score"] = float(focus_report.get("score", 0.0))
            except (TypeError, ValueError):
                pass
            metrics = focus_report.get("metrics", {})
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    try:
                        flattened[f"focus.{focus_name}.{metric_name}"] = float(value)
                    except (TypeError, ValueError):
                        continue

    try:
        flattened["suite.overall_score"] = float(report.get("overall_score", 0.0))
    except (TypeError, ValueError):
        pass
    return flattened


def build_phase3_trend(
    current_report: Dict[str, Any],
    previous_report: Optional[Dict[str, Any]] = None,
    regression_tolerance: float = 1e-9,
) -> Dict[str, Any]:
    current_metrics = flatten_phase3_metrics(current_report)
    previous_metrics = flatten_phase3_metrics(previous_report or {})

    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    unchanged: List[str] = []
    new_metrics: List[str] = []

    for metric_name, current_value in sorted(current_metrics.items()):
        if metric_name not in previous_metrics:
            new_metrics.append(metric_name)
            continue

        previous_value = previous_metrics[metric_name]
        delta = current_value - previous_value
        record = {
            "metric": metric_name,
            "previous": previous_value,
            "current": current_value,
            "delta": delta,
        }
        if delta < -regression_tolerance:
            regressions.append(record)
        elif delta > regression_tolerance:
            improvements.append(record)
        else:
            unchanged.append(metric_name)

    return {
        "has_previous": bool(previous_metrics),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "unchanged_count": len(unchanged),
        "new_metric_count": len(new_metrics),
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "new_metrics": new_metrics,
    }


def load_phase3_history(history_path: str) -> List[Dict[str, Any]]:
    try:
        with open(history_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def latest_phase3_report(history_path: str) -> Optional[Dict[str, Any]]:
    history = load_phase3_history(history_path)
    if not history:
        return None
    return history[-1]


def append_phase3_history(
    history_path: str,
    report: Dict[str, Any],
    max_entries: int = 50,
) -> List[Dict[str, Any]]:
    history = load_phase3_history(history_path)
    entry = dict(report)
    entry.setdefault("recorded_at", time.time())
    history.append(entry)
    if max_entries > 0:
        history = history[-max_entries:]

    resolved_path = ensure_parent_directory(history_path)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)
    return history
