"""Helpers for artifact-state normalization in managed evaluation reports."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def artifact_state(
    report: Mapping[str, Any] | None,
    *,
    pass_field: str | None = "passed",
) -> str:
    if not isinstance(report, Mapping) or not report:
        return "missing"
    if pass_field is None:
        return "present"
    if pass_field in report:
        return "passed" if bool(report.get(pass_field)) else "failed"
    return "present"


def display_artifact_value(value: Any, *, missing_label: str = "missing_artifact") -> str:
    return missing_label if value is None else str(value)


def format_artifact_state_line(
    label: str,
    items: Sequence[tuple[str, Any]],
    *,
    missing_label: str = "missing",
) -> str:
    parts = [
        f"{name}={missing_label if value is None else value}"
        for name, value in items
    ]
    return f"{label}: " + ", ".join(parts)
