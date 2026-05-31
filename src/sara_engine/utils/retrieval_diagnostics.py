from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _float_value(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _bool_value(payload: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(payload.get(key, default))


def normalize_retrieval_diagnostic(
    payload: Dict[str, Any],
    *,
    source: str,
    content_key: str = "clean_content",
) -> Dict[str, Any]:
    content_preview = str(payload.get(content_key, payload.get("content_preview", ""))).strip()
    return {
        "source": source,
        "memory_hit": str(payload.get("memory_hit", payload.get("memory_source", "retrieval"))),
        "content_preview": content_preview,
        "keyword_score": _float_value(payload, "keyword_score"),
        "base_score": _float_value(payload, "base_score", _float_value(payload, "retrieval_score_base")),
        "stability_score": _float_value(payload, "stability_score"),
        "current_keyword_coverage": _float_value(payload, "current_keyword_coverage"),
        "context_keyword_coverage": _float_value(payload, "context_keyword_coverage"),
        "metadata_keyword_coverage": _float_value(payload, "metadata_keyword_coverage"),
        "metadata_keyword_overlap": _float_value(payload, "metadata_keyword_overlap", _float_value(payload, "ltm_metadata_keyword_overlap")),
        "suffix_match": _float_value(payload, "suffix_match"),
        "drift_penalty": _float_value(payload, "drift_penalty"),
        "context_match": _bool_value(payload, "context_match", _bool_value(payload, "ltm_context_match")),
        "role_match": _bool_value(payload, "role_match", _bool_value(payload, "ltm_role_match")),
    }


def format_retrieval_diagnostics(
    diagnostics: Iterable[Dict[str, Any]],
    *,
    title: str = "Recent retrieval diagnostics:",
) -> str:
    normalized = [item for item in diagnostics if isinstance(item, dict)]
    if not normalized:
        return "No retrieval diagnostics recorded."
    normalized = list(reversed(normalized))

    lines: List[str] = [title]
    for item in normalized:
        lines.append(
            "- "
            f"{str(item.get('content_preview', ''))[:60]} | "
            f"source={item.get('source', 'unknown')} "
            f"memory={item.get('memory_hit', 'retrieval')} "
            f"base={_float_value(item, 'base_score'):.2f} "
            f"total={_float_value(item, 'keyword_score'):.2f} "
            f"current={_float_value(item, 'current_keyword_coverage'):.2f} "
            f"context={_float_value(item, 'context_keyword_coverage'):.2f} "
            f"metadata={_float_value(item, 'metadata_keyword_coverage'):.2f} "
            f"overlap={_float_value(item, 'metadata_keyword_overlap'):.2f} "
            f"suffix={_float_value(item, 'suffix_match'):.2f} "
            f"drift={_float_value(item, 'drift_penalty'):.2f} "
            f"stability={_float_value(item, 'stability_score'):.2f}"
        )
    return "\n".join(lines)
