#!/usr/bin/env python3
"""Repair a specific direct-memory association in a managed SARA artifact."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path
from sara_engine.utils.tokenizer import SaraTokenizer


def _build_runtime(model_path_value: str) -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = model_path_value
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.session_memory = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine._load_memory()
    return engine


def _parse_token_list(raw: str) -> List[int]:
    values = [item.strip() for item in raw.replace(",", " ").split()]
    if not values:
        raise ValueError("Token list must contain at least one integer.")
    return [int(item) for item in values]


def _encode_text(text: str, tokenizer_path: Optional[str]) -> List[int]:
    tokenizer = SaraTokenizer(model_path=tokenizer_path) if tokenizer_path else SaraTokenizer()
    return tokenizer.encode(text)


def _find_context_key(engine: SaraInference, context_tokens: Iterable[int]) -> Tuple[int, ...]:
    context_tuple = tuple(int(token) for token in context_tokens)
    for key, stored_context in engine.context_index.items():
        if tuple(int(token) for token in stored_context) == context_tuple:
            return tuple(int(item) for item in key)
    return engine._encode_context_sdr(context_tuple)


def _decode_wrong_token(
    *,
    wrong_token_id: Optional[int],
    wrong_text: Optional[str],
    tokenizer_path: Optional[str],
) -> int:
    if wrong_token_id is not None:
        return int(wrong_token_id)
    if not wrong_text:
        raise ValueError("Either wrong_token_id or wrong_text must be provided.")
    token_ids = _encode_text(wrong_text, tokenizer_path)
    if not token_ids:
        raise ValueError("wrong_text did not produce any token ids.")
    return int(token_ids[-1])


def fix_inference_memory(
    model_path_value: str,
    output_path: str,
    *,
    context_tokens: Optional[List[int]] = None,
    context_text: Optional[str] = None,
    wrong_token_id: Optional[int] = None,
    wrong_text: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    decay: Optional[float] = None,
    dry_run: bool = False,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_model_path = os.path.abspath(model_path_value)
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

    if context_tokens is None:
        if not context_text:
            raise ValueError("Either context_tokens or context_text must be provided.")
        context_tokens = _encode_text(context_text, tokenizer_path)

    if not context_tokens:
        raise ValueError("Context tokens must not be empty.")

    token_id = _decode_wrong_token(
        wrong_token_id=wrong_token_id,
        wrong_text=wrong_text,
        tokenizer_path=tokenizer_path,
    )

    engine = _build_runtime(resolved_model_path)
    context_key = _find_context_key(engine, context_tokens)
    row = engine.direct_map.get(context_key, {})
    before_weight = row.get(token_id)
    removed = False
    decayed = False
    after_weight: Optional[float] = None

    if before_weight is not None:
        if decay is None:
            del row[token_id]
            removed = True
        else:
            after_weight = max(0.0, float(before_weight) * float(decay))
            if after_weight <= 0.0:
                del row[token_id]
                removed = True
            else:
                row[token_id] = after_weight
                decayed = True
        if not row:
            engine.direct_map.pop(context_key, None)
            engine.context_index.pop(context_key, None)
    elif context_key not in engine.direct_map:
        row = {}

    resolved_output_path = ensure_parent_directory(output_path)
    if not dry_run:
        engine.quantization_enabled = bool(getattr(engine, "quantization_enabled", False))
        engine.save_pretrained(resolved_output_path)

    report = {
        "schema": "sara-memory-fix-v1",
        "input_model_path": resolved_model_path,
        "output_model_path": resolved_output_path,
        "dry_run": bool(dry_run),
        "context_tokens": [int(token) for token in context_tokens],
        "context_key": list(context_key),
        "wrong_token_id": int(token_id),
        "matched_context": bool(context_key in engine.direct_map or before_weight is not None),
        "matched_token": before_weight is not None,
        "before_weight": before_weight,
        "after_weight": after_weight,
        "removed": removed,
        "decayed": decayed,
        "pattern_count_after": len(engine.direct_map),
        "context_count_after": len(engine.context_index),
    }

    if report_path:
        resolved_report_path = ensure_parent_directory(report_path)
        with open(resolved_report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        report["report_path"] = resolved_report_path

    return report


def default_fixed_model_path() -> str:
    return model_path("repaired", "memory_fixed.msgpack")


def default_fix_report_path() -> str:
    return workspace_path("reports", "memory_fix_report.json")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove or decay one SARA direct-memory association.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", default=default_fixed_model_path())
    parser.add_argument("--context-tokens", help="Context token ids, separated by spaces or commas.")
    parser.add_argument("--context-text", help="Context text encoded with SaraTokenizer.")
    parser.add_argument("--wrong-token-id", type=int)
    parser.add_argument("--wrong-text", help="Wrong text; the final encoded token is repaired.")
    parser.add_argument("--tokenizer-path", help="Managed SaraTokenizer JSON path for text inputs.")
    parser.add_argument("--decay", type=float, help="Multiply the wrong association weight instead of deleting it.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", default=default_fix_report_path())
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    context_tokens = _parse_token_list(args.context_tokens) if args.context_tokens else None
    report = fix_inference_memory(
        args.model_path,
        args.output_path,
        context_tokens=context_tokens,
        context_text=args.context_text,
        wrong_token_id=args.wrong_token_id,
        wrong_text=args.wrong_text,
        tokenizer_path=args.tokenizer_path,
        decay=args.decay,
        dry_run=bool(args.dry_run),
        report_path=args.report_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["matched_token"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
