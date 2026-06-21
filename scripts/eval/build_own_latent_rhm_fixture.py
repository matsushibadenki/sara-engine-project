#!/usr/bin/env python3
"""Build a tiny deterministic sparse own-latent hierarchy fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path  # noqa: E402


DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "own_latent_rhm_cases.jsonl")


GROUPS: Dict[str, Dict[str, object]] = {
    "avian_motion": {
        "parent": "animal_dynamics",
        "latent_terms": ["animal_dynamics", "winged_agent", "flight_motion", "feather_signal"],
        "train_terms": ["sparrow", "falcon", "wing", "glide", "nest", "feather", "sky", "perch"],
        "eval_terms": ["raptor", "soar", "plume", "aerial", "roost"],
    },
    "aquatic_motion": {
        "parent": "animal_dynamics",
        "latent_terms": ["animal_dynamics", "water_agent", "swim_motion", "fin_signal"],
        "train_terms": ["salmon", "dolphin", "fin", "current", "reef", "stream", "dive", "gill"],
        "eval_terms": ["orca", "lagoon", "submerge", "flipper", "tide"],
    },
    "wheeled_transport": {
        "parent": "machine_dynamics",
        "latent_terms": ["machine_dynamics", "ground_vehicle", "wheel_motion", "road_signal"],
        "train_terms": ["truck", "bicycle", "wheel", "road", "axle", "brake", "lane", "engine"],
        "eval_terms": ["scooter", "highway", "tire", "pedal", "garage"],
    },
    "rail_transport": {
        "parent": "machine_dynamics",
        "latent_terms": ["machine_dynamics", "rail_vehicle", "track_motion", "station_signal"],
        "train_terms": ["train", "tram", "rail", "station", "track", "signal", "carriage", "platform"],
        "eval_terms": ["subway", "locomotive", "depot", "switchyard", "tunnel"],
    },
}


def _surface_sentence(label: str, terms: Sequence[str], index: int) -> str:
    pivot = terms[index % len(terms)]
    partner = terms[(index * 3 + 1) % len(terms)]
    support = terms[(index * 5 + 2) % len(terms)]
    return (
        f"{pivot} context links {partner} with {support} in a sparse event hierarchy "
        f"for latent group {label.replace('_', ' ')}."
    )


def build_cases(train_per_group: int = 8, eval_per_group: int = 4) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for label, spec in GROUPS.items():
        latent_terms = [str(item) for item in spec["latent_terms"]]
        parent = str(spec["parent"])
        train_terms = [str(item) for item in spec["train_terms"]]
        eval_terms = [str(item) for item in spec["eval_terms"]]
        for index in range(train_per_group):
            cases.append(
                {
                    "case_id": f"train_{label}_{index:02d}",
                    "split": "train",
                    "latent_group": label,
                    "parent_group": parent,
                    "surface_text": _surface_sentence(label, train_terms, index),
                    "latent_terms": latent_terms,
                    "expected_behavior": "recover_latent_group",
                }
            )
        for index in range(eval_per_group):
            cases.append(
                {
                    "case_id": f"eval_{label}_{index:02d}",
                    "split": "eval",
                    "latent_group": label,
                    "parent_group": parent,
                    "surface_text": _surface_sentence(label, eval_terms, index),
                    "latent_terms": latent_terms,
                    "expected_behavior": "recover_latent_group",
                }
            )
    return cases


def write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return resolved


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the own-latent RHM-style benchmark fixture.")
    parser.add_argument("--output-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--train-per-group", type=int, default=8)
    parser.add_argument("--eval-per-group", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cases = build_cases(train_per_group=args.train_per_group, eval_per_group=args.eval_per_group)
    output_path = write_jsonl(args.output_path, cases)
    print(json.dumps({"case_count": len(cases), "output_path": output_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
