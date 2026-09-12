"""Evaluate the frozen conservative verified-chat routing fixture."""

import json
import sys
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.memory.verified_chat_router import route_verified_chat
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory


def evaluate(protocol):
    rows = []
    for language in protocol["languages"]:
        aliases = tuple(tuple(pair) for pair in language["aliases"])
        markers = tuple(language["markers"])
        for text, expected in language["cases"]:
            route = route_verified_chat(text, language=language["language"], aliases=aliases, markers=markers)
            rows.append({"language": language["language"], "text": text, "expected": expected,
                         "decision": route.decision, "parse_decision": route.parse_decision,
                         "correct": route.decision == expected})
    return {"schema": protocol["schema"], "scope": protocol["scope"], "cases": rows,
            "correct": sum(row["correct"] for row in rows), "case_count": len(rows),
            "router_passed": all(row["correct"] for row in rows), "automatic_chat_promotion": False}


def main():
    raw = Path(processed_data_path("benchmark_fixtures", "verified_chat_routing_v1.json")).read_bytes()
    report = evaluate(json.loads(raw))
    report["fixture_sha256"] = sha256(raw).hexdigest()
    output = Path(ensure_parent_directory(workspace_path("evaluation", "verified_chat_routing_v1.json")))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("correct", "case_count", "router_passed", "automatic_chat_promotion")}))


if __name__ == "__main__":
    main()
