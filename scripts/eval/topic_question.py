"""Evaluate a fixed closed-grammar question adapter without enabling chat."""

import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.memory.topic_question import parse_topic_question
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory


def evaluate(protocol):
    cases = []
    for language in protocol["languages"]:
        aliases = tuple(tuple(pair) for pair in language["aliases"])
        for text, requested, excluded in language["cases"]:
            result = parse_topic_question(text, language=language["language"], aliases=aliases)
            if requested is None:
                correct = result.decision == "unsupported_question" and result.query is None
            else:
                correct = result.decision == "parsed" and result.query is not None
                correct = correct and result.query.requested == tuple(requested) and result.query.excluded == tuple(excluded)
                correct = correct and result.query.language == language["language"] and not result.query.unresolved
            cases.append({"language": language["language"], "text": text, "result": asdict(result), "correct": correct})
    return {"schema": protocol["schema"], "scope": protocol["scope"], "cases": cases,
            "correct": sum(case["correct"] for case in cases), "case_count": len(cases), "normal_chat_promotion": False}


def main():
    raw = Path(processed_data_path("benchmark_fixtures", "topic_question_v1.json")).read_bytes()
    report = evaluate(json.loads(raw))
    report["fixture_sha256"] = sha256(raw).hexdigest()
    output = Path(ensure_parent_directory(workspace_path("evaluation", "topic_question_v1.json")))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("correct", "case_count", "normal_chat_promotion")}))


if __name__ == "__main__":
    main()
