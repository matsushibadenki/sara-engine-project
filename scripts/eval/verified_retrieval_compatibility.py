"""Run the isolated retrieval comparison without loading production state."""

import json
from hashlib import sha256
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.evaluation.verified_retrieval_compatibility import evaluate
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory


def main():
    fixture = Path(processed_data_path("benchmark_fixtures", "verified_retrieval_compatibility.json"))
    raw = fixture.read_bytes()
    report = evaluate(json.loads(raw))
    report["fixture_sha256"] = sha256(raw).hexdigest()
    output = Path(ensure_parent_directory(workspace_path("evaluation", "verified_retrieval_compatibility_v2.json")))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "metrics": report["metrics"], "component_contract_passed": report["component_contract_passed"]}))


if __name__ == "__main__":
    main()
