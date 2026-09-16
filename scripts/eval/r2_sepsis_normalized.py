#!/usr/bin/env python3
"""Execute the registered training-only normalized-router diagnostic."""
from __future__ import annotations
import hashlib,json,platform,sys
from pathlib import Path
from sara_engine.evaluation.r2_sepsis_normalized import PROTOCOL_SHA256,load_protocol,run_diagnostic
from sara_engine.utils.project_paths import ensure_parent_directory,workspace_path
SOURCES=("src/sara_engine/learning/normalized_router.py","src/sara_engine/evaluation/r2_sepsis_normalized.py","scripts/eval/r2_sepsis_normalized.py")
def main():
 marker=Path(ensure_parent_directory(workspace_path("evaluation","r2_sepsis_normalized_v1_attempt.json")));output=Path(ensure_parent_directory(workspace_path("evaluation","r2_sepsis_normalized_v1_result.json")))
 if marker.exists() or output.exists():raise RuntimeError("Registered normalized-router attempt already consumed")
 provenance={"protocol_sha256":PROTOCOL_SHA256,"sources":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES},"python":sys.version,"platform":platform.platform()}
 marker.write_text(json.dumps(provenance,indent=2)+"\n");result=run_diagnostic(load_protocol());result["provenance"]=provenance;output.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({"output":str(output),"decision":result["decision"]}));return 0
if __name__=="__main__":raise SystemExit(main())
