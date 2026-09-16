#!/usr/bin/env python3
"""Execute the single registered final BPI 2012 hybrid attempt."""

from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
from sara_engine.evaluation.r2_bpi2012_hybrid_final import PROTOCOL_SHA256, load_protocol, run_final
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path

SOURCES=("src/sara_engine/learning/confidence_router.py","src/sara_engine/learning/sparse_multiclass.py",
"src/sara_engine/evaluation/r2_bpi2012_hybrid_final.py","scripts/eval/r2_bpi2012_hybrid_final.py")

def main()->int:
    marker=Path(ensure_parent_directory(workspace_path("evaluation","r2_bpi2012_hybrid_final_v1_attempt.json")))
    output=Path(ensure_parent_directory(workspace_path("evaluation","r2_bpi2012_hybrid_final_v1_result.json")))
    if marker.exists() or output.exists():raise RuntimeError("Registered final BPI hybrid attempt was already consumed")
    provenance={"protocol_sha256":PROTOCOL_SHA256,"sources":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES},
                "python":sys.version,"platform":platform.platform()}
    marker.write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    result=run_final(load_protocol());result["provenance"]=provenance
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"decision":result["decision"]}));return 0

if __name__=="__main__":raise SystemExit(main())
