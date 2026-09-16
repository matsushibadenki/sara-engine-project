#!/usr/bin/env python3
"""Evaluate durable event-stream checkpoint publication boundaries."""
from __future__ import annotations
import json,os,subprocess,sys
from pathlib import Path
from sara_engine.learning.event_stream_engine import BoundedEventStreamEngine
from sara_engine.learning.normalized_hybrid import NormalizedHybridConfig
from sara_engine.utils.project_paths import ensure_parent_directory,model_path,workspace_path

def _cleanup(path):
    path.unlink(missing_ok=True);path.with_suffix(path.suffix+".lock").unlink(missing_ok=True)
def main():
    engine=BoundedEventStreamEngine(("a","b"),hybrid_config=NormalizedHybridConfig(max_active=7,max_classes=2));checks={};details={}
    collision="checkpoint-gate-collision.json";first=engine.publish(collision,expected_generation=0)
    command=[sys.executable,"scripts/eval/checkpoint_publish_worker.py","--filename",collision,"--expected-generation","1"]
    processes=[subprocess.Popen(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(4)];rows=[]
    for process in processes:
        stdout,stderr=process.communicate(timeout=30)
        if process.returncode:raise RuntimeError(stderr[-1000:])
        rows.append(json.loads(stdout))
    _,generation=BoundedEventStreamEngine.load_with_generation(collision);checks["equal_generation_single_writer"]=sum(row["published"] for row in rows)==1 and generation==2;details["collision_results"]=rows;_cleanup(Path(first.path))
    corruption="checkpoint-gate-corruption.json";receipt=engine.publish(corruption,expected_generation=0);path=Path(receipt.path);document=json.loads(path.read_text());document["payload"]["hybrid"]["sequence"]+=1;path.write_text(json.dumps(document));os.chmod(path,0o600)
    try:BoundedEventStreamEngine.load(corruption);checks["checksum_rejection"]=False
    except ValueError as error:checks["checksum_rejection"]="checksum mismatch" in str(error)
    _cleanup(path)
    permissions="checkpoint-gate-permissions.json";receipt=engine.publish(permissions,expected_generation=0);path=Path(receipt.path);os.chmod(path,0o644)
    try:BoundedEventStreamEngine.load(permissions);checks["permission_rejection"]=False
    except ValueError as error:checks["permission_rejection"]="owner-only" in str(error)
    _cleanup(path)
    oversize="checkpoint-gate-oversize.json";path=Path(ensure_parent_directory(model_path("event_stream_engine",oversize)));path.write_bytes(b" "*(BoundedEventStreamEngine.MAX_ARTIFACT_BYTES+1));os.chmod(path,0o600)
    try:BoundedEventStreamEngine.load(oversize);checks["size_rejection"]=False
    except ValueError as error:checks["size_rejection"]="size limit" in str(error)
    _cleanup(path)
    report={"schema":"sara-event-stream-checkpoint-gate-v1","checks":checks,"details":details,"artifact_max_bytes":BoundedEventStreamEngine.MAX_ARTIFACT_BYTES,"passed":all(checks.values())}
    output=Path(ensure_parent_directory(workspace_path("evaluation","event_stream_checkpoint_gate_v1.json")));output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"output":str(output),"passed":report["passed"],"checks":checks}));return 0
if __name__=="__main__":raise SystemExit(main())
