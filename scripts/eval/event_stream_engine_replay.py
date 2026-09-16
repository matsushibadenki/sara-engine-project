#!/usr/bin/env python3
"""Replay accepted streams through the generic serializable event-stream engine."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from sara_engine.evaluation.bpi2012_data import load_traces as load_bpi,split_name as bpi_split
from sara_engine.evaluation.sepsis_data import load_traces as load_sepsis,split_name as sepsis_split
from sara_engine.learning.event_stream_engine import BoundedEventStreamEngine,EventRouteConfig
from sara_engine.learning.normalized_hybrid import NormalizedHybridConfig
from sara_engine.utils.project_paths import ensure_parent_directory,workspace_path

def _accepted():
    b=json.loads(Path(workspace_path("evaluation","r2_bpi2012_hybrid_final_v1_result.json")).read_text());s=json.loads(Path(workspace_path("evaluation","r2_sepsis_normalized_final_v1_result.json")).read_text())
    return {"bpi2012":b["arms"]["snn_hybrid"]["frozen_test"]["prediction_trace_sha256"],"sepsis":s["arms"]["snn_normalized"]["frozen_test"]["prediction_trace_sha256"]}
def _run(dataset):
    if dataset=="bpi2012":
        traces=load_bpi();split=bpi_split;route=EventRouteConfig(include_calendar_route=True);hybrid=NormalizedHybridConfig(routing_mode="absolute",base_probability_cap=.75,local_score_margin=.10,
            minimum_base_support=16,max_active=8,max_classes=24,explicit_neurons=True,compact_event_units=True)
    else:
        traces=load_sepsis();split=sepsis_split;route=EventRouteConfig(include_calendar_route=False);hybrid=NormalizedHybridConfig(routing_mode="normalized",normalized_threshold=.6223091976516634,
            minimum_base_support=16,max_active=7,max_classes=16,explicit_neurons=True,compact_event_units=True)
    labels=tuple(sorted({e.activity for t in traces if split(t)!="frozen_test" for e in t.events}));engine=BoundedEventStreamEngine(labels,route_config=route,hybrid_config=hybrid);rows=[];restored=False;checkpoint_sha=None
    for trace in traces:
        if split(trace)=="frozen_test" and not restored:
            state=engine.state_dict();encoded=json.dumps(state,sort_keys=True,separators=(",",":")).encode();checkpoint_sha=hashlib.sha256(encoded).hexdigest();engine=BoundedEventStreamEngine.from_state_dict(json.loads(encoded));restored=True
        for index in range(len(trace.events)-1):
            receipt=engine.predict(trace.events,index);target=trace.events[index+1].activity
            if split(trace)=="frozen_test":rows.append((target,receipt.predicted))
            engine.observe(receipt,target)
    digest=hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest();state=engine.state_dict()
    return {"predictions":len(rows),"prediction_trace_sha256":digest,"checkpoint_restored":restored,"checkpoint_sha256":checkpoint_sha,
        "routes":len(state["encoder"]["routes"]),"weights":len(state["hybrid"]["weights"]),"contexts":len(state["hybrid"]["contexts"])}
def main():
    accepted=_accepted();results={dataset:_run(dataset) for dataset in ("bpi2012","sepsis")};checks={dataset:{"accepted_trace":row["prediction_trace_sha256"]==accepted[dataset],"checkpoint_restored":row["checkpoint_restored"]} for dataset,row in results.items()}
    report={"schema":"sara-event-stream-engine-replay-v1","accepted":accepted,"results":results,"checks":checks,"passed":all(all(v.values()) for v in checks.values())}
    output=Path(ensure_parent_directory(workspace_path("evaluation","event_stream_engine_replay_v1.json")));output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"output":str(output),"passed":report["passed"],"checks":checks}));return 0
if __name__=="__main__":raise SystemExit(main())
