#!/usr/bin/env python3
"""Replay both accepted real streams through the generic normalized-hybrid runtime."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
from sara_engine.evaluation.bpi2012_data import load_traces as load_bpi,split_name as bpi_split
from sara_engine.evaluation.r2_bpi2012_development import EventRouteEncoder
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.evaluation.r2_sepsis_hybrid import SepsisRouteEncoder
from sara_engine.evaluation.sepsis_data import load_traces as load_sepsis,split_name as sepsis_split
from sara_engine.learning.normalized_hybrid import BoundedNormalizedHybrid,NormalizedHybridConfig
from sara_engine.utils.project_paths import ensure_parent_directory,workspace_path

def _expected():
    bpi=json.loads(Path(workspace_path("evaluation","r2_bpi2012_hybrid_final_v1_result.json")).read_text())
    sepsis=json.loads(Path(workspace_path("evaluation","r2_sepsis_normalized_final_v1_result.json")).read_text())
    return {"bpi2012":bpi["arms"]["snn_hybrid"]["frozen_test"]["prediction_trace_sha256"],
            "sepsis":sepsis["arms"]["snn_normalized"]["frozen_test"]["prediction_trace_sha256"]}

def _run(dataset,explicit,compact=False):
    if dataset=="bpi2012":
        traces=load_bpi();split=bpi_split;parent=json.loads(Path("data/processed/benchmark_fixtures/r2_bpi2012_model_v1.json").read_text())
        encoder=EventRouteEncoder(parent,spiking=False);config=NormalizedHybridConfig(routing_mode="absolute",normalized_threshold=.6223091976516634,
            base_probability_cap=.75,local_score_margin=.10,minimum_base_support=16,max_active=8,max_classes=24,explicit_neurons=explicit,compact_event_units=compact)
    else:
        traces=load_sepsis();split=sepsis_split;encoder=SepsisRouteEncoder(spiking=False);config=NormalizedHybridConfig(routing_mode="normalized",
            normalized_threshold=.6223091976516634,minimum_base_support=16,max_active=7,max_classes=16,explicit_neurons=explicit,compact_event_units=compact)
    labels=tuple(sorted({e.activity for t in traces if split(t)!="frozen_test" for e in t.events}));model=BoundedNormalizedHybrid(labels,config);rows=[];started=time.process_time_ns();events=0
    for trace in traces:
        for index in range(len(trace.events)-1):
            current=trace.events[index].activity;previous=trace.events[index-1].activity if index else "<BOS>"
            receipt=model.predict(encoder.encode(trace,index),context=(previous,current));target=trace.events[index+1].activity
            if split(trace)=="frozen_test":rows.append((target,receipt.predicted))
            model.observe(receipt,target);events+=1
    cpu_ms=(time.process_time_ns()-started)/1e6;digest=hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest()
    return {"predictions":len(rows),"prediction_trace_sha256":digest,"cpu_ms":cpu_ms,"cpu_us_per_event":cpu_ms*1000/events,
        "state_bytes":_deep_size(model),"routes":len(model.state_dict()["routes"]),"weights":len(model.state_dict()["weights"]),"events":events}

def main():
    expected=_expected();results={dataset:{mode:_run(dataset,mode=="explicit_neurons") for mode in ("explicit_neurons","scalar")} for dataset in ("bpi2012","sepsis")}
    checks={}
    for dataset,modes in results.items():
        checks[dataset]={"accepted_trace":modes["explicit_neurons"]["prediction_trace_sha256"]==expected[dataset],
            "mode_equivalence":modes["explicit_neurons"]["prediction_trace_sha256"]==modes["scalar"]["prediction_trace_sha256"]}
    report={"schema":"sara-normalized-hybrid-runtime-replay-v1","expected":expected,"results":results,"checks":checks,
        "passed":all(all(v.values()) for v in checks.values()),"physical_energy_measured":False}
    output=Path(ensure_parent_directory(workspace_path("evaluation","normalized_hybrid_runtime_replay_v1.json")));output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(output),"checks":checks,"passed":report["passed"]}));return 0
if __name__=="__main__":raise SystemExit(main())
