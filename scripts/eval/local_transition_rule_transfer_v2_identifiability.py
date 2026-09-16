#!/usr/bin/env python3
"""Materialize and audit observable signatures before v2 candidate execution."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/"src"
if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
from sara_engine.evaluation.local_transition_rule_transfer import generate_transition_episodes,TransitionRuleLearner
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path
PARENT=Path(processed_data_path("benchmark_fixtures","local_transition_rule_transfer_v2.json"));PARENT_SHA="c79206dc8bde4d6ef8b0fb593a6a3da2982d718ffb70d89f2c492076da9d3082"
MATERIAL=Path(processed_data_path("benchmark_fixtures","local_transition_rule_transfer_v2_materialization.json"));MATERIAL_SHA="7681511e5b63d0a3fdf10d3e6bafa7129fd5b610431a80ab11459a9f13cbfffd"
CONTEXT={"relative_direction":0,"relative_equality":1,"relative_distance":2,"two_transition_composition":3,"categorical_control":4}
def _signature(row):
 rel=[TransitionRuleLearner._relation(row.symbols[i-1],v) for i,v in enumerate(row.symbols) if i];ctx=CONTEXT[row.family]
 if row.family=="relative_direction":return (ctx,rel[0][0])
 if row.family=="relative_equality":return (ctx,rel[0][0])
 if row.family=="relative_distance":return (ctx,rel[0][1])
 if row.family=="two_transition_composition":return (ctx,rel[0][0],rel[1][0])
 return (ctx,row.symbols[-1])
def audit():
 if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError("Parent protocol changed")
 if hashlib.sha256(MATERIAL.read_bytes()).hexdigest()!=MATERIAL_SHA:raise ValueError("Materialization protocol changed")
 p=json.loads(PARENT.read_text());m=json.loads(MATERIAL.read_text());i=p["fresh_identity"];all_rows={}
 for split,symbols,count_key in (("training",i["training_symbols"],"training_count_per_context_per_seed"),("development",i["development_symbols"],"development_count_per_context_per_seed")):
  rows=generate_transition_episodes(seeds=i["seeds"],symbols=symbols,count_per_family=m[count_key],split=split,namespace=i["namespace"]);all_rows[split]=rows
 labels=defaultdict(set);balance=defaultdict(set)
 for split,rows in all_rows.items():
  for row in rows:labels[(split,_signature(row))].add(row.label);parts=row.identity.split(":");balance[(split,parts[2],row.family)].add(row.label)
 train_ids={r.identity for r in all_rows["training"]};dev_ids={r.identity for r in all_rows["development"]};checks={"signature_unique":all(len(v)==1 for v in labels.values()),"balanced":all(v=={0,1} for v in balance.values()),"namespace":all(r.identity.startswith(i["namespace"]+":") for rows in all_rows.values() for r in rows),"identity_disjoint":not(train_ids&dev_ids),"symbol_disjoint":not(set(i["training_symbols"])&set(i["development_symbols"])),"held_out_not_materialized":m["audit"]["held_out_materialized"] is False}
 report={"schema":"sara-local-transition-rule-transfer-v2-identifiability-v1","parent_protocol_sha256":PARENT_SHA,"materialization_sha256":MATERIAL_SHA,"counts":{k:len(v) for k,v in all_rows.items()},"signature_count":len(labels),"checks":checks,"passed":all(checks.values()),"candidate_execution_authorized":all(checks.values()),"held_out_consumed":False,"production_authorized":False};out=Path(ensure_parent_directory(workspace_path("evaluation","local_transition_rule_transfer_v2_identifiability.json")));out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return {"output":str(out),**report}
if __name__=="__main__":
 r=audit();print(json.dumps(r,sort_keys=True));raise SystemExit(0 if r["passed"] else 1)
