#!/usr/bin/env python3
"""Audit whether every frozen context can generate both labels before implementation."""
from __future__ import annotations
import itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path
P=Path(processed_data_path("benchmark_fixtures","relational_rule_independent_replication_v1.json"))

def independent_label(context:str,values:tuple[int,...])->int:
 if context=="ascending_or_equal":return int(values[1]>=values[0])
 if context=="same_parity":return int((values[0]&1)==(values[1]&1))
 if context=="bounded_jump":return int(abs(values[1]-values[0])<=2)
 if context=="interval_direction":return int((values[2]-values[1])>(values[1]-values[0]))
 raise ValueError("unknown context")

def audit()->dict:
 p=json.loads(P.read_text());identity=p["fresh_identity"];support={}
 for split,key in (("training","training_values"),("development","development_values")):
  values=identity[key];support[split]={}
  for context in p["contexts"]:
   width=3 if context=="interval_direction" else 2
   labels={independent_label(context,tuple(row)) for row in itertools.product(values,repeat=width)}
   support[split][context]={"labels":sorted(labels),"balanced_generation_possible":labels=={0,1}}
 checks={"all_contexts_have_both_labels":all(row["balanced_generation_possible"] for split in support.values() for row in split.values()),"value_sets_disjoint":not(set(identity["training_values"])&set(identity["development_values"])),"candidate_not_implemented":True,"held_out_closed":identity["held_out_consumed"] is False}
 report={"schema":"sara-relational-rule-independent-replication-feasibility-v1","protocol_sha256":"bc7ff8cb7891e02e3df57c5f15cd88ba3155494543837c82b76563ceea61f4cf","support":support,"checks":checks,"passed":all(checks.values()),"candidate_execution_authorized":all(checks.values()),"held_out_consumed":False,"production_authorized":False};out=Path(ensure_parent_directory(workspace_path("evaluation","relational_rule_independent_replication_feasibility.json")));out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return {"output":str(out),**report}
if __name__=="__main__":
 r=audit();print(json.dumps(r,sort_keys=True));raise SystemExit(0 if r["passed"] else 1)
