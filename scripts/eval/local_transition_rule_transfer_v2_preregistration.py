#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/"src"
if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path
P=Path(processed_data_path("benchmark_fixtures","local_transition_rule_transfer_v2.json"));H="c79206dc8bde4d6ef8b0fb593a6a3da2982d718ffb70d89f2c492076da9d3082"
def validate():
 raw=P.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if digest!=H:raise ValueError("Frozen contextual transition protocol changed")
 p=json.loads(raw);i=p["fresh_identity"];audit=p["pre_execution_audit"];b=p["boundaries"]
 checks={"schema":p["schema"]=="sara-local-transition-rule-transfer-preregistration-v2","arms":tuple(p["arms"])==("L_contextual_categorical","R_contextual_relation","S_contextual_composition"),"contexts":len(p["rule_contexts"])==5,"five_seeds":len(i["seeds"])==5,"symbol_disjoint":not(set(i["training_symbols"])&set(i["development_symbols"])) and i["symbol_overlap"]==0,"signature_unique_required":audit["maximum_labels_per_signature"]==1,"no_future_fields":audit["future_or_evaluator_fields_allowed"] is False,"held_out_closed":i["held_out_consumed"] is False and b["v2_held_out_consumed"] is False,"prior_closed":b["v1_held_out_used"] is False,"no_backward":p["resource_budgets"]["maximum_backward_events"]==0,"one_attempt":p["resource_budgets"]["maximum_tuning_attempts"]==1,"production_closed":b["production_authorized"] is False}
 if not all(checks.values()):raise ValueError("Contextual transition protocol invalid")
 report={"schema":"sara-local-transition-rule-transfer-v2-preregistration-validation-v1","protocol_sha256":digest,"checks":checks,"passed":True,"candidate_implemented":False,"signature_audit_executed":False,"held_out_consumed":False,"production_authorized":False};out=Path(ensure_parent_directory(workspace_path("evaluation","local_transition_rule_transfer_v2_preregistration.json")));out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return {"output":str(out),**report}
if __name__=="__main__":print(json.dumps(validate(),sort_keys=True))
