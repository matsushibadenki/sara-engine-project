#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/"src"
if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path
PROTOCOL_PATH=Path(processed_data_path("benchmark_fixtures","local_transition_rule_transfer_v1.json"));PROTOCOL_SHA256="58c1959ced4f3ede26f2bc4c85066b826bb4c7d60cc5219ec29fcce1deb5393e"
def validate():
 raw=PROTOCOL_PATH.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if digest!=PROTOCOL_SHA256:raise ValueError("Frozen transition-rule protocol changed")
 p=json.loads(raw);identity=p["fresh_identity"];checks={"schema":p["schema"]=="sara-local-transition-rule-transfer-preregistration-v1","arms":tuple(p["arms"])==("L_categorical_pair","R_relational_transition","S_relational_composition"),"five_seeds":len(identity["seeds"])==5,"symbol_disjoint":not(set(identity["training_symbols"])&set(identity["development_symbols"])) and identity["symbol_overlap"]==0,"held_out_closed":identity["held_out_consumed"] is False,"one_attempt":p["resource_budgets"]["maximum_tuning_attempts"]==1,"no_backward":p["resource_budgets"]["maximum_backward_events"]==0,"production_closed":p["boundaries"]["production_authorized"] is False}
 if not all(checks.values()):raise ValueError("Transition-rule protocol is invalid")
 report={"schema":"sara-local-transition-rule-transfer-preregistration-validation-v1","protocol_sha256":digest,"checks":checks,"passed":True,"candidate_implemented":False,"held_out_consumed":False,"production_authorized":False};out=Path(ensure_parent_directory(workspace_path("evaluation","local_transition_rule_transfer_preregistration_v1.json")));out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return {"output":str(out),**report}
if __name__=="__main__":print(json.dumps(validate(),sort_keys=True))
