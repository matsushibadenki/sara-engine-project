#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path
P=Path(processed_data_path("benchmark_fixtures","local_transition_rule_zero_shot_v1.json"));H="5216b01ae192e82aad94c2bb8add001bb9ab42f79d00ae67ffd695189146efc5"
def validate():
 raw=P.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if digest!=H:raise ValueError("Frozen zero-shot protocol changed")
 p=json.loads(raw);i=p["fresh_identity"];e=p["evaluation_boundary"];checks={"schema":p["schema"]=="sara-local-transition-rule-zero-shot-preregistration-v1","arms":p["arms"]==["L_contextual_categorical","R_contextual_relation","S_contextual_composition"],"five_seeds":len(i["seeds"])==5,"symbol_disjoint":not(set(i["training_symbols"])&set(i["development_symbols"])) and i["symbol_overlap"]==0,"zero_shot_frozen":e["zero_shot_development_updates"] is False,"digest_before_online":e["zero_shot_digest_finalized_before_online_phase"] is True,"separate_online_copy":e["online_adaptation_uses_separate_learner_copy"] is True,"online_non_gating":e["online_metrics_used_for_acceptance"] is False,"held_out_closed":i["held_out_consumed"] is False,"one_attempt":p["resource_budgets"]["maximum_tuning_attempts"]==1,"no_backward":p["resource_budgets"]["maximum_backward_events"]==0,"production_closed":p["boundaries"]["production_authorized"] is False}
 if not all(checks.values()):raise ValueError("Zero-shot protocol invalid")
 report={"schema":"sara-local-transition-rule-zero-shot-preregistration-validation-v1","protocol_sha256":digest,"checks":checks,"passed":True,"candidate_implemented":False,"held_out_consumed":False,"production_authorized":False};out=Path(ensure_parent_directory(workspace_path("evaluation","local_transition_rule_zero_shot_preregistration.json")));out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return {"output":str(out),**report}
if __name__=="__main__":print(json.dumps(validate(),sort_keys=True))
