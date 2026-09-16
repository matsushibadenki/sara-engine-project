#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path
P=Path(processed_data_path("benchmark_fixtures","relational_rule_independent_replication_v1.json"));H="bc7ff8cb7891e02e3df57c5f15cd88ba3155494543837c82b76563ceea61f4cf"
def validate():
 raw=P.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if digest!=H:raise ValueError("Frozen independent replication protocol changed")
 p=json.loads(raw);g=p["generator_independence"];i=p["fresh_identity"];e=p["evaluation_boundary"];b=p["boundaries"]
 checks={"schema":p["schema"]=="sara-relational-rule-independent-replication-preregistration-v1","two_arms":p["arms"]==["categorical_zero_shot","contextual_relational_zero_shot"],"new_generator":g["new_generator_required"] and not g["prior_transition_generator_import_allowed"],"separate_evaluator":g["label_evaluator_implemented_separately_from_candidate"],"five_seeds":len(i["seeds"])==5,"values_disjoint":not(set(i["training_values"])&set(i["development_values"])) and i["value_overlap"]==0,"zero_shot":e["development_updates"] is False and e["score_before_any_adaptation"],"unique_pairs":e["development_value_pairs_unique_within_context"],"held_out_closed":i["held_out_consumed"] is False,"composition_non_gating":b["composition_is_non_gating"],"no_backward":p["resource_budgets"]["maximum_backward_events"]==0,"one_attempt":p["resource_budgets"]["maximum_tuning_attempts"]==1,"production_closed":b["production_authorized"] is False}
 if not all(checks.values()):raise ValueError("Independent replication protocol invalid")
 report={"schema":"sara-relational-rule-independent-replication-preregistration-validation-v1","protocol_sha256":digest,"checks":checks,"passed":True,"generator_implemented":False,"audit_executed":False,"held_out_consumed":False,"production_authorized":False};out=Path(ensure_parent_directory(workspace_path("evaluation","relational_rule_independent_replication_preregistration.json")));out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return {"output":str(out),**report}
if __name__=="__main__":print(json.dumps(validate(),sort_keys=True))
