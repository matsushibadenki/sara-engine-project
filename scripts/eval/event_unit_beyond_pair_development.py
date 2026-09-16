#!/usr/bin/env python3
"""Run development-only beyond-pair event-unit controls."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/"src"
if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
from sara_engine.evaluation.event_unit_beyond_pair import BeyondPairLearner,generate_beyond_pair,run_beyond_pair
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path

PROTOCOL_PATH=Path(processed_data_path("benchmark_fixtures","event_unit_beyond_pair_v1.json"));PROTOCOL_SHA256="9a8005ed0d9e08adabadbf44249386dba332fdac9e758e1e366281adac6dd1af"

def main()->int:
 raw=PROTOCOL_PATH.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:raise ValueError("Frozen beyond-pair protocol changed")
 p=json.loads(raw);identity=p["fresh_identity"];seeds=identity["seeds"]
 train=generate_beyond_pair(seeds=seeds,count_per_family=identity["training_count_per_family_per_seed"],split="training")
 dev=generate_beyond_pair(seeds=seeds,count_per_family=identity["development_count_per_family_per_seed"],split="development")
 arms={arm:run_beyond_pair(arm,train,dev) for arm in BeyondPairLearner.ARMS}
 controls={
  "history_truncate_T_to_pair":run_beyond_pair("T_bounded_triplet",train,dev,intervention="history_truncate"),
  "event_order_shuffle_T":run_beyond_pair("T_bounded_triplet",train,dev,intervention="event_order_shuffle"),
  "distractor_identity_shuffle_T":run_beyond_pair("T_bounded_triplet",train,dev,intervention="distractor_shuffle"),
  "branch_assignment_shuffle_D":run_beyond_pair("D_triplet_branch",train,dev,intervention="branch_shuffle"),
  "outcome_shuffle_D":run_beyond_pair("D_triplet_branch",train,dev,outcome_shuffle_seed=918991),
  "capacity_matched_pair":run_beyond_pair("P_temporal_pair",train,dev,capacity_reserve_bytes=16000),
 }
 replay={arm:run_beyond_pair(arm,train,dev)["prediction_trace_sha256"] for arm in BeyondPairLearner.ARMS};b=p["resource_budgets"]
 harness={"held_out_closed":True,"replay":all(replay[a]==arms[a]["prediction_trace_sha256"] for a in replay),
  "work":all(r["maximum_event_work"]<=b["maximum_event_work"] for r in arms.values()),"state":all(r["state_bytes"]<=b["maximum_retained_state_bytes"] for r in arms.values()),
  "features":all(r["feature_count"]<=b["maximum_learned_scalars"] for r in arms.values())}
 per_seed={}
 for seed in seeds:
  tr=generate_beyond_pair(seeds=(seed,),count_per_family=identity["training_count_per_family_per_seed"],split="training");dv=generate_beyond_pair(seeds=(seed,),count_per_family=identity["development_count_per_family_per_seed"],split="development")
  scores={a:run_beyond_pair(a,tr,dv) for a in BeyondPairLearner.ARMS};per_seed[str(seed)]={"T_minus_P":scores["T_bounded_triplet"]["ambiguous_accuracy"]-scores["P_temporal_pair"]["ambiguous_accuracy"],"D_minus_T_branch":scores["D_triplet_branch"]["accuracy_by_family"]["branch_specific_conjunction"]-scores["T_bounded_triplet"]["accuracy_by_family"]["branch_specific_conjunction"]}
 a=p["acceptance"];deltas={"triplet":arms["T_bounded_triplet"]["ambiguous_accuracy"]-arms["P_temporal_pair"]["ambiguous_accuracy"],"history_control":arms["T_bounded_triplet"]["ambiguous_accuracy"]-controls["history_truncate_T_to_pair"]["ambiguous_accuracy"],"order_control":arms["T_bounded_triplet"]["ambiguous_accuracy"]-controls["event_order_shuffle_T"]["ambiguous_accuracy"],"branch":arms["D_triplet_branch"]["accuracy_by_family"]["branch_specific_conjunction"]-arms["T_bounded_triplet"]["accuracy_by_family"]["branch_specific_conjunction"],"branch_control":arms["D_triplet_branch"]["accuracy_by_family"]["branch_specific_conjunction"]-controls["branch_assignment_shuffle_D"]["accuracy_by_family"]["branch_specific_conjunction"]}
 checks={"pair_ambiguous":arms["P_temporal_pair"]["ambiguous_accuracy"]<=a["maximum_pair_accuracy_on_ambiguous_families"],"pair_control":arms["P_temporal_pair"]["accuracy_by_family"]["pair_sufficient_control"]>=a["minimum_pair_control_accuracy"],"triplet":deltas["triplet"]>=a["minimum_triplet_gain_on_ambiguous_families"],"history_control":deltas["history_control"]>=a["minimum_targeted_control_drop"],"order_control":deltas["order_control"]>=a["minimum_targeted_control_drop"],"branch":deltas["branch"]>=a["minimum_branch_gain_on_branch_family"],"branch_control":deltas["branch_control"]>=a["minimum_targeted_control_drop"],"capacity":controls["capacity_matched_pair"]["prediction_trace_sha256"]==arms["P_temporal_pair"]["prediction_trace_sha256"],"seed_signs":all(r["T_minus_P"]>0 and r["D_minus_T_branch"]>0 for r in per_seed.values())}
 for group in (arms,controls):
  for row in group.values():row.pop("prediction_rows")
 report={"schema":"sara-event-unit-beyond-pair-development-v1","protocol_sha256":PROTOCOL_SHA256,"arms":arms,"controls":controls,"per_seed":per_seed,"deltas":deltas,"harness_checks":harness,"development_checks":checks,"harness_passed":all(harness.values()),"development_gate_passed":all(checks.values()),"held_out_consumed":False,"production_authorized":False}
 output=Path(ensure_parent_directory(workspace_path("evaluation","event_unit_beyond_pair_development_v1.json")));output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"output":str(output),"harness_passed":report["harness_passed"],"development_gate_passed":report["development_gate_passed"],"deltas":deltas,"checks":checks},sort_keys=True));return 0 if report["harness_passed"] and report["development_gate_passed"] else 1
if __name__=="__main__":raise SystemExit(main())
