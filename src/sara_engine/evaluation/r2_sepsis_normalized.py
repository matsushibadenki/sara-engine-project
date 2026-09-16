"""Training-only normalized-router diagnostic on Sepsis Cases."""

from __future__ import annotations
import hashlib, json, random
from pathlib import Path

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.evaluation.r2_bpi2012_development import _metrics
from sara_engine.evaluation.r2_sepsis_hybrid import SepsisRouteEncoder, _base
from sara_engine.evaluation.sepsis_data import load_traces, split_name
from sara_engine.learning.normalized_router import calibrate_override_threshold, normalized_override_score, route_normalized
from sara_engine.learning.sparse_multiclass import BoundedSparseMulticlassReadout, SparseMulticlassConfig
from sara_engine.utils.project_paths import processed_data_path

PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "r2_sepsis_normalized_router_v1.json")
PROTOCOL_SHA256 = "77f11279defb04befdd877315be3288c08dc9fd089e3c60e78e3ff5e67e8f83e"

def load_protocol():
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256: raise ValueError("Frozen normalized-router protocol changed")
    return json.loads(raw)

def _collect(protocol, arm):
    from collections import Counter, defaultdict
    traces = [trace for trace in load_traces() if split_name(trace) == "training"]
    labels = tuple(sorted({event.activity for trace in traces for event in trace.events})); boundary = int(len(traces) * .70)
    scalar = arm == "scalar_normalized"; encoder = SepsisRouteEncoder(spiking=not scalar, constant_gap=arm == "constant_gap_normalized")
    inherited = protocol["inherited_unchanged"]
    learner = BoundedSparseMulticlassReadout(labels, SparseMulticlassConfig(inherited["learning_rate"], inherited["weight_cap"], 4096, 7, 16))
    tables = defaultdict(Counter); targets = [t.events[i+1].activity for t in traces for i in range(len(t.events)-1)]; feedback = list(targets)
    if arm == "shuffled_outcome_normalized": random.Random(protocol["shuffled_outcome_seed"]).shuffle(feedback)
    feedback_index = 0; records = {"calibration": [], "confirmation": []}
    for trace_index, trace in enumerate(traces):
        fold = "calibration" if trace_index < boundary else "confirmation"
        for index in range(len(trace.events)-1):
            event=trace.events[index]; previous=trace.events[index-1].activity if index else "<BOS>"; table=tables[(previous,event.activity)]
            base_predicted,base_probabilities=_base(table,labels); receipt=learner.predict(encoder.encode(trace,index)); target=trace.events[index+1].activity
            records[fold].append({"target":target,"base_predicted":base_predicted,"base_probabilities":base_probabilities,
                "base_support":sum(table.values()),"local_predicted":receipt.predicted,"local_scores":dict(receipt.scores)})
            learner.observe(receipt,feedback[feedback_index] if arm=="shuffled_outcome_normalized" else target);feedback_index+=1;table[target]+=1
    return {"labels":labels,"records":records,"routes":len(encoder.routes),"neurons":len(encoder.units),
        "weight_entries":len(learner.snapshot()["weights"]),"state_bytes":_deep_size((encoder,learner,tables))}

def _evaluate(records,labels,threshold):
    rows=[];overrides=0
    for record in records:
        if threshold is None: predicted=record["base_predicted"];probabilities=record["base_probabilities"]
        else:
            decision=route_normalized(labels=labels,base_predicted=record["base_predicted"],base_probabilities=record["base_probabilities"],
                base_support=record["base_support"],local_predicted=record["local_predicted"],local_scores=record["local_scores"],threshold=threshold)
            predicted=decision.predicted;probabilities=dict(decision.probabilities);overrides+=int(decision.overridden)
        target=record["target"];rows.append({"target":target,"predicted":predicted,"brier":sum((probabilities[l]-int(l==target))**2 for l in labels)})
    metrics=_metrics(rows,labels,set());metrics.update({"overrides":overrides,"override_rate":overrides/len(rows),
        "prediction_trace_sha256":hashlib.sha256(json.dumps([(r["target"],r["predicted"]) for r in rows],separators=(",",":")).encode()).hexdigest()});return metrics

def run_diagnostic(protocol):
    arms={arm:_collect(protocol,arm) for arm in protocol["arms"] if arm!="online_second_order_transition"};candidate=arms["snn_normalized"];labels=candidate["labels"]
    calibration_scores=[normalized_override_score(base_predicted=r["base_predicted"],base_probabilities=r["base_probabilities"],base_support=r["base_support"],
        local_predicted=r["local_predicted"],local_scores=r["local_scores"]) for r in candidate["records"]["calibration"]]
    threshold=calibrate_override_threshold(calibration_scores,protocol["router"]["target_override_fraction_of_all_predictions"])
    confirmation={name:_evaluate(arm["records"]["confirmation"],labels,threshold) for name,arm in arms.items()}
    base=_evaluate(candidate["records"]["confirmation"],labels,None);c=confirmation["snn_normalized"];gate=protocol["confirmation_acceptance"]
    checks={"accuracy":c["top1_accuracy"]-base["top1_accuracy"]>=gate["minimum_top1_gain_over_base"],
        "macro_f1":c["macro_f1"]-base["macro_f1"]>=gate["minimum_macro_f1_gain_over_base"],
        "brier":c["multiclass_brier"]-base["multiclass_brier"]<=gate["maximum_brier_increase_over_base"],
        "timing":c["macro_f1"]-confirmation["constant_gap_normalized"]["macro_f1"]>=gate["minimum_constant_gap_macro_f1_drop"],
        "shuffled":c["macro_f1"]-confirmation["shuffled_outcome_normalized"]["macro_f1"]>=gate["minimum_shuffled_outcome_macro_f1_drop"],
        "override_budget":c["override_rate"]<=gate["maximum_candidate_override_rate"],
        "scalar_equivalence":c["prediction_trace_sha256"]==confirmation["scalar_normalized"]["prediction_trace_sha256"],
        "resources":all(a["routes"]<=4096 and a["weight_entries"]<=65536 and a["state_bytes"]<=16777216 for a in arms.values())}
    return {"schema":"sara-r2-sepsis-normalized-router-result-v1","threshold":threshold,"fold_case_counts":{"calibration":514,"confirmation":221},
        "calibration_override_rate":sum(score is not None and score>=threshold for score in calibration_scores)/len(calibration_scores),
        "confirmation":{"base":base,**confirmation},"resources":{n:{k:a[k] for k in ("routes","neurons","weight_entries","state_bytes")} for n,a in arms.items()},
        "decision":{"checks":checks,"passed":all(checks.values()),"status":"diagnostic_pass" if all(checks.values()) else "diagnostic_negative_result",
            "development_opened":False,"frozen_test_opened":False,"production_authorized":False}}

__all__=["load_protocol","run_diagnostic"]
