"""Final frozen-test evaluation for the fixed Sepsis normalized router."""

from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, random, resource, time
from pathlib import Path

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.evaluation.r2_bpi2012_development import _metrics
from sara_engine.evaluation.r2_sepsis_hybrid import SepsisRouteEncoder, _base
from sara_engine.evaluation.sepsis_data import load_traces, split_name
from sara_engine.learning.normalized_router import route_normalized
from sara_engine.learning.sparse_multiclass import BoundedSparseMulticlassReadout, SparseMulticlassConfig
from sara_engine.utils.project_paths import processed_data_path

PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "r2_sepsis_normalized_final_v1.json")
PROTOCOL_SHA256 = "114ccfdacf9747f537d33ff03c85dc56b78f8c7a016b50bd848cb875d1b35707"

def load_protocol():
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen Sepsis normalized final protocol changed")
    return json.loads(raw)

def run_arm(protocol, arm):
    traces = load_traces(); labels = tuple(sorted({e.activity for t in traces if split_name(t) != "frozen_test" for e in t.events}))
    scalar = arm == "scalar_normalized"; encoder = SepsisRouteEncoder(spiking=not scalar, constant_gap=arm == "constant_gap_normalized")
    learner = BoundedSparseMulticlassReadout(labels, SparseMulticlassConfig(protocol["learning_rate"], protocol["weight_cap"], 4096, 7, 16))
    tables = defaultdict(Counter); targets = [t.events[i+1].activity for t in traces for i in range(len(t.events)-1)]; feedback = list(targets)
    if arm == "shuffled_outcome_normalized": random.Random(protocol["shuffled_outcome_seed"]).shuffle(feedback)
    feedback_index=0;rows=[];base_rows=[];overrides=0;latencies=[];max_work=0
    for trace in traces:
        evaluate=split_name(trace)=="frozen_test"
        for index in range(len(trace.events)-1):
            started=time.perf_counter_ns();event=trace.events[index];previous=trace.events[index-1].activity if index else "<BOS>";table=tables[(previous,event.activity)]
            base_predicted,base_probabilities=_base(table,labels);active=encoder.encode(trace,index);receipt=learner.predict(active)
            decision=route_normalized(labels=labels,base_predicted=base_predicted,base_probabilities=base_probabilities,base_support=sum(table.values()),
                local_predicted=receipt.predicted,local_scores=dict(receipt.scores),threshold=protocol["fixed_normalized_score_threshold"],minimum_base_support=protocol["minimum_base_support"])
            target=trace.events[index+1].activity;update=learner.observe(receipt,feedback[feedback_index] if arm=="shuffled_outcome_normalized" else target)
            feedback_index+=1;table[target]+=1;latencies.append((time.perf_counter_ns()-started)/1e6)
            max_work=max(max_work,len(active)*(5 if not scalar else 1)+len(labels)*len(active)+update.updates+16)
            if evaluate:
                probabilities=dict(decision.probabilities);rows.append({"target":target,"predicted":decision.predicted,"brier":sum((probabilities[l]-int(l==target))**2 for l in labels)})
                base_rows.append({"target":target,"predicted":base_predicted,"brier":sum((base_probabilities[l]-int(l==target))**2 for l in labels)});overrides+=int(decision.overridden)
    ordered=sorted(latencies);percentile=lambda q:ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))];state=_deep_size((encoder,learner,tables));rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;b=protocol["resource_budgets"]
    resources={"routes":len(encoder.routes),"neurons":len(encoder.units),"weight_entries":len(learner.snapshot()["weights"]),"state_bytes":state,"max_event_work":max_work,
        "latency_p50_ms":percentile(.5),"latency_p95_ms":percentile(.95),"latency_p99_ms":percentile(.99),"latency_watchdog_ms":max(ordered),"peak_rss_bytes":rss}
    resources["contracts_passed"]=(resources["routes"]<=b["max_routes"] and resources["neurons"]<=b["max_neurons"] and resources["weight_entries"]<=b["max_weight_entries"]
        and state<=b["max_state_bytes"] and max_work<=b["max_event_work"] and resources["latency_p99_ms"]<=b["cpu_latency_p99_ms"]
        and resources["latency_watchdog_ms"]<=b["cpu_latency_watchdog_ms"] and rss<=b["max_peak_rss_bytes"])
    metrics=_metrics(rows,labels,set());metrics.update({"overrides":overrides,"override_rate":overrides/len(rows),
        "prediction_trace_sha256":hashlib.sha256(json.dumps([(r["target"],r["predicted"]) for r in rows],separators=(",",":")).encode()).hexdigest()})
    return {"frozen_test":metrics,"base_frozen_test":_metrics(base_rows,labels,set()),"resources":resources}

def run_final(protocol):
    arms={arm:run_arm(protocol,arm) for arm in protocol["arms"] if arm!="online_second_order_transition"};c=arms["snn_normalized"]["frozen_test"];base=arms["snn_normalized"]["base_frozen_test"];g=protocol["final_acceptance"]
    checks={"accuracy":c["top1_accuracy"]-base["top1_accuracy"]>=g["minimum_top1_gain_over_base"],
        "macro_f1":c["macro_f1"]-base["macro_f1"]>=g["minimum_macro_f1_gain_over_base"],
        "brier":c["multiclass_brier"]-base["multiclass_brier"]<=g["maximum_brier_increase_over_base"],
        "timing":c["macro_f1"]-arms["constant_gap_normalized"]["frozen_test"]["macro_f1"]>=g["minimum_constant_gap_macro_f1_drop"],
        "shuffled_relative":arms["shuffled_outcome_normalized"]["frozen_test"]["macro_f1"]-base["macro_f1"]<=g["maximum_shuffled_macro_f1_gain_over_base"],
        "override_budget":c["override_rate"]<=g["maximum_candidate_override_rate"],
        "scalar_equivalence":c["prediction_trace_sha256"]==arms["scalar_normalized"]["frozen_test"]["prediction_trace_sha256"],
        "resources":all(a["resources"]["contracts_passed"] for a in arms.values())}
    return {"schema":"sara-r2-sepsis-normalized-final-result-v1","arms":arms,"decision":{"checks":checks,"passed":all(checks.values()),
        "status":"final_pass" if all(checks.values()) else "final_negative_result","production_authorized":False}}

__all__=["load_protocol","run_final"]
