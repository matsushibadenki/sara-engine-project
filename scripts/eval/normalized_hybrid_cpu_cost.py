#!/usr/bin/env python3
"""Execute the registered multi-process normalized-hybrid CPU cost protocol."""
from __future__ import annotations
import hashlib,json,os,random,statistics,subprocess,sys
from pathlib import Path
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path

PROTOCOL_PATH=processed_data_path("benchmark_fixtures","normalized_hybrid_cpu_cost_v1.json")
PROTOCOL_SHA256="1b695699b22e819afade9e4a7538c4fb91c035137a28f722661d2f59f3fb0818"

def _load_protocol():
    raw=Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:raise ValueError("Frozen CPU cost protocol changed")
    return json.loads(raw)

def _sample(dataset,mode):
    environment=dict(os.environ);environment["PYTHONHASHSEED"]="0"
    process=subprocess.run([sys.executable,"scripts/eval/normalized_hybrid_cpu_worker.py","--dataset",dataset,"--mode",mode],
        cwd=Path.cwd(),env=environment,text=True,capture_output=True,check=False)
    if process.returncode!=0:raise RuntimeError(f"CPU worker failed: {process.stderr[-1000:]}")
    return json.loads(process.stdout)

def _summary(values):
    median=statistics.median(values)
    return {"median":median,"minimum":min(values),"maximum":max(values),
        "median_absolute_deviation":statistics.median(abs(value-median) for value in values)}

def main():
    protocol=_load_protocol();marker=Path(ensure_parent_directory(workspace_path("evaluation","normalized_hybrid_cpu_cost_v1_attempt.json")))
    output=Path(ensure_parent_directory(workspace_path("evaluation","normalized_hybrid_cpu_cost_v1_result.json")))
    if output.exists():raise RuntimeError("Registered CPU cost attempt already completed")
    identity={"protocol_sha256":PROTOCOL_SHA256,"runtime_sha256":protocol["runtime_sha256"]}
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise RuntimeError("CPU cost attempt identity changed")
    else:marker.write_text(json.dumps(identity,indent=2)+"\n")
    pairs=[(dataset,mode) for dataset in protocol["datasets"] for mode in protocol["modes"]]
    warmups=[_sample(dataset,mode) for dataset,mode in pairs]
    order=pairs*protocol["repetitions_per_dataset_mode"];random.Random(protocol["order_randomization_seed"]).shuffle(order)
    samples=[_sample(dataset,mode) for dataset,mode in order]
    expected=json.loads(Path(workspace_path("evaluation","normalized_hybrid_runtime_replay_v1.json")).read_text())["expected"]
    summaries={}
    for dataset,mode in pairs:
        selected=[row for row in samples if row["dataset"]==dataset and row["mode"]==mode]
        summaries.setdefault(dataset,{})[mode]={metric:_summary([row[metric] for row in selected]) for metric in ("process_cpu_ms","wall_ms","peak_rss_bytes")}
    checks={"all_processes_exit_zero":True,
        "accepted_traces":all(row["prediction_trace_sha256"]==expected[row["dataset"]] for row in samples),
        "mode_equivalence":all(len({row["prediction_trace_sha256"] for row in samples if row["dataset"]==dataset})==1 for dataset in protocol["datasets"])}
    ratios={dataset:{metric:summaries[dataset]["explicit_neurons"][metric]["median"]/summaries[dataset]["scalar"][metric]["median"]
        for metric in ("process_cpu_ms","wall_ms","peak_rss_bytes")} for dataset in protocol["datasets"]}
    report={"schema":"sara-normalized-hybrid-cpu-cost-result-v1","protocol_sha256":PROTOCOL_SHA256,"warmup_count":len(warmups),
        "sample_count":len(samples),"execution_order":[list(pair) for pair in order],"samples":samples,"summaries":summaries,"explicit_to_scalar_ratios":ratios,
        "checks":checks,"passed":all(checks.values()),"energy":{"status":"unmeasured","reason":"No physical meter was used; powermetrics reports estimates only."}}
    output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"output":str(output),"passed":report["passed"],"ratios":ratios}));return 0
if __name__=="__main__":raise SystemExit(main())
