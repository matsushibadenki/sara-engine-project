#!/usr/bin/env python3
"""Execute the registered compact-event backend equivalence and cost gate."""
from __future__ import annotations
import hashlib,json,os,random,statistics,subprocess,sys
from pathlib import Path
from sara_engine.utils.project_paths import ensure_parent_directory,processed_data_path,workspace_path
PROTOCOL_PATH=processed_data_path("benchmark_fixtures","compact_event_backend_v1.json");PROTOCOL_SHA256="aa8995873f9bf825a4b8bdf123caa362d9c2abec9f65b3a43deece8e20a4dc63"
def _load():
 raw=Path(PROTOCOL_PATH).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:raise ValueError("Frozen compact backend protocol changed")
 return json.loads(raw)
def _sample(dataset,mode):
 env=dict(os.environ);env["PYTHONHASHSEED"]="0";p=subprocess.run([sys.executable,"scripts/eval/normalized_hybrid_cpu_worker.py","--dataset",dataset,"--mode",mode],text=True,capture_output=True,env=env,check=False)
 if p.returncode:raise RuntimeError(p.stderr[-1000:])
 return json.loads(p.stdout)
def main():
 protocol=_load();output=Path(ensure_parent_directory(workspace_path("evaluation","compact_event_backend_v1_result.json")));marker=Path(ensure_parent_directory(workspace_path("evaluation","compact_event_backend_v1_attempt.json")))
 if marker.exists() or output.exists():raise RuntimeError("Registered compact backend attempt already consumed")
 runtime_hash=hashlib.sha256(Path("src/sara_engine/learning/normalized_hybrid.py").read_bytes()).hexdigest();marker.write_text(json.dumps({"protocol_sha256":PROTOCOL_SHA256,"implemented_runtime_sha256":runtime_hash},indent=2)+"\n")
 order=[(d,m) for d in protocol["datasets"] for m in ("explicit_neurons","compact_events")]*protocol["cost_gate"]["independent_process_repetitions"];random.Random(protocol["cost_gate"]["randomized_order_seed"]).shuffle(order)
 samples=[_sample(d,m) for d,m in order];expected=json.loads(Path(workspace_path("evaluation","normalized_hybrid_runtime_replay_v1.json")).read_text())["expected"];summaries={};checks={}
 for dataset in protocol["datasets"]:
  selected={mode:[r for r in samples if r["dataset"]==dataset and r["mode"]==mode] for mode in ("explicit_neurons","compact_events")}
  summaries[dataset]={mode:{"median_cpu_ms":statistics.median(r["process_cpu_ms"] for r in rows),"median_state_bytes":statistics.median(r["state_bytes"] for r in rows)} for mode,rows in selected.items()}
  cpu_ratio=summaries[dataset]["compact_events"]["median_cpu_ms"]/summaries[dataset]["explicit_neurons"]["median_cpu_ms"];state_ratio=summaries[dataset]["compact_events"]["median_state_bytes"]/summaries[dataset]["explicit_neurons"]["median_state_bytes"]
  checks[dataset]={"accepted_trace":all(r["prediction_trace_sha256"]==expected[dataset] for rows in selected.values() for r in rows),
   "mode_equivalence":len({r["prediction_trace_sha256"] for rows in selected.values() for r in rows})==1,"cpu_gate":cpu_ratio<=protocol["cost_gate"]["maximum_compact_to_explicit_median_cpu_ratio"],
   "state_gate":state_ratio<=protocol["cost_gate"]["maximum_compact_runtime_state_to_explicit_ratio"]};summaries[dataset]["ratios"]={"compact_to_explicit_cpu":cpu_ratio,"compact_to_explicit_state":state_ratio}
 report={"schema":"sara-compact-event-backend-result-v1","protocol_sha256":PROTOCOL_SHA256,"implemented_runtime_sha256":runtime_hash,"samples":samples,"summaries":summaries,"checks":checks,
  "passed":all(all(v.values()) for v in checks.values()),"physical_energy_measured":False};output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"output":str(output),"passed":report["passed"],"checks":checks,"summaries":summaries}));return 0
if __name__=="__main__":raise SystemExit(main())
