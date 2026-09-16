#!/usr/bin/env python3
"""Execute the registered component profile for the normalized-hybrid runtime."""
from __future__ import annotations
import cProfile,hashlib,json,pstats
from pathlib import Path
from sara_engine.utils.project_paths import ensure_output_directory,ensure_parent_directory,processed_data_path,workspace_path
try:
    from normalized_hybrid_runtime_replay import _expected,_run
except ModuleNotFoundError:
    from scripts.eval.normalized_hybrid_runtime_replay import _expected,_run

PROTOCOL_PATH=processed_data_path("benchmark_fixtures","normalized_hybrid_profile_v1.json")
PROTOCOL_SHA256="1140029c2977ce61b2c65a3c228fe64453e0ff9804e3c21b61aae177b632e8de"
TARGETS={"load_traces","encode","predict","_activate","step","add_input_to_branch","observe","route_prediction","route_normalized"}

def _load_protocol():
    raw=Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:raise ValueError("Frozen component profile protocol changed")
    return json.loads(raw)

def _extract(profile_path):
    stats=pstats.Stats(str(profile_path));components={name:{"primitive_calls":0,"total_calls":0,"self_seconds":0.0,"cumulative_seconds":0.0} for name in TARGETS}
    top=[]
    for (filename,line,name),(cc,nc,tt,ct,_) in stats.stats.items():
        top.append({"function":f"{Path(filename).name}:{line}:{name}","self_seconds":tt,"cumulative_seconds":ct,"calls":nc})
        if name in components:
            row=components[name];row["primitive_calls"]+=cc;row["total_calls"]+=nc;row["self_seconds"]+=tt;row["cumulative_seconds"]+=ct
    return {"total_calls":stats.total_calls,"primitive_calls":stats.prim_calls,"total_profile_seconds":stats.total_tt,
        "components":components,"top_by_cumulative":sorted(top,key=lambda row:row["cumulative_seconds"],reverse=True)[:25]}

def main():
    protocol=_load_protocol();marker=Path(ensure_parent_directory(workspace_path("evaluation","normalized_hybrid_profile_v1_attempt.json")));output=Path(ensure_parent_directory(workspace_path("evaluation","normalized_hybrid_profile_v1_result.json")))
    if output.exists():raise RuntimeError("Registered component profile attempt already completed")
    identity={"protocol_sha256":PROTOCOL_SHA256,"runtime_sha256":protocol["runtime_sha256"]}
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise RuntimeError("Component profile attempt identity changed")
    else:marker.write_text(json.dumps(identity,indent=2)+"\n")
    profile_dir=Path(ensure_output_directory(workspace_path("profiles","normalized_hybrid_v1")));expected=_expected();results={}
    for dataset in protocol["datasets"]:
        results[dataset]={}
        for mode in protocol["modes"]:
            path=profile_dir/f"{dataset}-{mode}.prof";profiler=cProfile.Profile();profiler.enable();run=_run(dataset,mode=="explicit_neurons");profiler.disable();profiler.dump_stats(path)
            results[dataset][mode]={"run":run,"profile":_extract(path),"profile_path":str(path)}
    checks={dataset:{"accepted_trace":all(results[dataset][mode]["run"]["prediction_trace_sha256"]==expected[dataset] for mode in protocol["modes"]),
        "mode_equivalence":len({results[dataset][mode]["run"]["prediction_trace_sha256"] for mode in protocol["modes"]})==1} for dataset in protocol["datasets"]}
    activation_excess={dataset:results[dataset]["explicit_neurons"]["profile"]["components"]["_activate"]["cumulative_seconds"]-
        results[dataset]["scalar"]["profile"]["components"]["_activate"]["cumulative_seconds"] for dataset in protocol["datasets"]}
    report={"schema":"sara-normalized-hybrid-component-profile-result-v1","protocol_sha256":PROTOCOL_SHA256,"results":results,"checks":checks,
        "activation_cumulative_excess_seconds":activation_excess,"compact_event_unit_candidate":all(value>0 for value in activation_excess.values()),
        "passed":all(all(value.values()) for value in checks.values()),"physical_energy_measured":False}
    output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"output":str(output),"passed":report["passed"],"activation_excess":activation_excess,
        "compact_event_unit_candidate":report["compact_event_unit_candidate"]}));return 0
if __name__=="__main__":raise SystemExit(main())
