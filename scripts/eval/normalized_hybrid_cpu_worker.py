#!/usr/bin/env python3
"""Run one isolated normalized-hybrid CPU cost sample."""
from __future__ import annotations
import argparse,json,resource,time
from normalized_hybrid_runtime_replay import _run

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=("bpi2012","sepsis"),required=True);parser.add_argument("--mode",choices=("explicit_neurons","compact_events","scalar"),required=True);args=parser.parse_args()
    cpu_start=time.process_time_ns();wall_start=time.perf_counter_ns();result=_run(args.dataset,args.mode!="scalar",args.mode=="compact_events")
    result.update({"dataset":args.dataset,"mode":args.mode,"process_cpu_ms":(time.process_time_ns()-cpu_start)/1e6,
        "wall_ms":(time.perf_counter_ns()-wall_start)/1e6,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(json.dumps(result,separators=(",",":")));return 0
if __name__=="__main__":raise SystemExit(main())
