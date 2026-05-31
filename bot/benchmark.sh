#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "compare" ]]; then
  shift
  python3 bot/benchmark_hybrid_compare.py "$@"
else
  python3 bot/benchmark_suite.py "$@"
fi
