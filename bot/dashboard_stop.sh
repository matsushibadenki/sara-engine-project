#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f workspace/autobot/dashboard.pid ]]; then
  pid="$(cat workspace/autobot/dashboard.pid)"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid"
    echo "dashboard stopped pid=$pid"
  else
    echo "dashboard process not running: $pid"
  fi
else
  echo "dashboard pid file not found"
fi
