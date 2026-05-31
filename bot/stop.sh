#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f workspace/autobot/bot.pid ]]; then
  pid="$(cat workspace/autobot/bot.pid)"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid"
    echo "stopped pid=$pid"
  else
    echo "process not running: $pid"
  fi
else
  echo "pid file not found"
fi
