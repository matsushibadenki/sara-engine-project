#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p workspace/autobot
mkdir -p workspace/autobot/.mplconfig workspace/autobot/.cache
touch workspace/autobot/alerts.log
if [[ -f workspace/autobot/bot.log ]]; then
  size=$(wc -c < workspace/autobot/bot.log | tr -d ' ')
  if [[ "${size:-0}" -ge 10000000 ]]; then
    mv workspace/autobot/bot.log workspace/autobot/bot.log.1 2>/dev/null || true
    : > workspace/autobot/bot.log
  fi
fi
export MPLCONFIGDIR="$(pwd)/workspace/autobot/.mplconfig"
export XDG_CACHE_HOME="$(pwd)/workspace/autobot/.cache"
nohup python3 bot/autonomous_bot.py > workspace/autobot/bot.log 2>&1 &
echo $! > workspace/autobot/bot.pid
echo "started pid=$(cat workspace/autobot/bot.pid)"
