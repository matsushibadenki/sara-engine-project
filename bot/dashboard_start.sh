#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p workspace/autobot
nohup python3 bot/dashboard.py > workspace/autobot/dashboard.log 2>&1 &
echo $! > workspace/autobot/dashboard.pid
echo "dashboard started pid=$(cat workspace/autobot/dashboard.pid) url=http://127.0.0.1:8765"
