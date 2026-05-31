#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 bot/status.py "$@"
