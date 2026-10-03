#!/usr/bin/env bash
# Launch Part 1 real-time sign recognition (webcam).
set -euo pipefail
cd "$(dirname "$0")/part1-sign-classifier"
exec ../venv/bin/python real_time_predict.py
