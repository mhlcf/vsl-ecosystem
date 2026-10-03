#!/usr/bin/env bash
# Render UI screenshots for both apps (offscreen) into docs/screenshots/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/venv/bin/python"
"$PY" tools/make_part3_screenshots.py
"$PY" tools/make_part2_screenshots.py
echo "== verifying =="
ls -la docs/screenshots/