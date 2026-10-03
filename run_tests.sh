#!/usr/bin/env bash
# Run the full test suite for the VSL ecosystem (both parts + shared).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-./venv/bin/python}"
exec "$PY" -m pytest -q "$@"