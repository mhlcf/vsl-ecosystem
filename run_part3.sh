#!/usr/bin/env bash
# Launch the Part 3 gamified VSL learning GUI (PyQt6 + webcam).
set -euo pipefail
cd "$(dirname "$0")/part3-gamification"
exec ../venv/bin/python -m frontend.main