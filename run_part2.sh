#!/usr/bin/env bash
# Launch the Part 2 Speech-to-Sign GUI (PyQt6 + keypoint avatar).
set -euo pipefail
cd "$(dirname "$0")/part2-speech-to-sign"
# faster-whisper/ctranslate2 needs the pip-shipped CUDA 12 libs
# (libcusparse.so.12, libcublas, ...), which live under
# venv/.../site-packages/nvidia/*/lib and are NOT on the system loader path
# (this box has the NVIDIA driver but no CUDA toolkit installed).
_NV_LIB="../venv/lib/python3.12/site-packages/nvidia"
for _d in "$_NV_LIB"/*/lib; do
    if [ -d "$_d" ]; then
        LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH
exec ../venv/bin/python -m frontend.main