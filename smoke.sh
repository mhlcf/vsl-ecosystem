#!/usr/bin/env bash
# M5 integration smoke: boot both API servers and hit real endpoints.
set -euo pipefail
cd "$(dirname "$0")"
PY="$PWD/venv/bin/python"

cleanup() { kill "${P2:-}" "${P3:-}" 2>/dev/null || true; }
trap cleanup EXIT

echo "== booting servers =="
(cd part2-speech-to-sign && "$PY" -m backend.app --port 8001 >/tmp/p2.log 2>&1) & P2=$!
(cd part3-gamification  && "$PY" -m backend.app --port 8002 >/tmp/p3.log 2>&1) & P3=$!
sleep 4

echo "== part2 /health =="
curl -sf http://localhost:8001/health && echo
echo "== part2 translate-text =="
curl -sf -X POST http://localhost:8001/api/v1/translate-text \
  -H 'Content-Type: application/json' -d '{"text":"Tôi không thích cà phê","language":"vi"}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);d.pop("controls",None);print(json.dumps(d,ensure_ascii=False,indent=1)[:600])'
echo "== part3 /health =="
curl -sf http://localhost:8002/health && echo
echo "== part3 start-session =="
L1=$(curl -sf -X POST http://localhost:8002/api/v1/game/start-session \
  -H 'Content-Type: application/json' -d '{"user_id":1,"level_id":1}')
echo "$L1" | python3 -m json.tool | head -12
SID=$(echo "$L1" | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
echo "== part3 submit-pose =="
PAYLOAD=$(python3 -c '
import json
seq = [[0.0]*201 for _ in range(60)]
print(json.dumps({"session_id":"'$SID'","frame_data":{"sequence":seq}}))
')
curl -sf -X POST http://localhost:8002/api/v1/game/submit-pose \
  -H 'Content-Type: application/json' -d "$PAYLOAD" \
  | python3 -c 'import sys,json;print(json.dumps(json.load(sys.stdin),ensure_ascii=False)[:300])'
echo "== part3 stats =="
curl -sf http://localhost:8002/api/v1/users/1/stats | python3 -m json.tool | head -8
echo "== SMOKE OK =="