#!/usr/bin/env bash
#
# run-local.sh — run AquaVision on this machine and expose it publicly (free)
#
# Serves the FULL working site (frontend + YOLO analysis, computed locally) and
# tunnels it to a public https://<random>.trycloudflare.com URL via Cloudflare.
# No account, no billing. Stop with Ctrl-C.
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8002}"
VENV=".venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "❌ $VENV not found. Create it first:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "❌ cloudflared not installed.  brew install cloudflared"
  exit 1
fi

cleanup() { echo; echo "Shutting down…"; kill "${SERVER_PID:-}" "${TUNNEL_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "▶ Starting AquaVision server on http://localhost:$PORT (loading YOLO models…)"
"$VENV/bin/uvicorn" server:app --host 0.0.0.0 --port "$PORT" > server.log 2>&1 &
SERVER_PID=$!

# wait until the server answers /health
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 1
  if [[ $i -eq 60 ]]; then echo "❌ server didn't come up — see server.log"; exit 1; fi
done
echo "✓ Server is up. Models: $(curl -s http://localhost:$PORT/health)"

echo "▶ Opening public Cloudflare tunnel…"
cloudflared tunnel --url "http://localhost:$PORT" > tunnel.log 2>&1 &
TUNNEL_PID=$!

# grab the public URL from cloudflared's log
URL=""
for i in $(seq 1 30); do
  URL=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' tunnel.log | head -1 || true)
  [[ -n "$URL" ]] && break
  sleep 1
done

echo
echo "════════════════════════════════════════════════════════════"
if [[ -n "$URL" ]]; then
  echo "  🌐 LIVE (analysis runs on THIS machine):"
  echo "     $URL"
else
  echo "  ⚠ Tunnel URL not detected yet — check tunnel.log"
fi
echo "  Local:  http://localhost:$PORT"
echo "  Ctrl-C to stop."
echo "════════════════════════════════════════════════════════════"

wait
