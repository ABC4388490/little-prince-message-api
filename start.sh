#!/bin/sh
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="/app/.venv/bin/python"
fi
# Dependencies are installed during the image build, not when a sleeping service wakes.
exec "$PY" -m uvicorn asgi:app --host 0.0.0.0 --port "${PORT:-5000}"
