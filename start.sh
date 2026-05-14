#!/bin/sh
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="/app/.venv/bin/python"
fi
"$PY" -m pip install -q -r requirements.txt && exec "$PY" -m uvicorn asgi:app --host 0.0.0.0 --port "${PORT:-5000}"
