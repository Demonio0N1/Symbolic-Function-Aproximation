#!/usr/bin/env bash
# Lanza la interfaz web local de symreg -> http://localhost:8000
cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n ml uvicorn webapp.app:app --host 127.0.0.1 --port 8000 "$@"
