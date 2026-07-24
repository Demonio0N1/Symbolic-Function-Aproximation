#!/usr/bin/env bash
# Lanza la interfaz web local de symreg -> http://localhost:8000
# Usa .venv/ si existe (creado por ./setup.sh); si no, el entorno conda "ml";
# si no, el python del sistema.
cd "$(dirname "$0")/.."

if [ -x .venv/bin/uvicorn ]; then
  exec .venv/bin/uvicorn webapp.app:app --host 127.0.0.1 --port 8000 "$@"
elif command -v conda >/dev/null 2>&1 && conda env list | grep -qE '^ml\s'; then
  exec conda run --no-capture-output -n ml uvicorn webapp.app:app --host 127.0.0.1 --port 8000 "$@"
else
  exec python3 -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000 "$@"
fi
