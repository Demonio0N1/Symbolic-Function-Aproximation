#!/usr/bin/env bash
# ============================================================================
# setup.sh — instalación automática de symreg (motor + notebook + interfaz web)
#
# Uso:
#   ./setup.sh                 # instala en .venv/ y verifica todo
#   ./setup.sh --cpu           # fuerza torch CPU (sin CUDA)
#   ./setup.sh --solo-verificar  # no instala nada: solo comprueba el entorno
#
# Al terminar deja listo:
#   .venv/                     entorno virtual con todas las dependencias
#   ./webapp/run.sh            interfaz web    -> http://localhost:8000
#   jupyter notebook           demo            -> FunctionAprox_GPU_Symbolic_AOS_PLUS.ipynb
# ============================================================================
set -u
cd "$(dirname "$0")"

ROJO='\033[0;31m'; VERDE='\033[0;32m'; AMARILLO='\033[1;33m'; NC='\033[0m'
ok()    { echo -e "${VERDE}  ✔ $*${NC}"; }
aviso() { echo -e "${AMARILLO}  ⚠ $*${NC}"; }
fallo() { echo -e "${ROJO}  ✘ $*${NC}"; exit 1; }

FORZAR_CPU=0; SOLO_VERIFICAR=0
for arg in "$@"; do
  case "$arg" in
    --cpu) FORZAR_CPU=1 ;;
    --solo-verificar) SOLO_VERIFICAR=1 ;;
    *) echo "argumento desconocido: $arg"; exit 2 ;;
  esac
done

echo "== 1/6 · Requisitos del sistema =="

# --- python >= 3.10 ---
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
      PY="$cand"; break
    fi
  fi
done
[ -n "$PY" ] || fallo "se necesita Python >= 3.10 (instala python3.10+ y reintenta)"
ok "Python: $($PY --version 2>&1)  ($PY)"

# --- venv y pip disponibles ---
"$PY" -m venv --help >/dev/null 2>&1 || fallo "falta el módulo venv (en Debian/Ubuntu: sudo apt install python3-venv)"
ok "módulo venv disponible"

# --- GPU NVIDIA (opcional) ---
GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU=1
  ok "GPU NVIDIA detectada: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  aviso "sin GPU NVIDIA visible: se instalará torch para CPU (el motor funciona igual, más lento)"
fi
[ "$FORZAR_CPU" = "1" ] && { GPU=0; aviso "--cpu: forzando instalación de torch para CPU"; }

if [ "$SOLO_VERIFICAR" = "0" ]; then
  echo "== 2/6 · Entorno virtual (.venv) =="
  if [ ! -d .venv ]; then
    "$PY" -m venv .venv || fallo "no se pudo crear .venv"
    ok ".venv creado"
  else
    ok ".venv ya existe (se reutiliza)"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip -q || fallo "no se pudo actualizar pip"
  ok "pip $(pip --version | awk '{print $2}')"

  echo "== 3/6 · PyTorch =="
  if [ "$GPU" = "1" ]; then
    # la rueda por defecto de PyPI en Linux trae soporte CUDA 12
    pip install -q "torch>=2.2" || fallo "no se pudo instalar torch (GPU)"
  else
    pip install -q "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu \
      || fallo "no se pudo instalar torch (CPU)"
  fi
  ok "torch instalado"

  echo "== 4/6 · Resto de dependencias (requirements.txt) =="
  pip install -q -r requirements.txt || fallo "fallo instalando requirements.txt"
  ok "dependencias instaladas"
else
  echo "== 2-4/6 · (--solo-verificar: sin instalación) =="
  if [ -d .venv ]; then source .venv/bin/activate; fi
fi

echo "== 5/6 · Verificación de requerimientos =="
python - <<'EOF' || exit 1
import importlib, sys

faltan = []
versiones = {}
for mod in ["numpy", "sympy", "scipy", "matplotlib", "torch", "fastapi", "uvicorn", "pytest"]:
    try:
        m = importlib.import_module(mod)
        versiones[mod] = getattr(m, "__version__", "?")
    except ImportError:
        faltan.append(mod)
if faltan:
    print("  ✘ faltan módulos:", ", ".join(faltan)); sys.exit(1)
for k, v in versiones.items():
    print(f"  ✔ {k} {v}")

import torch
if torch.cuda.is_available():
    print(f"  ✔ CUDA activa: {torch.cuda.get_device_name(0)}")
else:
    print("  ⚠ torch sin CUDA: el motor correrá en CPU")

# el paquete importa y el motor evoluciona (humo de ~5 s)
import numpy as np
import symreg
symreg.config.set_seed(0)
X = np.linspace(-3, 3, 120).astype(np.float32)
Y = (2*X + np.sin(X)).astype(np.float32)
cu, cb = symreg.build_catalogs(["sin", "abs", "neg"], ["add", "mul"])
best, fit, h, a, hof = symreg.evolve(X, Y, cu, cb, pop_size=40, generations=12, verbose=False)
assert fit < 10, fit
f_sym, xs = symreg.to_sympy(best)
print(f"  ✔ symreg {symreg.__version__} operativo en {symreg.info()}  (humo: fit={fit:.3g})")
EOF
[ $? -eq 0 ] || fallo "la verificación falló (revisa los mensajes de arriba)"

echo "== 6/6 · Tests =="
if python -m pytest tests/ -q --no-header 2>/dev/null | tail -1; then
  ok "suite de tests ejecutada"
else
  aviso "pytest devolvió errores: revisa 'python -m pytest tests/ -q'"
fi

echo
echo -e "${VERDE}Instalación completa.${NC} Cómo usar:"
echo "  · Interfaz web:   ./webapp/run.sh        ->  http://localhost:8000"
echo "  · Notebook demo:  source .venv/bin/activate && jupyter notebook FunctionAprox_GPU_Symbolic_AOS_PLUS.ipynb"
echo "  · Benchmarks:     source .venv/bin/activate && python benchmarks.py"
echo "  · Tests:          source .venv/bin/activate && pytest tests/ -q"
