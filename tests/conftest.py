# Los tests usan el paquete symreg directamente (Fase 6).
# El fixture se llama "eng" por compatibilidad con los tests de fases previas,
# cuando el motor se cargaba desde el notebook.
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import symreg  # noqa: E402


@pytest.fixture(scope="session")
def eng():
    return symreg
