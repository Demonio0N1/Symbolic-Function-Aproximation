# Carga el motor definido en el notebook (solo las celdas de definiciones,
# sin ejecutar la demo) y lo expone como fixture de sesión para pytest.
import json
import pathlib
import types

import pytest

NB_PATH = pathlib.Path(__file__).resolve().parents[1] / "FunctionAprox_GPU_Symbolic_AOS_PLUS.ipynb"


def _load_engine():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    chunks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        if "=== Datos de prueba ===" in code:
            break  # a partir de aquí empieza la demo (ejecuta la evolución)
        chunks.append(code)
    mod = types.ModuleType("symeng")
    exec(compile("\n".join(chunks), str(NB_PATH), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="session")
def eng():
    return _load_engine()
