# Tests de las islas distribuidas (Fase 7): serie (SerialComm), reparto de islas,
# y una corrida real con mpirun -np 2 si hay mpi4py + mpirun.
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest

from symreg.mpi import SerialComm, _island_block


def test_reparto_islas_contiguo():
    for n, size in [(4, 1), (4, 4), (8, 3), (5, 2)]:
        bloques = [_island_block(n, r, size) for r in range(size)]
        assert bloques[0][0] == 0 and bloques[-1][1] == n
        assert all(b[1] > b[0] for b in bloques)                 # >= 1 isla por rango
        assert all(bloques[i][1] == bloques[i+1][0] for i in range(size-1))


def test_rank_device_sin_cuda_es_cpu(eng):
    import torch
    if torch.cuda.is_available():
        pytest.skip("con CUDA el dispositivo depende del rango local")
    assert eng.rank_device(SerialComm()).type == "cpu"
    assert eng.set_device("cpu").type == "cpu"


def test_evolve_islands_mpi_serie(eng):
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    X = np.linspace(-2, 2, 120).astype(np.float32)
    Y = (X**2 + X).astype(np.float32)
    try:
        best, fit, hist, aos, hof = eng.evolve_islands_mpi(
            X, Y, eng.CAT_UN, eng.CAT_BIN, n_islands=2, migrate_every=10, generations=20,
            pop_size=60, verbose=False, comm=SerialComm(), finetune_steps=20, val_fraction=0.2)
        assert np.isfinite(fit) and len(hist) == 2 and hof.items
        assert fit < 0.5*float(np.var(Y)), fit
        assert eng.check_sympy_roundtrip(best, lo=-2, hi=2) in (True, None)
        with pytest.raises(ValueError):
            eng.evolve_islands_mpi(X, Y, eng.CAT_UN, eng.CAT_BIN, n_islands=0, comm=SerialComm())
    finally:
        eng.config.N_VARS = 1


def test_datos_solo_en_rango0_con_serialcomm(eng):
    # X/Y None fuera del rango 0 solo tiene sentido con MPI real; en serie debe fallar claro
    with pytest.raises(Exception):
        eng.evolve_islands_mpi(None, None, eng.CAT_UN, eng.CAT_BIN, generations=1, comm=SerialComm())


_PROG = r"""
import json, sys, numpy as np, random, torch
import symreg
from mpi4py import MPI
comm = MPI.COMM_WORLD; rank = comm.Get_rank()
X = Y = None
if rank == 0:
    X = np.linspace(-2, 2, 120).astype(np.float32); Y = (X**2 + X).astype(np.float32)
best, fit, hist, aos, hof = symreg.evolve_islands_mpi(
    X, Y, symreg.CAT_UN, symreg.CAT_BIN, n_islands=3, migrate_every=10, generations=20,
    pop_size=60, verbose=False, finetune_steps=10, val_fraction=0.2, comm=comm)
open(sys.argv[1] + f"/rank{rank}.json", "w").write(json.dumps(dict(
    rank=rank, fit=float(fit), expr=best.to_text(), n_hof=len(hof.items), hist=hist)))
"""


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="sin mpirun")
def test_mpirun_dos_rangos(tmp_path):
    pytest.importorskip("mpi4py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prog = tmp_path / "prog.py"; prog.write_text(_PROG)
    cmd = ["mpirun", "--oversubscribe", "-np", "2", sys.executable, str(prog), str(tmp_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0 and "oversubscribe" in (r.stderr or "").lower():
        cmd.remove("--oversubscribe")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    filas = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    assert sorted(f["rank"] for f in filas) == [0, 1]
    # todos los rangos devuelven el MISMO resultado fusionado
    assert filas[0]["expr"] == filas[1]["expr"] and filas[0]["fit"] == filas[1]["fit"]
    assert filas[0]["hist"] == filas[1]["hist"] and len(filas[0]["hist"]) == 2
    assert np.isfinite(filas[0]["fit"])
