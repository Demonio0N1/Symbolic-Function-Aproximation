# Tests de soporte multivariable (Fase 5.2)
import random

import numpy as np


def test_floatinput_indices(eng):
    # x1 debe leer la segunda columna; el caso univariable sigue intacto
    G = np.stack([np.linspace(0, 1, 10), np.linspace(10, 20, 10)], axis=1).astype(np.float32)
    x2 = eng.to_device(G)
    e = eng.Add(eng.FloatInput(0), eng.FloatInput(1))
    got = eng.to_numpy(e.eval_backend(x2)).reshape(-1)
    assert np.allclose(got, G[:, 0] + G[:, 1], rtol=1e-6)
    # univariable: (n,1)
    x1 = eng.to_device(G[:, :1])
    got1 = eng.to_numpy(eng.FloatInput(0).eval_backend(x1)).reshape(-1)
    assert np.allclose(got1, G[:, 0], rtol=1e-6)


def test_interprete_plano_multivar(eng):
    G = np.random.default_rng(0).uniform(-2, 2, size=(50, 2)).astype(np.float32)
    x2 = eng.to_device(G)
    y = eng.to_device((G[:, 0]*G[:, 1] + np.sin(G[:, 0])).reshape(-1, 1).astype(np.float32))
    e = eng.Add(eng.Mul(eng.FloatInput(0), eng.FloatInput(1)), eng.Sin(eng.FloatInput(0)))
    f_clasica = eng.fitness(e, x2, y, alpha=0.0)
    f_lote = float(eng.eval_population_batch([e], x2, y, alpha=0.0)[0])
    assert f_clasica < 1e-10 and abs(f_clasica - f_lote) < 1e-9


def test_roundtrip_multivar(eng):
    e = eng.Add(eng.Mul(eng.FloatInput(0), eng.FloatInput(1)), eng.Cos(eng.FloatInput(1)))
    f_sym, xs = eng.to_sympy(e)
    assert isinstance(xs, list) and len(xs) == 2
    assert eng.check_sympy_roundtrip(e, lo=-2.0, hi=2.0) is True


def test_evolve_multivar_smoke(eng):
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    rng = np.random.default_rng(0)
    G = rng.uniform(-2, 2, size=(300, 2)).astype(np.float32)
    Y = (G[:, 0]*G[:, 1] + np.sin(G[:, 0])).astype(np.float32)
    try:
        best, fit, h, a, hof = eng.evolve(G, Y, eng.CAT_UN, eng.CAT_BIN, pop_size=120,
                                          generations=60, max_depth=4, verbose=False, hof_k=5)
        # debe al menos acercarse claramente (var(Y) ~ 2.4)
        assert fit < 0.5*float(np.var(Y)), fit
        assert eng.config.N_VARS == 2
    finally:
        eng.config.N_VARS = 1   # evolve fija el global; lo restauramos para otros tests
