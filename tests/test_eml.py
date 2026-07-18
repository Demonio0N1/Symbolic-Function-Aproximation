# Tests del operador EML (Fase 4): eml(u,v) = exp(u) - ln(v)
import numpy as np


def test_eml_semantica_numerica(eng):
    # eml_ debe coincidir con exp_safe(a) - log_(b) y con exp(a) - ln(|b|+1e-8)
    x = eng.to_device(np.linspace(-2.0, 2.0, 64).reshape(-1, 1))
    e = eng.Eml(eng.FloatInput(), eng.Add(eng.FloatInput(), eng.Constant(3.0)))
    got = eng.to_numpy(e.eval_backend(x)).reshape(-1)
    xs = np.linspace(-2.0, 2.0, 64)
    esperado = np.exp(xs) - np.log(np.abs(xs + 3.0) + 1e-8)
    assert np.allclose(got, esperado, rtol=1e-5, atol=1e-6)


def test_eml_roundtrip_sympy(eng):
    # forma expandida exp/ln: evaluable y numéricamente fiel al motor
    e = eng.Eml(eng.Mul(eng.Constant(0.5), eng.FloatInput()),
                eng.Add(eng.FloatInput(), eng.Constant(4.0)))
    assert eng.check_sympy_roundtrip(e, lo=-2.0, hi=2.0) is True


def test_eml_forma_compacta(eng):
    import sympy as sp
    e = eng.Eml(eng.FloatInput(), eng.FloatInput())
    old = eng.config.EML_SYMPY_EXPAND
    try:
        eng.config.EML_SYMPY_EXPAND = False
        f, xs = eng.to_sympy(e)
        assert "eml" in str(f)
        eng.config.EML_SYMPY_EXPAND = True
        f2, xs = eng.to_sympy(e)
        assert f2.has(sp.exp) and f2.has(sp.log)
    finally:
        eng.config.EML_SYMPY_EXPAND = old


def test_eml_en_interprete_plano(eng):
    # el intérprete por lotes debe evaluar eml igual que la ruta recursiva
    x = eng.to_device(np.linspace(0.5, 3.0, 50).reshape(-1, 1))
    y = eng.to_device(np.zeros((50, 1), dtype=np.float32))
    e = eng.Eml(eng.Sin(eng.FloatInput()), eng.Add(eng.FloatInput(), eng.Constant(1.0)))
    f_clasica = eng.fitness(e, x, y)
    f_lote = float(eng.eval_population_batch([e], x, y)[0])
    assert abs(f_clasica - f_lote) < 1e-6 * max(1.0, abs(f_clasica))


def test_eml_pure_train_recupera(eng):
    # objetivo generado por una estructura eml de profundidad 1: el entrenamiento
    # multi-start por lotes debe bajar el MSE claramente
    import torch
    if False:  # el paquete es torch-only (CPU o CUDA)
        return
    torch.manual_seed(0)
    np.random.seed(0)
    X = np.linspace(-2.0, 2.0, 200).astype(np.float32)
    Y = (np.exp(0.8*X + 0.1) - np.log(np.abs(1.5*X + 4.0) + 1e-8)).astype(np.float32)
    spec, n_leaves = eng._eml_full_structure(1)
    tree, mse = eng.eml_pure_train(spec, n_leaves, X, Y, n_starts=64, steps=400,
                                   lr=1e-2, verbose=False, use_compile=False)
    # umbral holgado: los RNG de torch difieren entre CPU y CUDA (var(Y) ~ 2.9)
    assert np.isfinite(mse) and mse < 5e-2, mse
    assert "eml(" in tree.to_text()
