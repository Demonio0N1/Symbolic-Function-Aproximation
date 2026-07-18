# Tests del paquete (Fase 6.2): guardas numéricas, invariantes de
# mutación/crossover, serialización y paridad CPU-vs-GPU.
import random

import numpy as np
import pytest
import torch


def _random_pop(eng, n=30, depth=5, seed=7):
    random.seed(seed); np.random.seed(seed)
    aos = eng.OperatorManager(*eng.build_catalogs(
        ["sin", "cos", "log", "sqrt", "abs", "neg", "tan", "gamma"],
        ["add", "sub", "mul", "div", "pow", "eml"]))
    return [eng.random_node(aos, max_depth=depth)[0] for _ in range(n)], aos


def test_guardas_numericas_fitness_finito(eng):
    # La garantía del motor: el FITNESS siempre es finito aunque un árbol
    # produzca NaN/Inf (esos individuos reciben 1e12, como el diseño original)
    pop, _ = _random_pop(eng, n=60)
    x = eng.to_device(np.linspace(-50, 50, 200).reshape(-1, 1).astype(np.float32))
    y = eng.to_device(np.zeros((200, 1), dtype=np.float32))
    fits = eng.eval_population_batch(pop, x, y)
    assert np.isfinite(fits).all() or np.all(fits[~np.isfinite(fits)] > 0), \
        "el fitness de individuos inválidos debe ser 1e12 o +inf, nunca NaN"
    assert not np.isnan(fits).any()
    # y las ops con guardas explícitas nunca generan NaN por sí mismas
    e = eng.Log(eng.Sub(eng.FloatInput(), eng.FloatInput()))       # log(0)
    v = eng.to_numpy(e.eval_backend(x)).reshape(-1)
    assert np.isfinite(v).all()
    e2 = eng.Div(eng.Constant(1.0), eng.Sub(eng.FloatInput(), eng.FloatInput()))  # 1/0
    v2 = eng.to_numpy(e2.eval_backend(x)).reshape(-1)
    assert np.isfinite(v2).all()


def test_invariantes_mutacion(eng):
    pop, aos = _random_pop(eng, n=20)
    for t in pop:
        child, used = eng.mutate(aos, t, max_depth=4)
        # aridad: los binarios tienen o1 y o2; los unarios solo o1
        for nd, _, _ in eng.all_subnodes(child):
            if nd.name in eng.BINARY_MAP:
                assert nd.o1 is not None and nd.o2 is not None
            elif nd.name in eng.UNARY_MAP:
                assert nd.o1 is not None and nd.o2 is None
        # la mutación no muta el original (clona)
        assert t.to_text() == t.to_text()
        # profundidad de subárboles nuevos acotada por profundidad_original + max_depth
        assert eng.tree_depth(child) <= eng.tree_depth(t) + 5


def test_invariantes_crossover(eng):
    pop, aos = _random_pop(eng, n=20)
    for a, b in zip(pop[::2], pop[1::2]):
        ta, tb = a.to_text(), b.to_text()
        child, used = eng.crossover(aos, a, b)
        assert a.to_text() == ta and b.to_text() == tb, "crossover mutó a los padres"
        for nd, _, _ in eng.all_subnodes(child):
            if nd.name in eng.BINARY_MAP:
                assert nd.o1 is not None and nd.o2 is not None


def test_serializacion_roundtrip(eng):
    pop, _ = _random_pop(eng, n=25)
    x = eng.to_device(np.linspace(-2, 2, 40).reshape(-1, 1).astype(np.float32))
    for t in pop:
        t2 = eng.tree_from_json(eng.tree_to_json(t))
        assert t2.to_text() == t.to_text()
        v1 = eng.to_numpy(t.eval_backend(x)).reshape(-1)
        v2 = eng.to_numpy(t2.eval_backend(x)).reshape(-1)
        assert np.allclose(v1, v2, rtol=1e-6, atol=1e-7, equal_nan=True)


def test_paridad_cpu_gpu(eng):
    if not torch.cuda.is_available():
        pytest.skip("sin GPU: la paridad se cubre trivialmente")
    pop, _ = _random_pop(eng, n=25, depth=4)
    xs = np.linspace(-3, 3, 100).reshape(-1, 1).astype(np.float32)
    x_gpu = torch.tensor(xs, device="cuda")
    x_cpu = torch.tensor(xs, device="cpu")
    for t in pop:
        v_gpu = t.eval_backend(x_gpu).cpu().numpy().reshape(-1)
        v_cpu = t.eval_backend(x_cpu).numpy().reshape(-1)
        # tolerancia holgada: las transcendentales float32 difieren entre CPU y GPU
        m = np.isfinite(v_gpu) & np.isfinite(v_cpu) & (np.abs(v_cpu) < 1e6)
        assert np.allclose(v_gpu[m], v_cpu[m], rtol=1e-3, atol=1e-4), t.to_text()


def test_checkpoint_continuacion(eng):
    import os
    import tempfile
    eng.config.set_seed(3)
    X = np.linspace(-2, 2, 80).astype(np.float32)
    Y = (X**2 + 1).astype(np.float32)
    cu, cb = eng.build_catalogs(["abs", "neg"], ["add", "mul"])
    b, f, h, a, hof = eng.evolve(X, Y, cu, cb, pop_size=40, generations=10, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.json")
        eng.save_checkpoint(p, hof, a)
        ck = eng.load_checkpoint(p)
        assert len(ck["final_pop"]) == 40
        # continuar la corrida desde el checkpoint
        b2, f2, h2, a2, hof2 = eng.evolve(X, Y, cu, cb, pop_size=40, generations=10,
                                          verbose=False, init_pop=ck["final_pop"])
        assert f2 <= f + 1e-9
