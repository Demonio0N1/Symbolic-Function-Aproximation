# Suite de benchmarks (Fase 5.1): objetivos propios + Nguyen + Keijzer +
# Feynman de una variable, con las tres configuraciones comparadas.
import json
import random
import time

import numpy as np

from . import config
from .backend import to_device
from .eml import eml_pure_fit
from .evolve import evolve_islands, fitness
from .finetune import fine_tune_hof, recognize_constants
from .ops import build_catalogs
from .tree import all_subnodes

TARGETS = {
    "propia":     (lambda x: 2*x**3 + 3*np.sin(x) + 1,          -5.0, 5.0, 400),
    "nguyen5":    (lambda x: np.sin(x**2)*np.cos(x) - 1.0,      -1.0, 1.0, 200),
    "nguyen6":    (lambda x: np.sin(x) + np.sin(x + x**2),      -1.0, 1.0, 200),
    "nguyen7":    (lambda x: np.log(x + 1) + np.log(x**2 + 1),   0.0, 2.0, 200),
    "keijzer1":   (lambda x: 0.3*x*np.sin(2*np.pi*x),           -1.0, 1.0, 200),
    "feyn_gauss": (lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi),  -3.0, 3.0, 300),   # I.6.20a
    "feyn_edens": (lambda x: 0.5*x**2,                          -2.0, 2.0, 200),   # II.8.31 (eps=1)
    "feyn_rel":   (lambda x: 1.0/np.sqrt(1 - x**2),             -0.9, 0.9, 200),   # I.48.2 (m=c=1)
}

CONFIGS = ["catalogo", "catalogo+eml", "eml_puro"]


def correr_benchmark(nombre, cfg, seed, gen=150):
    """Corre un objetivo con una configuración y una semilla; devuelve un dict."""
    import torch
    fn, lo, hi, npts = TARGETS[nombre]
    X = np.linspace(lo, hi, npts).astype(np.float32)
    Y = fn(X.astype(np.float64)).astype(np.float32)

    config.set_seed(seed)
    x_d = to_device(X.reshape(-1, 1)); y_d = to_device(Y.reshape(-1, 1))
    t0 = time.time()

    if cfg == "eml_puro":
        best, _ = eml_pure_fit(X, Y, steps=1000, evo_generations=40, verbose=False)
        best = recognize_constants(best, X, Y, alpha=1e-3)
    else:
        bins = ["add", "sub", "mul", "div", "pow"] + (["eml"] if cfg == "catalogo+eml" else [])
        cat_un, cat_bin = build_catalogs(["sin", "cos", "log", "sqrt", "abs", "neg"], bins)
        curriculum = [
            {"until": 40, "depth": 3, "un": ["sin", "cos", "abs", "neg", "sqrt"], "bin": ["add", "sub", "mul"]},
            {"until": 90, "depth": 4, "un": ["sin", "cos", "log", "sqrt", "abs", "neg"], "bin": ["add", "sub", "mul", "div"]},
            {"until": gen, "depth": 5, "un": ["sin", "cos", "log", "sqrt", "abs", "neg"], "bin": bins},
        ]
        best, fit, h, aos, hof = evolve_islands(
            X, Y, cat_un, cat_bin, n_islands=4, migrate_every=50, migrants=3,
            generations=gen, pop_size=160, verbose=False, hof_k=10,
            max_depth=5, elite=10, p_mut=0.6, p_xover=0.3, alpha=1e-3,
            aos_params=dict(tau=0.8, lr=0.4, decay=0.98, prune_every=40,
                            min_keep_un=3, min_keep_bin=3, prune_threshold=0.04),
            use_macros=True, p_macro=0.25, use_hint=False,
            sample_schedule=[(40, 0.25), (90, 0.6), (130, 0.85), (gen, 1.0)],
            curriculum=curriculum, hill_climb_steps=3, restart_patience=60)
        hof = fine_tune_hof(hof, X, Y, steps=250, lr=5e-3, verbose=False, alpha=1e-3)
        _fb, best = hof.items[0]
        best = recognize_constants(best, X, Y, alpha=1e-3)

    import torch as _t
    mse = float(_t.mean((best.eval_backend(x_d) - y_d)**2).item())
    return dict(target=nombre, config=cfg, seed=seed, mse=mse,
                nodos=len(all_subnodes(best)), t=round(time.time() - t0, 1),
                recuperada=bool(mse < 1e-6), expr=best.to_text()[:160])


def tabla_markdown(resultados):
    lineas = ["| Objetivo | Config | Recuperación | MSE mediana | Nodos med. | t medio (s) |",
              "|----------|--------|:------------:|------------:|-----------:|------------:|"]
    targets = sorted({r["target"] for r in resultados}, key=lambda t: list(TARGETS).index(t))
    for tg in targets:
        for cf in CONFIGS:
            rs = [r for r in resultados if r["target"] == tg and r["config"] == cf]
            if not rs:
                continue
            rec = sum(r["recuperada"] for r in rs)
            lineas.append(
                f"| {tg} | {cf} | {rec}/{len(rs)} | {np.median([r['mse'] for r in rs]):.3g} "
                f"| {int(np.median([r['nodos'] for r in rs]))} | {np.mean([r['t'] for r in rs]):.0f} |")
    return "\n".join(lineas)
