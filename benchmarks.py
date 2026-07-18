# Suite de benchmarks (Fase 5.1) — corre en GPU.
# Compara: catálogo actual | catálogo + eml | modo EML puro, con >= 5 semillas,
# reportando tasa de recuperación exacta, MSE, complejidad y tiempo.
#
# Uso:
#   python benchmarks.py                        # suite completa (5 semillas)
#   python benchmarks.py --seeds 0,1 --targets propia,nguyen5 --configs catalogo
#
# Resultados: benchmarks_results.json + tabla markdown por stdout (y BENCHMARKS.md).
import argparse
import json
import pathlib
import random
import time
import types

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent
NB = REPO / "FunctionAprox_GPU_Symbolic_AOS_PLUS.ipynb"


def cargar_motor():
    """Carga las celdas de definiciones del notebook (sin ejecutar la demo)."""
    import matplotlib
    matplotlib.use("Agg")
    nb = json.loads(NB.read_text(encoding="utf-8"))
    chunks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        if "=== Datos de prueba ===" in code:
            break
        chunks.append(code)
    mod = types.ModuleType("symeng")
    exec(compile("\n".join(chunks), str(NB), "exec"), mod.__dict__)
    return mod


# --- Objetivos: propios + estándar (Nguyen, Keijzer, Feynman de 1 variable) ---
TARGETS = {
    "propia":    (lambda x: 2*x**3 + 3*np.sin(x) + 1,                    -5.0, 5.0, 400),
    "nguyen5":   (lambda x: np.sin(x**2)*np.cos(x) - 1.0,                -1.0, 1.0, 200),
    "nguyen6":   (lambda x: np.sin(x) + np.sin(x + x**2),                -1.0, 1.0, 200),
    "nguyen7":   (lambda x: np.log(x + 1) + np.log(x**2 + 1),             0.0, 2.0, 200),
    "keijzer1":  (lambda x: 0.3*x*np.sin(2*np.pi*x),                     -1.0, 1.0, 200),
    # Feynman (una variable, constantes normalizadas):
    "feyn_gauss": (lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi),           -3.0, 3.0, 300),   # I.6.20a
    "feyn_edens": (lambda x: 0.5*x**2,                                   -2.0, 2.0, 200),   # II.8.31 (eps=1)
    "feyn_rel":   (lambda x: 1.0/np.sqrt(1 - x**2),                      -0.9, 0.9, 200),   # I.48.2 (m=c=1)
}

CONFIGS = ["catalogo", "catalogo+eml", "eml_puro"]


def correr(eng, config, nombre, seed, gen=150):
    import torch
    fn, lo, hi, npts = TARGETS[nombre]
    X = np.linspace(lo, hi, npts).astype(np.float32)
    Y = fn(X.astype(np.float64)).astype(np.float32)

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    x_d = eng.to_device(X.reshape(-1, 1)); y_d = eng.to_device(Y.reshape(-1, 1))
    t0 = time.time()

    if config == "eml_puro":
        best, _ = eng.eml_pure_fit(X, Y, steps=1000, evo_generations=40, verbose=False)
        best = eng.recognize_constants(best, X, Y, alpha=1e-3)
    else:
        bins = ["add", "sub", "mul", "div", "pow"]
        if config == "catalogo+eml":
            bins = bins + ["eml"]
        cat_un, cat_bin = eng.build_catalogs(["sin", "cos", "log", "sqrt", "abs", "neg"], bins)
        curriculum = [
            {"until": 40, "depth": 3, "un": ["sin", "cos", "abs", "neg", "sqrt"], "bin": ["add", "sub", "mul"]},
            {"until": 90, "depth": 4, "un": ["sin", "cos", "log", "sqrt", "abs", "neg"], "bin": ["add", "sub", "mul", "div"]},
            {"until": gen, "depth": 5, "un": ["sin", "cos", "log", "sqrt", "abs", "neg"], "bin": bins},
        ]
        best, fit, h, aos, hof = eng.evolve_islands(
            X, Y, cat_un, cat_bin, n_islands=4, migrate_every=50, migrants=3,
            generations=gen, pop_size=160, verbose=False, hof_k=10,
            max_depth=5, elite=10, p_mut=0.6, p_xover=0.3, alpha=1e-3,
            aos_params=dict(tau=0.8, lr=0.4, decay=0.98, prune_every=40,
                            min_keep_un=3, min_keep_bin=3, prune_threshold=0.04),
            use_macros=True, p_macro=0.25, use_hint=False,
            sample_schedule=[(40, 0.25), (90, 0.6), (130, 0.85), (gen, 1.0)],
            curriculum=curriculum, hill_climb_steps=3, restart_patience=60)
        hof = eng.fine_tune_hof(hof, X, Y, steps=250, lr=5e-3, verbose=False, alpha=1e-3)
        _fb, best = hof.items[0]
        best = eng.recognize_constants(best, X, Y, alpha=1e-3)

    t_total = time.time() - t0
    import torch as _t
    mse = float(_t.mean((best.eval_backend(x_d) - y_d)**2).item())
    n_nodos = len(eng.all_subnodes(best))
    return dict(target=nombre, config=config, seed=seed, mse=mse,
                nodos=n_nodos, t=round(t_total, 1),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--gen", type=int, default=150)
    ap.add_argument("--out", default="benchmarks_results.json")
    args = ap.parse_args()

    eng = cargar_motor()
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (fallback)")

    seeds = [int(s) for s in args.seeds.split(",")]
    resultados = []
    for tg in args.targets.split(","):
        for cf in args.configs.split(","):
            for sd in seeds:
                r = correr(eng, cf, tg, sd, gen=args.gen)
                resultados.append(r)
                print(f"[{tg:10s}|{cf:12s}|s{sd}] mse={r['mse']:.3g} rec={r['recuperada']} "
                      f"nodos={r['nodos']} t={r['t']}s", flush=True)
                json.dump(resultados, open(args.out, "w"), indent=1)

    tabla = tabla_markdown(resultados)
    print("\n" + tabla)
    (REPO / "BENCHMARKS.md").write_text(
        "# Benchmarks (Fase 5)\n\nGPU: RTX 4090 — 5 semillas por celda; "
        "recuperación exacta = MSE < 1e-6 tras fine-tuning + reconocimiento de constantes.\n\n"
        + tabla + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
