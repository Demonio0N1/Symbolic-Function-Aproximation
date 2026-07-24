# Interfaz web local del proyecto: configura los datos y la evolución desde el
# navegador, corre en TU máquina (usa la GPU vía symreg) y muestra progreso en
# vivo, resultado simbólico, gráficas y descargas.
#
# Arranque:  conda run -n ml uvicorn webapp.app:app --port 8000
# (o ./webapp/run.sh)  ->  http://localhost:8000
import base64
import io
import pathlib
import sys
import threading
import time
import traceback

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import symreg
from symreg import config
from symreg.benchmarks import TARGETS

app = FastAPI(title="symreg web")

ESTATICOS = pathlib.Path(__file__).parent / "static"
SALIDAS = pathlib.Path(__file__).parent / "salidas"
SALIDAS.mkdir(exist_ok=True)

# Estado global de la corrida (una a la vez: la GPU es una)
ESTADO = {
    "corriendo": False,
    "cancelar": False,
    "progreso": None,     # {gen, total, best_fit, elapsed, hist:[...]}
    "resultado": None,
    "error": None,
    "csv": None,          # datos subidos: {"X": ..., "Y": ..., "nombre": ...}
}


# ---------- utilidades ----------
def _fig_a_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _construir_datos(p):
    """Devuelve (X, Y) según la fuente elegida: preset, expresión o CSV."""
    fuente = p.get("fuente", "preset")
    if fuente == "csv":
        if ESTADO["csv"] is None:
            raise ValueError("No hay CSV subido")
        return ESTADO["csv"]["X"], ESTADO["csv"]["Y"]
    lo = float(p.get("lo", -5)); hi = float(p.get("hi", 5))
    npts = int(p.get("npts", 400))
    X = np.linspace(lo, hi, npts).astype(np.float32)
    if fuente == "preset":
        nombre = p.get("preset", "propia")
        fn = TARGETS[nombre][0]
        Y = fn(X.astype(np.float64)).astype(np.float32)
    else:  # expresión personalizada evaluada con numpy (espacio restringido)
        expr = p.get("expr", "2*x**3 + 3*sin(x) + 1")
        entorno = {"__builtins__": {}, "x": X.astype(np.float64), "np": np,
                   "pi": np.pi, "e": np.e,
                   "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
                   "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "tanh": np.tanh,
                   "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan}
        Y = np.asarray(eval(expr, entorno), dtype=np.float64).astype(np.float32)  # noqa: S307 (herramienta local)
        if Y.shape != X.shape:
            Y = np.full_like(X, float(Y))
    ruido = float(p.get("ruido", 0.0))
    if ruido > 0:
        Y = (Y + np.random.default_rng(0).normal(0, ruido, size=Y.shape)).astype(np.float32)
    return X, Y


def _trabajador(p):
    """Corre la evolución por bloques (progreso + cancelación) y deja el resultado."""
    t0 = time.time()
    try:
        seed = int(p.get("seed", 2))
        config.set_seed(seed)
        config.EML_MODE = p.get("eml_mode", "off")
        config.SELECTION = p.get("selection", "alpha")
        config.AOS_UPDATE_CHUNK = int(p.get("aos_chunk", 1))

        X, Y = _construir_datos(p)
        X2 = np.asarray(X).reshape(len(X), -1)
        multivar = X2.shape[1] > 1

        POP = int(p.get("pop", 160)); GEN = int(p.get("gen", 250))
        ISLAS = max(1, int(p.get("islas", 4))); DEPTH = int(p.get("depth", 5))
        ALPHA = float(p.get("alpha", 1e-3))
        VAL = float(p.get("val_fraction", 0.2))
        RESTART = int(p.get("restart_patience", 80))
        BLOQUE = max(10, int(p.get("bloque", 25)))   # gens entre actualizaciones de progreso

        unarios = p.get("unarios") or ["sin", "cos", "log", "sqrt", "abs", "neg"]
        binarios = p.get("binarios") or ["add", "sub", "mul", "div", "pow"]
        if config.EML_MODE == "operator" and "eml" not in binarios:
            binarios = binarios + ["eml"]
        cat_un, cat_bin = symreg.build_catalogs(unarios, binarios)

        # split de validación único (los bloques no deben re-partir)
        X_val = Y_val = None
        Xtr, Ytr = X2, np.asarray(Y)
        if VAL > 0:
            perm = np.random.permutation(len(X2))
            n_val = max(4, int(len(X2)*VAL))
            X_val, Y_val = X2[perm[:n_val]], np.asarray(Y)[perm[:n_val]]
            Xtr, Ytr = X2[perm[n_val:]], np.asarray(Y)[perm[n_val:]]

        comunes = dict(
            max_depth=DEPTH, elite=10, p_mut=0.6, p_xover=0.3, alpha=ALPHA,
            aos_params=dict(tau=0.8, lr=0.4, decay=0.98, prune_every=40,
                            min_keep_un=3, min_keep_bin=3, prune_threshold=0.04),
            use_macros=not multivar, p_macro=0.25,
            use_hint=not multivar, hint_fraction=0.35, hint_jitter=0.25,
            # priors suaves (evolve ignora los que no estén en el catálogo)
            unary_priors={"sin": 0.6, "cos": 0.1, "log": 0.1},
            binary_priors={"add": 0.3, "mul": 0.1, "pow": 0.4},
            sample_schedule=[(40, 0.25), (90, 0.6), (130, 0.85), (GEN, 1.0)],
            # currículum como el de la demo, intersectado con el catálogo elegido
            curriculum=[
                {"until": 40, "depth": min(3, DEPTH),
                 "un": [u for u in ["sin", "cos", "abs", "neg", "sqrt"] if u in unarios] or unarios,
                 "bin": [b for b in ["add", "sub", "mul"] if b in binarios] or binarios},
                {"until": 90, "depth": min(4, DEPTH),
                 "un": [u for u in ["sin", "cos", "log", "sqrt", "abs", "neg"] if u in unarios] or unarios,
                 "bin": [b for b in ["add", "sub", "mul", "div"] if b in binarios] or binarios},
                {"until": GEN, "depth": DEPTH, "un": unarios, "bin": binarios},
            ],
            hill_climb_steps=3, hof_k=10, restart_patience=RESTART, verbose=False)

        # modo EML puro: sin bucle evolutivo por bloques
        if config.EML_MODE == "pure":
            if multivar:
                raise ValueError("El modo EML puro solo soporta una variable")
            best, mse_puro = symreg.eml_pure_fit(Xtr.reshape(-1), Ytr, verbose=False)
            hof = symreg.HallOfFame(k=1)
            x_tr = symreg.to_device(Xtr); y_tr = symreg.to_device(Ytr.reshape(-1, 1))
            hof.update(best, symreg.fitness(best, x_tr, y_tr, alpha=ALPHA))
            aos = None
        else:
            # bucle por bloques con migración en anillo (progreso + cancelación)
            pops = [None]*ISLAS; aoss = [None]*ISLAS
            hof = symreg.HallOfFame(k=10)
            hist = []
            done = 0
            while done < GEN:
                if ESTADO["cancelar"]:
                    break
                bloque = min(BLOQUE, GEN - done)
                for i in range(ISLAS):
                    _b, _f, _h, aos_i, hof_i = symreg.evolve(
                        Xtr, Ytr, cat_un, cat_bin, pop_size=POP, generations=bloque,
                        init_pop=pops[i], init_aos=aoss[i], gen_offset=done, **comunes)
                    pops[i] = hof_i.final_pop; aoss[i] = aos_i
                    for f, e in hof_i.items:
                        hof.update(e, f)
                if ISLAS > 1:
                    for i in range(ISLAS):
                        src = pops[(i-1) % ISLAS]; dst = pops[i]
                        for k in range(min(3, len(src), len(dst))):
                            dst[-(k+1)] = src[k].clone()
                done += bloque
                mejor = hof.items[0][0] if hof.items else float("inf")
                hist.append(mejor)
                ESTADO["progreso"] = {"gen": done, "total": GEN, "best_fit": mejor,
                                      "elapsed": round(time.time()-t0, 1), "hist": hist[-200:]}
            aos = aoss[0]

        # fine-tuning + reconocimiento de constantes
        if p.get("fine_tune", True) and hof.items:
            hof = symreg.fine_tune_hof(hof, Xtr, Ytr, steps=int(p.get("ft_steps", 250)),
                                       lr=5e-3, verbose=False, alpha=ALPHA)
        fitb, best = hof.items[0]
        best = best.clone()
        if p.get("recognize", True):
            best = symreg.recognize_constants(best, Xtr, Ytr, alpha=ALPHA)

        # selección por validación
        val_mse = None
        if X_val is not None and hof.items:
            xv = symreg.to_device(X_val); yv = symreg.to_device(Y_val.reshape(-1, 1))
            scored = sorted(((symreg.mse_expr(e, xv, yv), f, e) for f, e in hof.items),
                            key=lambda t: t[0])
            hof.items = [(f, e) for vm, f, e in scored]
            cand = symreg.recognize_constants(hof.items[0][1].clone(), Xtr, Ytr, alpha=ALPHA) \
                if p.get("recognize", True) else hof.items[0][1].clone()
            best = cand
            val_mse = float(scored[0][0])

        # métricas finales sobre TODOS los datos
        x_all = symreg.to_device(X2); y_all = symreg.to_device(np.asarray(Y).reshape(-1, 1))
        import torch
        y_pred_t = best.eval_backend(x_all)
        mse = float(torch.mean((y_pred_t - y_all)**2).item())
        y_pred = symreg.to_numpy(y_pred_t).reshape(-1)

        # gráficas
        if not multivar:
            fig, ax = plt.subplots(figsize=(8.5, 4.6))
            ax.scatter(X2[:, 0], Y, s=12, alpha=0.6, label="Datos")
            orden = np.argsort(X2[:, 0])
            ax.plot(X2[orden, 0], y_pred[orden], "r-", lw=2, label="Expresión encontrada")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3); ax.legend()
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(Y, y_pred, s=12, alpha=0.6)
            lim = [min(np.min(Y), np.min(y_pred)), max(np.max(Y), np.max(y_pred))]
            ax.plot(lim, lim, "r--"); ax.set_xlabel("y real"); ax.set_ylabel("y predicho")
            ax.grid(alpha=0.3)
        img_ajuste = _fig_a_base64(fig)

        img_conv = None
        if ESTADO["progreso"] and ESTADO["progreso"].get("hist"):
            h = np.asarray(ESTADO["progreso"]["hist"], dtype=np.float64)
            fig, ax = plt.subplots(figsize=(8.5, 3))
            ax.semilogy(np.maximum(h, 1e-16), "o-", ms=3)
            ax.set_xlabel("bloque"); ax.set_ylabel("mejor fitness (log)")
            ax.grid(alpha=0.3, which="both")
            img_conv = _fig_a_base64(fig)

        # exportación simbólica y descargas
        f_sym, xs = symreg.to_sympy(best)
        try:
            f_latex = sp.latex(sp.nsimplify(f_sym, rational=False))
        except Exception:
            f_latex = sp.latex(f_sym)
        rt = symreg.check_sympy_roundtrip(best, lo=float(X2[:, 0].min()), hi=float(X2[:, 0].max()))
        symreg.save_tree(best, SALIDAS / "mejor_expresion.json")
        if aos is not None:
            symreg.save_checkpoint(SALIDAS / "checkpoint.json", hof, aos)

        ESTADO["resultado"] = {
            "expr": best.to_text(),
            "sympy": str(f_sym),
            "latex": f_latex,
            "mse": mse,
            "val_mse": val_mse,
            "nodos": len(symreg.all_subnodes(best)),
            "tiempo": round(time.time()-t0, 1),
            "roundtrip": ("OK" if rt else ("FALLO" if rt is not None else "no concluyente")),
            "cancelada": bool(ESTADO["cancelar"]),
            "hof": [{"fit": float(f), "expr": e.to_text()[:200]} for f, e in hof.items],
            "pareto": ([{"mse": m, "nodos": n, "expr": e.to_text()[:200]}
                        for m, n, e in hof.pareto] if getattr(hof, "pareto", None) else None),
            "img_ajuste": img_ajuste,
            "img_convergencia": img_conv,
        }
    except Exception:
        ESTADO["error"] = traceback.format_exc()[-2500:]
    finally:
        ESTADO["corriendo"] = False


# ---------- rutas ----------
@app.get("/", response_class=HTMLResponse)
def indice():
    return (ESTATICOS / "index.html").read_text(encoding="utf-8")


@app.get("/api/info")
def info():
    return {"dispositivo": symreg.info(), "version": symreg.__version__,
            "presets": list(TARGETS.keys())}


@app.post("/api/iniciar")
async def iniciar(request: Request):
    if ESTADO["corriendo"]:
        return JSONResponse({"error": "ya hay una corrida en marcha"}, status_code=409)
    p = await request.json()
    ESTADO.update(corriendo=True, cancelar=False, progreso=None, resultado=None, error=None)
    threading.Thread(target=_trabajador, args=(p,), daemon=True).start()
    return {"ok": True}


@app.post("/api/cancelar")
def cancelar():
    ESTADO["cancelar"] = True
    return {"ok": True}


@app.get("/api/estado")
def estado():
    return {k: ESTADO[k] for k in ("corriendo", "progreso", "resultado", "error")}


@app.post("/api/csv")
async def subir_csv(archivo: UploadFile):
    contenido = await archivo.read()
    try:
        M = np.genfromtxt(io.StringIO(contenido.decode("utf-8")), delimiter=",")
        M = M[~np.isnan(M).any(axis=1)]
        if M.ndim != 2 or M.shape[1] < 2:
            raise ValueError("se esperan >= 2 columnas (x..., y)")
        X = M[:, :-1].astype(np.float32); Y = M[:, -1].astype(np.float32)
        ESTADO["csv"] = {"X": X, "Y": Y, "nombre": archivo.filename}
        return {"ok": True, "filas": int(len(Y)), "variables": int(X.shape[1])}
    except Exception as e:
        return JSONResponse({"error": f"CSV inválido: {e}"}, status_code=400)


@app.get("/api/descargar/{cual}")
def descargar(cual: str):
    nombres = {"arbol": "mejor_expresion.json", "checkpoint": "checkpoint.json"}
    if cual not in nombres:
        return JSONResponse({"error": "no existe"}, status_code=404)
    ruta = SALIDAS / nombres[cual]
    if not ruta.exists():
        return JSONResponse({"error": "aún no hay resultado"}, status_code=404)
    return FileResponse(ruta, filename=nombres[cual], media_type="application/json")
