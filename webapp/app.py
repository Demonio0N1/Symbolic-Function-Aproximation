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


# alias de variables para la pestaña "Expresión": x->col0, y->col1, z->col2, ...
ALIAS_VARS = ["x", "y", "z", "u", "v", "w"]

_FUNS_NP = {"np": np, "pi": np.pi, "e": np.e,
            "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
            "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "tanh": np.tanh,
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan}


def _vars_de_expr(expr):
    """Detecta cuántas variables usa la expresión (x,y,z,u,v,w o x1..x5)."""
    import re
    nombres = set(re.findall(r"\b([a-zA-Z_][a-zA-Z_0-9]*)\b", expr))
    n = 1
    for i, a in enumerate(ALIAS_VARS):
        if a in nombres:
            n = max(n, i+1)
    for m in nombres:
        if re.fullmatch(r"x[1-5]", m):
            n = max(n, int(m[1]) + 1)
    return n


def _construir_datos(p):
    """Devuelve (X, Y, nombres_vars) según la fuente: preset, expresión o CSV.
    nombres_vars es None para preset/CSV (se usan x, x1, x2... del motor)."""
    fuente = p.get("fuente", "preset")
    if fuente == "csv":
        if ESTADO["csv"] is None:
            raise ValueError("No hay CSV subido")
        return ESTADO["csv"]["X"], ESTADO["csv"]["Y"], None
    lo = float(p.get("lo", -5)); hi = float(p.get("hi", 5))
    npts = int(p.get("npts", 400))
    if fuente == "preset":
        X = np.linspace(lo, hi, npts).astype(np.float32)
        fn = TARGETS[p.get("preset", "propia")][0]
        Y = fn(X.astype(np.float64)).astype(np.float32)
        nombres = None
    else:
        # expresión personalizada (multivariable): ^ como potencia, numpy restringido
        import re
        expr = p.get("expr", "2*x**3 + 3*sin(x) + 1").strip().replace("^", "**")
        # admite formas de ecuación: "... = 0", "f(x,y) = ...", "g = ..." o "izq = der"
        if "=" in expr:
            izq, _, der = expr.partition("=")
            izq, der = izq.strip(), der.strip()
            if der in ("", "0", "0.0"):
                expr = izq                       # "expr = 0"  ->  expr
            elif re.fullmatch(r"[a-zA-Z_]\w*\s*(\(.*\))?", izq):
                expr = der                       # "f(x,y) = expr"  ->  expr
            else:
                expr = f"({izq}) - ({der})"      # "izq = der"  ->  izq - der
        if not expr:
            raise ValueError("La expresión está vacía")
        n_vars = _vars_de_expr(expr)
        if n_vars == 1:
            G = np.linspace(lo, hi, npts).reshape(-1, 1)
        else:
            # con varias variables se muestrea uniforme en [lo, hi]^n_vars
            G = np.random.default_rng(int(p.get("seed", 2))).uniform(lo, hi, size=(npts, n_vars))
        entorno = {"__builtins__": {}, **_FUNS_NP}
        for i in range(n_vars):
            entorno[ALIAS_VARS[i]] = G[:, i]
            if i >= 1:
                entorno[f"x{i}"] = G[:, i]   # alias alternativo x1, x2...
        try:
            with np.errstate(all="ignore"):
                Y = np.asarray(eval(expr, entorno), dtype=np.float64)  # noqa: S307 (herramienta local)
        except Exception as e:
            raise ValueError(
                f"Expresión inválida ({type(e).__name__}: {e}). Usa funciones numpy "
                "(sin, cos, exp, log, sqrt, abs...), variables x, y, z, u, v, w "
                "(o x1..x5) y potencias con ^ o **. Ejemplo: x^2 + y^2 + z^3")
        if Y.shape != (len(G),):
            Y = np.full(len(G), float(Y))
        if not np.isfinite(Y).all():
            m = np.isfinite(Y)
            if m.sum() < 16:
                raise ValueError("La expresión produce valores no finitos en casi todo el dominio "
                                 "(revisa divisiones por cero o log de negativos en [x mín, x máx])")
            G = G[m]; Y = Y[m]   # descarta puntos no finitos (p.ej. log fuera de dominio)
        X = G[:, 0].astype(np.float32) if n_vars == 1 else G.astype(np.float32)
        Y = Y.astype(np.float32)
        nombres = ALIAS_VARS[:n_vars] if n_vars > 1 else None
    ruido = float(p.get("ruido", 0.0))
    if ruido > 0:
        Y = (Y + np.random.default_rng(0).normal(0, ruido, size=Y.shape)).astype(np.float32)
    return X, Y, nombres


def _polyfit_tree(X2, Y, grado=3):
    """Fit inicial (warm start): polinomio por mínimos cuadrados convertido a
    árbol del motor — c0 + sum_j sum_k c_jk * x_j^k. Sirve de semilla (hint)."""
    from symreg import Add, Constant, FloatInput, Mul, Pow
    X2 = np.asarray(X2, dtype=np.float64).reshape(len(X2), -1)
    n, d = X2.shape
    cols = [np.ones(n)]; terminos = []
    for j in range(d):
        for k in range(1, int(grado)+1):
            cols.append(X2[:, j]**k)
            terminos.append((j, k))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, np.asarray(Y, dtype=np.float64), rcond=None)
    arbol = Constant(float(coef[0]))
    for (j, k), c in zip(terminos, coef[1:]):
        if not np.isfinite(c) or abs(c) < 1e-12:
            continue
        term = FloatInput(j) if k == 1 else Pow(FloatInput(j), Constant(float(k)))
        arbol = Add(arbol, Mul(Constant(float(c)), term))
    return arbol


def _trabajador(p):
    """Corre la evolución por bloques (progreso + cancelación) y deja el resultado."""
    t0 = time.time()
    try:
        seed = int(p.get("seed", 2))
        config.set_seed(seed)
        config.EML_MODE = p.get("eml_mode", "off")
        config.SELECTION = p.get("selection", "alpha")
        config.AOS_UPDATE_CHUNK = int(p.get("aos_chunk", 1))

        X, Y, nombres_vars = _construir_datos(p)
        X2 = np.asarray(X).reshape(len(X), -1)
        multivar = X2.shape[1] > 1

        POP = int(p.get("pop", 160)); GEN = int(p.get("gen", 250))
        ISLAS = max(1, int(p.get("islas", 4))); DEPTH = int(p.get("depth", 5))
        ALPHA = float(p.get("alpha", 1e-3))
        VAL = float(p.get("val_fraction", 0.2))
        RESTART = int(p.get("restart_patience", 80))
        BLOQUE = max(10, int(p.get("bloque", 25)))   # gens entre actualizaciones de progreso

        # listas vacías son válidas (p.ej. catálogo solo-eml: unarios=[], binarios=["eml"])
        unarios = p.get("unarios")
        if unarios is None: unarios = ["sin", "cos", "log", "sqrt", "abs", "neg"]
        binarios = p.get("binarios")
        if binarios is None: binarios = ["add", "sub", "mul", "div", "pow"]
        if config.EML_MODE == "operator" and "eml" not in binarios:
            binarios = binarios + ["eml"]
        if not unarios and not binarios:
            raise ValueError("Elige al menos un operador (unario o binario)")
        cat_un, cat_bin = symreg.build_catalogs(unarios, binarios)

        # split de validación único (los bloques no deben re-partir)
        X_val = Y_val = None
        Xtr, Ytr = X2, np.asarray(Y)
        if VAL > 0:
            perm = np.random.permutation(len(X2))
            n_val = max(4, int(len(X2)*VAL))
            X_val, Y_val = X2[perm[:n_val]], np.asarray(Y)[perm[:n_val]]
            Xtr, Ytr = X2[perm[n_val:]], np.asarray(Y)[perm[n_val:]]

        # Fit inicial (warm start): siembra la población con un polinomio LSQ
        hint_tree = None; warm_expr = None
        if p.get("warm_start"):
            hint_tree = _polyfit_tree(Xtr, Ytr, grado=int(p.get("warm_grado", 3)))
            warm_expr = hint_tree.to_text()

        # catálogo "solo eml": sin macros ni hint (inyectarían sin/add/mul ajenos al catálogo)
        solo_eml = (not unarios) and set(binarios) == {"eml"}
        comunes = dict(
            max_depth=DEPTH, elite=10, p_mut=0.6, p_xover=0.3, alpha=ALPHA,
            aos_params=dict(tau=0.8, lr=0.4, decay=0.98, prune_every=40,
                            min_keep_un=3, min_keep_bin=3, prune_threshold=0.04),
            use_macros=not solo_eml, p_macro=0.25,   # las macros eligen variable al azar (multivar OK)
            # con warm start la pista es el polinomio ajustado (vale multivar);
            # si no, la pista clásica sin+lineal (solo univariable)
            use_hint=(hint_tree is not None) or ((not multivar) and (not solo_eml)),
            hint_tree=hint_tree, hint_fraction=0.35, hint_jitter=0.25,
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

        # modo EML puro (solo eml): gramática S -> hoja_afin | eml(S,S), por gradiente
        if config.EML_MODE == "pure":
            if multivar:
                raise ValueError("El modo EML puro solo soporta una variable (usa el catálogo "
                                 "solo-eml de la evolución para multivariable)")
            n_starts = int(p.get("eml_starts", config.EML_PURE_STARTS))
            pasos = int(p.get("eml_steps", config.EML_PURE_STEPS))
            profundidad = int(p.get("depth", config.EML_PURE_DEPTH))

            ESTADO["progreso"] = {"gen": 0, "total": 2, "best_fit": float("nan"),
                                  "elapsed": round(time.time()-t0, 1), "hist": [],
                                  "fase": "1/2 · evolución eml-only (inicializador de pesos)"}
            spec = None; leaf_init = None
            cat_un_e, cat_bin_e = symreg.build_catalogs([], ["eml"])
            best_e, _f, _h, _a, _hof = symreg.evolve(
                Xtr.reshape(-1), Ytr, cat_un_e, cat_bin_e, pop_size=100, generations=60,
                max_depth=profundidad, verbose=False, use_macros=False, use_hint=False, hof_k=5)
            x_d = symreg.to_device(Xtr.reshape(-1, 1))
            spec_e, leaf_init_e = symreg._eml_structure_from_tree(
                best_e, x_d, max_depth=max(2*profundidad, 8))
            if spec_e[0] == "eml":
                spec, leaf_init = spec_e, leaf_init_e
            if spec is None:
                spec, n_hojas = symreg._eml_full_structure(profundidad)
                leaf_init = None
            else:
                n_hojas = len(leaf_init)

            if not ESTADO["cancelar"]:
                ESTADO["progreso"] = {"gen": 1, "total": 2, "best_fit": float("nan"),
                                      "elapsed": round(time.time()-t0, 1), "hist": [],
                                      "fase": f"2/2 · entrenando {n_starts} reinicios en paralelo "
                                              f"({n_hojas} hojas, {pasos} pasos de Adam)"}
                best, mse_puro = symreg.eml_pure_train(
                    spec, n_hojas, Xtr.reshape(-1), Ytr, n_starts=n_starts,
                    steps=pasos, lr=config.EML_PURE_LR, leaf_init=leaf_init, verbose=False)
            else:
                best, mse_puro = best_e, float("inf")
            hof = symreg.HallOfFame(k=1)
            x_tr = symreg.to_device(Xtr); y_tr = symreg.to_device(Ytr.reshape(-1, 1))
            hof.update(best, symreg.fitness(best, x_tr, y_tr, alpha=ALPHA))
            ESTADO["progreso"].update(gen=2, best_fit=float(mse_puro),
                                      elapsed=round(time.time()-t0, 1), fase="completado")
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

        # exportación simbólica y descargas (con los nombres de variable del usuario)
        f_sym, xs = symreg.to_sympy(best)
        leyenda_vars = None
        if nombres_vars and isinstance(xs, (list, tuple)):
            subs = {s: sp.Symbol(nombres_vars[i], real=True)
                    for i, s in enumerate(xs) if i < len(nombres_vars)}
            f_sym = f_sym.subs(subs, simultaneous=True)
            leyenda_vars = {str(s): nombres_vars[i]
                            for i, s in enumerate(xs) if i < len(nombres_vars)}
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
            "variables": leyenda_vars,   # p.ej. {"x":"x","x1":"y","x2":"z"} (None si 1 variable)
            "warm_expr": warm_expr,      # polinomio semilla del warm start (None si off)
            "hof": [{"fit": float(f), "expr": e.to_text()[:200]} for f, e in hof.items],
            "pareto": ([{"mse": m, "nodos": n, "expr": e.to_text()[:200]}
                        for m, n, e in hof.pareto] if getattr(hof, "pareto", None) else None),
            "img_ajuste": img_ajuste,
            "img_convergencia": img_conv,
        }
    except ValueError as e:
        ESTADO["error"] = str(e)   # errores de usuario: mensaje limpio, sin traceback
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
