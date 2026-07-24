# Bucle evolutivo: mutación/cruce/macros/hints, fitness (con derivadas opcionales),
# intérprete plano por lotes en GPU, Hall of Fame, NSGA-II, islas.
import random

import numpy as np
import torch

from . import config
from .aos import OperatorManager
from .backend import to_device, to_numpy
from .ops import VOPS_BIN, build_catalogs, vop_unary
from .tree import (Add, Constant, FloatInput, Mul, Sin, all_subnodes,
                   simplify_tree)

_AUTO = object()  # centinela: usa config.DY/config.DDY (datos completos)


# ===== Terminales y generación aleatoria =====
def random_terminal():
    cls = random.choices([FloatInput, Constant], weights=[3, 2], k=1)[0]
    if cls is FloatInput and config.N_VARS > 1:
        return FloatInput(random.randrange(config.N_VARS))
    return cls()


def random_node(aos, max_depth=5):
    if max_depth <= 0: return random_terminal(), []
    r = random.random()
    use_bin = len(aos.bin_list) > 0
    use_un = len(aos.un_list) > 0
    if use_bin and (not use_un or r < 0.5):
        (name, ctor), _ = aos.sample_binary()
        left, ops_l = random_node(aos, max_depth-1)
        right, ops_r = random_node(aos, max_depth-1)
        return ctor(left, right), ops_l + ops_r + [name]
    elif use_un:
        (name, ctor), _ = aos.sample_unary()
        child, ops_c = random_node(aos, max_depth-1)
        return ctor(child), ops_c + [name]
    return random_terminal(), []


# ===== Macros de usuario =====
def _var_azar():
    # con varias variables, la macro elige una al azar (Fase 5.2)
    return FloatInput(random.randrange(config.N_VARS)) if config.N_VARS > 1 else FloatInput()


def macro_sin_lin():
    # sin(a*xi + b)
    return Sin(Add(Mul(Constant(1.0), _var_azar()), Constant(0.0)))


def macro_ax_plus_b():
    return Add(Mul(Constant(1.0), _var_azar()), Constant(0.0))


def macro_cuadrado():
    # a*xi^2 (útil para sumas de cuadrados multivariables)
    v = _var_azar()
    return Mul(Constant(1.0), Mul(v, v.clone()))


USER_MACROS = [macro_sin_lin, macro_ax_plus_b, macro_cuadrado]  # edita esta lista


# ===== Mutación / crossover =====
def mutate(aos, node, max_depth=5, p_point=0.5):
    node = node.clone()
    subs = all_subnodes(node)
    target, parent, attr = random.choice(subs)
    if isinstance(target, Constant) and random.random() < p_point:
        k = random.choice([0.1, 0.5, 1, 2, 5])
        new = Constant(target.value + random.uniform(-k, k)); used = []
    else:
        new, used = random_node(aos, max_depth=max_depth)
    if parent is None: return new, used
    setattr(parent, attr, new)
    return node, used


def mutate_with_macros(aos, node, max_depth=5, p_macro=0.2, p_point=0.5):
    if USER_MACROS and random.random() < p_macro:
        node = node.clone()
        subs = all_subnodes(node)
        _, parent, attr = random.choice(subs)
        new = random.choice(USER_MACROS)()
        if parent is None: return new, ['macro']
        setattr(parent, attr, new)
        return node, ['macro']
    return mutate(aos, node, max_depth=max_depth, p_point=p_point)


def crossover(aos, a, b):
    a = a.clone(); b = b.clone()
    sa = all_subnodes(a); sb = all_subnodes(b)
    ta, pa, attra = random.choice(sa)
    tb, pb, attrb = random.choice(sb)
    donated = tb.clone()
    # Fase 3.4: ops del subárbol donado, para recompensar al crossover en el AOS
    used = ([n.name for n, _, _ in all_subnodes(donated) if n.name not in ("const", "x")]
            if config.AOS_XOVER_REWARD else [])
    if pa is None: child = donated
    else: setattr(pa, attra, donated); child = a
    return child, used


# ===== Hint (sembrar población con función parecida) =====
def make_hint_tree():
    # Edita esta función para representar tu "pista" g(x)
    return Add(Add(Constant(0.0), Mul(Constant(1.0), FloatInput())),
               Mul(Constant(1.0), Sin(FloatInput())))


def inject_hint_population(pop, fraction=0.3, jitter=0.2, tree=None):
    """Siembra una fracción de la población con clones (con jitter en las
    constantes) de una función 'pista': `tree` si se da, o make_hint_tree()."""
    base = tree if tree is not None else make_hint_tree()
    n = max(1, int(len(pop)*fraction))
    for i in range(n):
        clone = base.clone()
        for node, _, _ in all_subnodes(clone):
            if isinstance(node, Constant):
                node.value += np.random.uniform(-jitter, jitter)
        pop[i] = clone


# ===== Fitness =====
def mse_expr(expr, x_arr, y_arr):
    yhat = expr.eval_backend(x_arr)
    if torch.isnan(yhat).any() or torch.isinf(yhat).any():
        return 1e12
    return float(torch.mean((yhat - y_arr)**2).item())


def mse_expr_with_derivs(expr, x_arr, y_arr, dy_arr=None, ddy_arr=None):
    yhat = expr.eval_backend(x_arr)
    mse_y = torch.mean((yhat - y_arr)**2)
    mse_p = torch.tensor(0.0, device=y_arr.device)
    mse_pp = torch.tensor(0.0, device=y_arr.device)
    if (dy_arr is not None) or (ddy_arr is not None):
        x_req = x_arr.clone().detach().requires_grad_(True)
        yhat2 = expr.eval_backend(x_req)
        (dyhat,) = torch.autograd.grad(yhat2, x_req, grad_outputs=torch.ones_like(yhat2),
                                       create_graph=True, retain_graph=True, allow_unused=True)
        if dyhat is None: dyhat = torch.zeros_like(yhat2)
        if dy_arr is not None:
            mse_p = torch.mean((dyhat - dy_arr)**2)
        if ddy_arr is not None:
            (ddyhat,) = torch.autograd.grad(dyhat, x_req, grad_outputs=torch.ones_like(dyhat),
                                            allow_unused=True)
            if ddyhat is None: ddyhat = torch.zeros_like(yhat2)
            mse_pp = torch.mean((ddyhat - ddy_arr)**2)
    loss = (config.LAMBDA_Y*mse_y
            + (0 if dy_arr is None else config.LAMBDA_Yp*mse_p)
            + (0 if ddy_arr is None else config.LAMBDA_Ypp*mse_pp))
    return float(loss.item())


def fitness(expr, x_arr, y_arr, alpha=1e-3, dy_arr=_AUTO, ddy_arr=_AUTO):
    if config.USE_DERIV_LOSS:
        d1 = config.DY if dy_arr is _AUTO else dy_arr
        d2 = config.DDY if ddy_arr is _AUTO else ddy_arr
        base = mse_expr_with_derivs(expr, x_arr, y_arr, dy_arr=d1, ddy_arr=d2)
    else:
        base = mse_expr(expr, x_arr, y_arr)
    return base + alpha*len(all_subnodes(expr))


# ===== Intérprete plano por niveles (Fase 2.2) =====
def _flatten_trees(pop):
    """Aplana una lista de árboles a un DAG por niveles: los operadores se agrupan
    por (nivel, tipo) para evaluarse con pocos kernels grandes."""
    from collections import defaultdict
    x_rows = defaultdict(list); c_rows = []; c_vals = []; roots = []
    groups = defaultdict(lambda: ([], [], []))   # (nivel, clave_op) -> (out, hijo1, hijo2)
    idx = 0
    def rec(nd):
        nonlocal idx
        if isinstance(nd, FloatInput):
            my = idx; idx += 1; x_rows[nd.idx].append(my); return my, 0
        if isinstance(nd, Constant):
            my = idx; idx += 1; c_rows.append(my); c_vals.append(nd.value); return my, 0
        if nd.o2 is None:
            a, la = rec(nd.o1)
            my = idx; idx += 1
            lvl = la + 1
            g = groups[(lvl, (nd.name, getattr(nd, "n", None)))]
            g[0].append(my); g[1].append(a)
            return my, lvl
        a, la = rec(nd.o1); b, lb = rec(nd.o2)
        my = idx; idx += 1
        lvl = max(la, lb) + 1
        g = groups[(lvl, (nd.name, None))]
        g[0].append(my); g[1].append(a); g[2].append(b)
        return my, lvl
    for t in pop:
        r, _ = rec(t); roots.append(r)
    return idx, x_rows, c_rows, c_vals, roots, groups


def eval_trees_matrix(pop, x_arr):
    """Evalúa todos los árboles y devuelve el tensor (pop, n_puntos) SIN sincronizar."""
    n_tot, x_rows, c_rows, c_vals, roots, groups = _flatten_trees(pop)
    dev = x_arr.device
    x2 = x_arr if x_arr.ndim == 2 else x_arr.reshape(-1, 1)   # (n_puntos, n_vars)
    B = torch.empty((n_tot, x2.shape[0]), dtype=x_arr.dtype, device=dev)
    for vidx, rows in x_rows.items():
        B[torch.as_tensor(rows, device=dev)] = x2[:, min(vidx, x2.shape[1]-1)]
    if c_rows:
        vals = torch.as_tensor(np.asarray(c_vals, dtype=np.float64), dtype=x_arr.dtype, device=dev)
        B[torch.as_tensor(c_rows, device=dev)] = vals[:, None]
    for (lvl, key) in sorted(groups.keys()):
        out_i, a_i, b_i = groups[(lvl, key)]
        oi = torch.as_tensor(out_i, device=dev)
        A = B[torch.as_tensor(a_i, device=dev)]
        if b_i:
            R = VOPS_BIN[key[0]](A, B[torch.as_tensor(b_i, device=dev)])
        else:
            R = vop_unary(key)(A)
        B[oi] = R
    return B[torch.as_tensor(roots, device=dev)]


def eval_population_batch(pop, x_arr, y_arr, alpha=1e-3, dy_arr=None, ddy_arr=None):
    """Evalúa una lista de expresiones por lotes y devuelve el vector de fitness
    (np.ndarray). Una única transferencia GPU->CPU por llamada."""
    use_derivs = config.USE_DERIV_LOSS and ((dy_arr is not None) or (ddy_arr is not None))
    if not use_derivs:
        with torch.no_grad():
            Yh = eval_trees_matrix(pop, x_arr)               # (pop, n_puntos)
            yt = y_arr.reshape(1, -1)
            loss = torch.mean((Yh - yt)**2, dim=1)           # todos los MSE en una reducción
            # misma semántica que mse_expr: yhat con NaN/Inf -> 1e12; overflow del MSE queda en inf
            bad = ~torch.isfinite(Yh).all(dim=1)
            loss = torch.where(bad, torch.full_like(loss, 1e12), loss)
    else:
        losses = []
        for e in pop:
            x_req = x_arr.clone().detach().requires_grad_(True)
            yh = e.eval_backend(x_req)
            l = config.LAMBDA_Y*torch.mean((yh - y_arr)**2)
            (dyh,) = torch.autograd.grad(yh, x_req, grad_outputs=torch.ones_like(yh),
                                         create_graph=(ddy_arr is not None), retain_graph=True,
                                         allow_unused=True)
            if dyh is None: dyh = torch.zeros_like(yh)
            if dy_arr is not None:
                l = l + config.LAMBDA_Yp*torch.mean((dyh - dy_arr)**2)
            if ddy_arr is not None:
                (ddyh,) = torch.autograd.grad(dyh, x_req, grad_outputs=torch.ones_like(dyh),
                                              allow_unused=True)
                if ddyh is None: ddyh = torch.zeros_like(yh)
                l = l + config.LAMBDA_Ypp*torch.mean((ddyh - ddy_arr)**2)
            losses.append(l.detach())
        loss = torch.stack(losses)
        loss = torch.where(~torch.isfinite(loss), torch.full_like(loss, 1e12), loss)
    fits = loss.detach().cpu().numpy().astype(np.float64)    # ÚNICA sincronización CPU-GPU
    n_nodes = np.array([len(all_subnodes(e)) for e in pop], dtype=np.float64)
    return fits + alpha*n_nodes


# ===== NSGA-II (Fase 3.1) =====
def nsga2_order(mse, size):
    """(orden, rango, crowding) para (mse, tamaño): sort no dominado + crowding."""
    mse = np.asarray(mse, dtype=np.float64); size = np.asarray(size, dtype=np.float64)
    P = len(mse)
    dom = ((mse[:, None] <= mse[None, :]) & (size[:, None] <= size[None, :]) &
           ((mse[:, None] < mse[None, :]) | (size[:, None] < size[None, :])))
    n_dom = dom.sum(axis=0).astype(np.int64)
    rank = np.full(P, -1, dtype=np.int64)
    remaining = np.ones(P, dtype=bool)
    fronts = []; fr = 0
    while remaining.any():
        cur = remaining & (n_dom <= 0)
        if not cur.any():
            cur = remaining.copy()
        idxs = np.where(cur)[0]
        fronts.append(idxs); rank[idxs] = fr
        remaining[idxs] = False
        n_dom = n_dom - dom[idxs].sum(axis=0)
        fr += 1
    crowd = np.zeros(P)
    for fidx in fronts:
        if len(fidx) <= 2:
            crowd[fidx] = np.inf; continue
        for obj in (mse, size):
            o = obj[fidx]; s = np.argsort(o)
            rng = float(o[s[-1]] - o[s[0]]) or 1.0
            crowd[fidx[s[0]]] = crowd[fidx[s[-1]]] = np.inf
            crowd[fidx[s[1:-1]]] += (o[s[2:]] - o[s[:-2]])/rng
    order = sorted(range(P), key=lambda i: (rank[i], -crowd[i]))
    return order, rank, crowd


# ===== Deduplicación semántica (Fase 2.5) =====
def dedup_population(pop, aos, max_depth, x_probe=None, lo=-3.0, hi=3.0, n_probe=16):
    """Firma semántica = valores en n_probe puntos (un lote en GPU); los
    duplicados funcionales se sustituyen por árboles frescos."""
    if x_probe is None:
        x_probe = to_device(np.linspace(lo, hi, n_probe).reshape(-1, 1).astype(np.float32))
    with torch.no_grad():
        M = to_numpy(eval_trees_matrix(pop, x_probe))
    M = np.nan_to_num(M, nan=1e30, posinf=1e30, neginf=-1e30)
    seen = set(); out = list(pop); n_rep = 0
    for i, row in enumerate(M):
        sig = np.round(row, 4).tobytes()
        if sig in seen:
            out[i], _ = random_node(aos, max_depth=max_depth)
            n_rep += 1
        else:
            seen.add(sig)
    return out, n_rep


# ===== Hall of Fame (Fase 1.2) =====
class HallOfFame:
    """Archiva los k mejores de TODA la corrida, evaluados con datos completos
    (el submuestreo por generación podía perder al mejor histórico)."""

    def __init__(self, k=10):
        self.k = int(k)
        self.items = []   # lista de (fitness_full, expr) ordenada ascendente
        self.pareto = None
        self.val_mse = None
        self.val_idx = None
        self.final_pop = None

    def update(self, expr, fit_full):
        if not np.isfinite(fit_full): return
        key = expr.to_text()
        for i, (f, e) in enumerate(self.items):
            if e.to_text() == key:
                if fit_full < f:
                    self.items[i] = (fit_full, expr.clone())
                    self.items.sort(key=lambda t: t[0])
                return
        self.items.append((fit_full, expr.clone()))
        self.items.sort(key=lambda t: t[0])
        del self.items[self.k:]

    def best(self):
        return self.items[0] if self.items else (None, None)


# ===== Evolución =====
def evolve(X, Y, cat_un, cat_bin, pop_size=160, generations=180, max_depth=5,
           elite=10, p_mut=0.6, p_xover=0.3, alpha=1e-3, verbose=True,
           aos_params=None,
           use_macros=True, p_macro=0.2,
           use_hint=True, hint_fraction=0.3, hint_jitter=0.2, hint_tree=None,
           unary_priors=None, binary_priors=None,
           sample_schedule=None,
           curriculum=None,
           hill_climb_steps=3,
           hof_k=10,
           selection=None,
           val_fraction=0.0,
           early_stop_patience=0,
           restart_patience=0,
           stagnation_tol=1e-10,
           init_pop=None,
           init_aos=None,
           gen_offset=0):
    """Bucle evolutivo principal. Devuelve (best, fit_full, best_hist, aos, hof).

    Con los flags de config por defecto el comportamiento del algoritmo es el
    del notebook original (misma dinámica AOS/torneo/memetic); la evaluación por
    lotes puede diferir en +-1 ulp del camino clásico (orden de reducción)."""
    X = np.asarray(X); Y = np.asarray(Y)
    X = X.reshape(len(X), -1)          # (n, n_vars): multivariable
    config.N_VARS = X.shape[1]
    if config.USE_DERIV_LOSS and config.N_VARS > 1:
        raise ValueError("USE_DERIV_LOSS solo está soportado con una variable de entrada")

    # Split train/validación (Fase 3.5): la evolución solo ve train
    X_val = Y_val = val_idx = None
    if val_fraction and val_fraction > 0.0:
        perm = np.random.permutation(len(X))
        n_val = max(4, int(len(X)*val_fraction))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_val, Y_val = X[val_idx], Y[val_idx]
        X, Y = X[tr_idx], Y[tr_idx]
        if config.DY is not None and isinstance(config.DY, np.ndarray):
            config.DY = config.DY[tr_idx]
        if config.DDY is not None and isinstance(config.DDY, np.ndarray):
            config.DDY = config.DDY[tr_idx]

    X_full = to_device(X)
    Y_full = to_device(Y.reshape(-1, 1))

    # Derivadas a dispositivo
    if config.DY is not None and isinstance(config.DY, np.ndarray):
        config.DY = to_device(config.DY.reshape(-1, 1))
    if config.DDY is not None and isinstance(config.DDY, np.ndarray):
        config.DDY = to_device(config.DDY.reshape(-1, 1))

    # Escala de Y para normalizar recompensas AOS (Fase 3.4)
    y_scale = (float(np.var(np.asarray(Y, dtype=np.float64))) + 1e-12) if config.AOS_REWARD_NORM else 1.0
    sel = selection if selection is not None else config.SELECTION

    if init_aos is not None:
        aos = init_aos    # continuación (islas): conserva scores y catálogo podado
    else:
        aos = OperatorManager(cat_un, cat_bin, **(aos_params or {}))
        if unary_priors:
            for k, v in unary_priors.items():
                if k in aos.un_scores: aos.un_scores[k] += float(v)
        if binary_priors:
            for k, v in binary_priors.items():
                if k in aos.bin_scores: aos.bin_scores[k] += float(v)

    # Población inicial (o continuación desde init_pop)
    if init_pop is not None:
        pop = [e.clone() for e in init_pop[:pop_size]]
        while len(pop) < pop_size:
            pop.append(random_node(aos, max_depth=max_depth)[0])
    else:
        pop = [random_node(aos, max_depth=max_depth)[0] for _ in range(pop_size)]
        if use_hint:
            inject_hint_population(pop, fraction=hint_fraction, jitter=hint_jitter, tree=hint_tree)

    hof = HallOfFame(k=hof_k)

    def fit_full(e):
        return fitness(e, X_full, Y_full, alpha=alpha)

    def pick_subset(gen):
        if not sample_schedule:
            return X_full, Y_full, config.DY, config.DDY
        frac = 1.0
        for g_lim, f in sample_schedule:
            if gen <= g_lim: frac = f; break
            frac = f
        n = max(8, int(len(X)*frac))
        idx = np.random.choice(len(X), size=n, replace=False)
        dy_s = config.DY[idx] if config.DY is not None else None
        ddy_s = config.DDY[idx] if config.DDY is not None else None
        return X_full[idx], Y_full[idx], dy_s, ddy_s

    def apply_curriculum(gen):
        nonlocal max_depth, aos
        if not curriculum: return
        for phase in curriculum:
            if gen <= phase.get("until", generations):
                max_depth = phase.get("depth", max_depth)
                un_names = phase.get("un", None)
                bin_names = phase.get("bin", None)
                if un_names is not None or bin_names is not None:
                    new_un, new_bin = build_catalogs(
                        un_names if un_names is not None else [n for n, _ in aos.un_list],
                        bin_names if bin_names is not None else [n for n, _ in aos.bin_list])
                    old_un = dict(aos.un_scores); old_bin = dict(aos.bin_scores)
                    aos.un_list = list(new_un); aos.bin_list = list(new_bin)
                    aos.un_scores = {n: old_un.get(n, 0.0) for n, _ in aos.un_list}
                    aos.bin_scores = {n: old_bin.get(n, 0.0) for n, _ in aos.bin_list}
                break

    _probe_idx = np.linspace(0, len(X)-1, 16).astype(int)   # filas reales para dedup
    best_track = np.inf; stagn = 0
    rank_key = {}
    best_hist = []

    for g in range(generations):
        apply_curriculum(g + gen_offset)
        x_arr, y_arr, dy_s, ddy_s = pick_subset(g + gen_offset)

        # Control de bloat periódico (Fase 2.5)
        if config.SIMPLIFY_EVERY and g and g % config.SIMPLIFY_EVERY == 0:
            pop = [simplify_tree(e) for e in pop]
        if config.DEDUP_EVERY and g and g % config.DEDUP_EVERY == 0:
            pop, _ = dedup_population(pop, aos, max_depth, x_probe=X_full[_probe_idx])

        fit_cache = {}   # caché por individuo y generación (Fase 2.1)

        def fit_sub(e):
            key = id(e)
            if config.FITNESS_CACHE and key in fit_cache: return fit_cache[key]
            v = fitness(e, x_arr, y_arr, alpha=alpha, dy_arr=dy_s, ddy_arr=ddy_s)
            if config.FITNESS_CACHE: fit_cache[key] = v
            return v

        use_batch = config.BATCH_EVAL
        if use_batch:
            farr = eval_population_batch(pop, x_arr, y_arr, alpha=alpha, dy_arr=dy_s, ddy_arr=ddy_s)
        else:
            farr = np.array([fit_sub(e) for e in pop], dtype=np.float64)
        if sel == "nsga2":
            nn_arr = np.array([len(all_subnodes(e)) for e in pop], dtype=np.float64)
            order, _rk, _cw = nsga2_order(farr - alpha*nn_arr, nn_arr)
            rank_key = {id(pop[i]): (int(_rk[i]), -float(_cw[i])) for i in range(len(pop))}
        else:
            order = np.argsort(farr)
        pop = [pop[i] for i in order]
        fits = [float(farr[i]) for i in order]
        for e, f in zip(pop, fits): fit_cache[id(e)] = f

        best, best_fit = pop[0], fits[0]
        best_hist.append(best_fit)
        hof.update(best, fit_full(best))
        if verbose and (g % 10 == 0 or g == generations-1):
            print(f"Gen {g:3d}  best={best_fit:.6g}   expr={best.to_text()}")
        aos.maybe_prune()

        # Estancamiento: early stopping y reinicio con diversidad (Fase 3.3)
        best_full_now = hof.items[0][0] if hof.items else np.inf
        if best_full_now < best_track - stagnation_tol:
            best_track = best_full_now; stagn = 0
        else:
            stagn += 1
        if early_stop_patience and stagn >= early_stop_patience:
            if verbose: print(f"Gen {g}: early stopping ({stagn} generaciones sin mejora)")
            break
        if restart_patience and stagn > 0 and stagn % restart_patience == 0:
            if verbose: print(f"Gen {g}: estancamiento ({stagn} gens) -> reinicio parcial")
            pop = pop[:elite] + [random_node(aos, max_depth=max_depth)[0]
                                 for _ in range(pop_size - elite)]
            for _scores in (aos.un_scores, aos.bin_scores):
                for _k in _scores: _scores[_k] = 0.0
            continue

        def tournament(k=3):
            pool = pop[:50] + pop
            cand = random.sample(pool, k=min(k, len(pool)))
            if sel == "nsga2":
                return min(cand, key=lambda c: rank_key.get(id(c), (10**9, 0.0)))
            vals = [fit_sub(c) for c in cand]
            return cand[int(np.argmin(vals))]

        new_pop = pop[:elite]
        k_hc = max(0, int(hill_climb_steps))

        # Memetic secuencial sobre el mejor (idéntico al original)
        cur = best; cur_fit = best_fit
        for _ in range(k_hc):
            child, _u = (mutate_with_macros(aos, cur, max_depth=max_depth) if use_macros
                         else mutate(aos, cur, max_depth=max_depth))
            child_fit = fitness(child, x_arr, y_arr, alpha=alpha, dy_arr=dy_s, ddy_arr=ddy_s)
            if child_fit < cur_fit:
                cur, cur_fit = child, child_fit
        new_pop[0] = cur
        if cur is not best:
            hof.update(cur, fit_full(cur))

        # Descendencia con recompensas AOS (sub-lotes si AOS_UPDATE_CHUNK > 1)
        chunk = max(1, int(config.AOS_UPDATE_CHUNK)) if use_batch else 1
        if chunk == 1:
            while len(new_pop) < pop_size:
                parent = tournament()
                base_fit = fit_sub(parent)
                r = random.random()
                if r < p_mut:
                    child, used = (mutate_with_macros(aos, parent, max_depth=max_depth) if use_macros
                                   else mutate(aos, parent, max_depth=max_depth))
                    child_fit = fitness(child, x_arr, y_arr, alpha=alpha, dy_arr=dy_s, ddy_arr=ddy_s)
                    if used: aos.update_reward(used, max(0.0, base_fit - child_fit)/y_scale)
                elif r < p_mut + p_xover:
                    p2 = tournament()
                    child, used = crossover(aos, parent, p2)
                    child_fit = fitness(child, x_arr, y_arr, alpha=alpha, dy_arr=dy_s, ddy_arr=ddy_s)
                    if used: aos.update_reward(used, max(0.0, base_fit - child_fit)/y_scale)
                else:
                    child, used = random_node(aos, max_depth=max_depth)
                    child_fit = fitness(child, x_arr, y_arr, alpha=alpha, dy_arr=dy_s, ddy_arr=ddy_s)
                    if used: aos.update_reward(used, max(0.0, best_fit - child_fit)/y_scale)
                new_pop.append(child)
        else:
            while len(new_pop) < pop_size:
                children = []; meta = []
                while len(children) < chunk and len(new_pop) + len(children) < pop_size:
                    parent = tournament()
                    r = random.random()
                    if r < p_mut:
                        child, used = (mutate_with_macros(aos, parent, max_depth=max_depth) if use_macros
                                       else mutate(aos, parent, max_depth=max_depth))
                        meta.append((used, fit_sub(parent)))
                    elif r < p_mut + p_xover:
                        p2 = tournament()
                        child, used = crossover(aos, parent, p2)
                        meta.append((used, fit_sub(parent)))
                    else:
                        child, used = random_node(aos, max_depth=max_depth)
                        meta.append((used, best_fit))
                    children.append(child)
                bf = eval_population_batch(children, x_arr, y_arr, alpha=alpha,
                                           dy_arr=dy_s, ddy_arr=ddy_s)
                for (used, base), f_child in zip(meta, bf):
                    if used:
                        aos.update_reward(used, max(0.0, base - float(f_child))/y_scale)
                new_pop.extend(children)
        pop = new_pop

    # Evaluación final con todos los datos (train) y Hall of Fame
    if config.BATCH_EVAL:
        farr = eval_population_batch(pop, X_full, Y_full, alpha=alpha,
                                     dy_arr=config.DY, ddy_arr=config.DDY)
        order = np.argsort(farr)
        pop = [pop[i] for i in order]
        fits = [float(farr[i]) for i in order]
    else:
        fits = [fit_full(e) for e in pop]
        order = np.argsort(fits)
        pop = [pop[i] for i in order]
        fits = [fits[i] for i in order]
    for e, f in zip(pop[:hof_k], fits[:hof_k]):
        hof.update(e, f)

    # Frente de Pareto precisión-complejidad (Fase 3.1)
    if sel == "nsga2":
        nn_f = np.array([len(all_subnodes(e)) for e in pop], dtype=np.float64)
        mse_f = np.array(fits) - alpha*nn_f
        _o, rk_f, _c = nsga2_order(mse_f, nn_f)
        seen_tx = set(); pareto = []
        for i in range(len(pop)):
            if rk_f[i] == 0:
                tx = pop[i].to_text()
                if tx not in seen_tx:
                    seen_tx.add(tx)
                    pareto.append((float(mse_f[i]), int(nn_f[i]), pop[i].clone()))
        hof.pareto = sorted(pareto, key=lambda t: t[0])

    # Selección del mejor por validación (Fase 3.5)
    hof.val_idx = val_idx
    if X_val is not None and hof.items:
        xv = to_device(X_val); yv = to_device(Y_val.reshape(-1, 1))
        scored = sorted(((mse_expr(e, xv, yv), f, e) for f, e in hof.items), key=lambda t: t[0])
        hof.val_mse = {e.to_text(): float(vm) for vm, f, e in scored}
        hof.items = [(f, e) for vm, f, e in scored]

    hof.final_pop = pop
    best_fit_full, best_full = hof.items[0]
    return best_full.clone(), best_fit_full, best_hist, aos, hof


def evolve_islands(X, Y, cat_un, cat_bin, n_islands=4, migrate_every=50, migrants=3,
                   generations=1000, pop_size=160, verbose=True, hof_k=10,
                   val_fraction=0.0, **kwargs):
    """Fase 3.6: islas en la misma GPU con migración periódica en anillo.
    Devuelve (best, fit, hist, aos_isla0, hof_global) como evolve()."""
    X = np.asarray(X); Y = np.asarray(Y)
    X = X.reshape(len(X), -1)
    X_val = Y_val = None
    if val_fraction and val_fraction > 0.0:
        perm = np.random.permutation(len(X))
        n_val = max(4, int(len(X)*val_fraction))
        X_val, Y_val = X[perm[:n_val]], Y[perm[:n_val]]
        X, Y = X[perm[n_val:]], Y[perm[n_val:]]

    pops = [None]*n_islands; aoss = [None]*n_islands
    hof_g = HallOfFame(k=hof_k); hist = []
    done = 0
    while done < generations:
        block = min(migrate_every, generations - done)
        for i in range(n_islands):
            _b, _f, _h, aos_i, hof_i = evolve(
                X, Y, cat_un, cat_bin, pop_size=pop_size, generations=block,
                verbose=False, hof_k=hof_k, init_pop=pops[i], init_aos=aoss[i],
                gen_offset=done, **kwargs)
            pops[i] = hof_i.final_pop; aoss[i] = aos_i
            for f, e in hof_i.items:
                hof_g.update(e, f)
        for i in range(n_islands):
            src = pops[(i-1) % n_islands]; dst = pops[i]
            for k in range(min(migrants, len(src), len(dst))):
                dst[-(k+1)] = src[k].clone()
        done += block
        hist.append(hof_g.items[0][0] if hof_g.items else np.inf)
        if verbose:
            print(f"[islas] gen {done:4d}  best_full={hist[-1]:.6g}")

    if X_val is not None and hof_g.items:
        xv = to_device(X_val); yv = to_device(Y_val.reshape(-1, 1))
        scored = sorted(((mse_expr(e, xv, yv), f, e) for f, e in hof_g.items), key=lambda t: t[0])
        hof_g.val_mse = {e.to_text(): float(vm) for vm, f, e in scored}
        hof_g.items = [(f, e) for vm, f, e in scored]
    f0, e0 = hof_g.items[0]
    return e0.clone(), f0, hist, aoss[0], hof_g
