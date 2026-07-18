# Modo EML puro (Fase 4.2): gramática S -> hoja_afin | eml(S,S) con hojas
# alpha_i + beta_i*x entrenadas por gradiente; multi-starts EN PARALELO como
# dimensión extra del tensor (la init aleatoria falla a profundidad > 4).
import numpy as np
import torch

from . import config
from .backend import to_device
from .evolve import evolve
from .ops import build_catalogs
from .tree import Add, Constant, Eml, FloatInput, Mul


def _eml_full_structure(depth):
    """Árbol eml completo de profundidad `depth`; devuelve (spec, n_hojas).
    spec anidado: ("eml", izq, der) | ("leaf", i)."""
    counter = [0]
    def rec(d):
        if d <= 0:
            i = counter[0]; counter[0] += 1
            return ("leaf", i)
        return ("eml", rec(d-1), rec(d-1))
    return rec(depth), counter[0]


def _eml_structure_from_tree(tree, x_d, max_depth=None):
    """Los nodos Eml se conservan (hasta max_depth); cualquier otro subárbol se
    aproxima por una hoja afín a + b*x ajustada por mínimos cuadrados en GPU."""
    leaf_inits = []
    xcol = x_d.reshape(-1)
    n = float(xcol.shape[0])
    def rec(nd, d=0):
        if isinstance(nd, Eml) and (max_depth is None or d < max_depth):
            return ("eml", rec(nd.o1, d+1), rec(nd.o2, d+1))
        with torch.no_grad():
            v = nd.eval_backend(x_d).reshape(-1)
            v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
            sx = xcol.sum(); sxx = (xcol*xcol).sum()
            sv = v.sum(); sxv = (xcol*v).sum()
            den = n*sxx - sx*sx
            b = (n*sxv - sx*sv)/den
            a = (sv - b*sx)/n
        leaf_inits.append((float(a.item()), float(b.item())))
        return ("leaf", len(leaf_inits)-1)
    return rec(tree), leaf_inits


def eml_pure_train(spec, n_leaves, X, Y, n_starts=256, steps=1500, lr=5e-3,
                   leaf_init=None, init_fraction=0.25, jitter=0.3,
                   verbose=True, use_compile=False):
    """Entrena las hojas afines de una estructura eml FIJA con n_starts reinicios
    en paralelo (tensores (N, n_hojas)); un Adam y una pérdida (N,) sumada.
    Devuelve (árbol del motor con las hojas ganadoras, mse)."""
    _Xa = np.asarray(X)
    assert _Xa.reshape(len(_Xa), -1).shape[1] == 1, "el modo EML puro solo soporta una variable"
    x_t = to_device(_Xa.reshape(-1)); y_t = to_device(np.asarray(Y).reshape(-1))
    N = int(n_starts); dev = x_t.device
    # init pequeña: las cadenas de exp anidadas divergen con hojas grandes
    A = torch.empty((N, n_leaves), dtype=x_t.dtype, device=dev).uniform_(-1.5, 1.5)
    B = torch.empty((N, n_leaves), dtype=x_t.dtype, device=dev).uniform_(-1.5, 1.5)
    if leaf_init:
        a0 = torch.tensor([a for a, b in leaf_init], dtype=x_t.dtype, device=dev)
        b0 = torch.tensor([b for a, b in leaf_init], dtype=x_t.dtype, device=dev)
        k = max(1, int(N*init_fraction))
        A[:k] = a0[None, :] + jitter*torch.randn(k, n_leaves, dtype=x_t.dtype, device=dev)
        B[:k] = b0[None, :] + jitter*torch.randn(k, n_leaves, dtype=x_t.dtype, device=dev)
        A[0] = a0; B[0] = b0   # un reinicio exacto con la init evolutiva
    A.requires_grad_(True); B.requires_grad_(True)

    def forward():
        def rec(node):
            if node[0] == "leaf":
                i = node[1]
                return A[:, i:i+1] + B[:, i:i+1]*x_t[None, :]
            l = rec(node[1]); r = rec(node[2])
            # recorte a 20 (exp<=5e8): mantiene la pérdida finita y el gradiente vivo
            return torch.exp(torch.clamp(l, max=20.0)) - torch.log(torch.abs(r) + 1e-8)
        yh = rec(spec)                                       # (N, n_puntos)
        return torch.mean((yh - y_t[None, :])**2, dim=1)     # (N,)

    fwd = forward
    if use_compile:
        try:
            fwd = torch.compile(forward)   # estructura fija => compila una vez
        except Exception:
            fwd = forward
    opt = torch.optim.Adam([A, B], lr=lr)
    reseed_every = max(1, steps//4)
    for i in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        losses = fwd()
        loss = torch.nan_to_num(losses, nan=1e12, posinf=1e12).sum()
        loss.backward(); opt.step()
        # re-siembra por lotes: los reinicios divergidos se relanzan desde el mejor
        if i % reseed_every == 0 and i < steps:
            with torch.no_grad():
                lf = torch.nan_to_num(losses, nan=1e12, posinf=1e12)
                bad = lf > max(1e6, 100.0*float(lf.min().item()) + 1e6)
                if bad.any() and (~bad).any():
                    jb = int(torch.argmin(lf).item())
                    nb = int(bad.sum().item())
                    A[bad] = A[jb][None, :] + 0.3*torch.randn(nb, n_leaves, dtype=x_t.dtype, device=dev)
                    B[bad] = B[jb][None, :] + 0.3*torch.randn(nb, n_leaves, dtype=x_t.dtype, device=dev)
        if verbose and (i % 300 == 0 or i == 1):
            print(f"EML-pure paso {i:4d}  mejor MSE={float(torch.nan_to_num(losses, nan=1e12).min().item()):.6g}")

    with torch.no_grad():
        losses = torch.nan_to_num(forward(), nan=1e12, posinf=1e12)
        j = int(torch.argmin(losses).item())
        best_mse = float(losses[j].item())
    Aj = A[j].detach().cpu().numpy(); Bj = B[j].detach().cpu().numpy()

    def build(node):
        if node[0] == "leaf":
            i = node[1]
            return Add(Constant(float(Aj[i])), Mul(Constant(float(Bj[i])), FloatInput()))
        return Eml(build(node[1]), build(node[2]))

    return build(spec), best_mse


def eml_pure_fit(X, Y, depth=None, n_starts=None, steps=None, lr=None,
                 use_evolution_init=True, evo_generations=60, evo_pop=100, verbose=True):
    """Modo EML puro completo: estructura desde una evolución corta con catálogo
    eml-only (inicializador de pesos) o árbol completo de profundidad EML_PURE_DEPTH."""
    depth = depth if depth is not None else config.EML_PURE_DEPTH
    n_starts = n_starts if n_starts is not None else config.EML_PURE_STARTS
    steps = steps if steps is not None else config.EML_PURE_STEPS
    lr = lr if lr is not None else config.EML_PURE_LR

    spec = None; leaf_init = None
    if use_evolution_init:
        cat_un_e, cat_bin_e = build_catalogs([], ["eml"])
        best_e, _f, _h, _a, _hof = evolve(np.asarray(X), np.asarray(Y), cat_un_e, cat_bin_e,
                                          pop_size=evo_pop, generations=evo_generations,
                                          max_depth=depth, verbose=False,
                                          use_macros=False, use_hint=False, hof_k=5)
        x_d = to_device(np.asarray(X).reshape(-1, 1))
        spec_e, leaf_init_e = _eml_structure_from_tree(best_e, x_d, max_depth=max(2*depth, 8))
        if spec_e[0] == "eml":
            spec, leaf_init = spec_e, leaf_init_e
            if verbose:
                print(f"init evolutiva: estructura con {len(leaf_init)} hojas")
    if spec is None:
        spec, n_leaves = _eml_full_structure(depth)
        leaf_init = None
    else:
        n_leaves = len(leaf_init)
    return eml_pure_train(spec, n_leaves, X, Y, n_starts=n_starts, steps=steps, lr=lr,
                          leaf_init=leaf_init, verbose=verbose)
