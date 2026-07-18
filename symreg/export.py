# Exportación a SymPy y test de ida y vuelta motor <-> SymPy.
import numpy as np
import sympy as sp

from .backend import to_device, to_numpy
from .tree import FloatInput, all_subnodes


def to_sympy(best):
    """Conversión recursiva a SymPy (Fase 1.1). Devuelve (expresión, símbolo(s)):
    un Symbol para el caso univariable, una lista para multivariable."""
    n = max([nd.idx for nd, _, _ in all_subnodes(best) if isinstance(nd, FloatInput)] + [0]) + 1
    if n == 1:
        x = sp.Symbol('x', real=True)
        return best.to_sympy(x), x
    xs = [sp.Symbol('x' if i == 0 else f'x{i}', real=True) for i in range(n)]
    return best.to_sympy(xs), xs


def check_sympy_roundtrip(expr, lo=-3.0, hi=3.0, n=64, rtol=1e-3, atol=1e-3, min_cover=0.5):
    """Test de ida y vuelta: la expresión SymPy y la del motor deben coincidir
    numéricamente en una malla. Devuelve True/False, o None si las guardas
    numéricas del motor dejan demasiados puntos sin comparar (no concluyente)."""
    f_sym, xs = to_sympy(expr)
    multi = isinstance(xs, (list, tuple))
    if multi:
        rng = np.random.default_rng(0)
        G = rng.uniform(lo, hi, size=(n, len(xs))).astype(np.float64)
    else:
        G = np.linspace(lo, hi, n).astype(np.float64).reshape(-1, 1)
    f_num = sp.lambdify(xs, f_sym, modules=["scipy", "numpy"])
    with np.errstate(all="ignore"):
        try:
            args = list(G.T) if multi else [G.reshape(-1)]
            y_sym = np.asarray(f_num(*args), dtype=np.complex128) * np.ones(n, dtype=np.complex128)
        except Exception:
            return None
    y_eng = to_numpy(expr.eval_backend(to_device(G))).reshape(-1).astype(np.float64)
    es_real = np.abs(y_sym.imag) < 1e-9
    y_real = y_sym.real
    mask = es_real & np.isfinite(y_real) & np.isfinite(y_eng) & (np.abs(y_eng) < 1e6)
    if mask.sum() < n*min_cover:
        return None
    return bool(np.allclose(y_real[mask], y_eng[mask], rtol=rtol, atol=atol))


def safe_derivative(expr, var, order=1):
    """Derivada simbólica robusta: fuerza forma explícita y anula DiracDelta
    (la derivada distribucional de |u| no es evaluable numéricamente)."""
    d = sp.diff(expr, var, order)
    d = sp.simplify(d.doit())
    d = d.replace(sp.DiracDelta, lambda *a: sp.S.Zero)
    return d


def lambdify_expr(xs, f_sym):
    """Lambdify con el printer NumPy real (maneja Piecewise/sign/Min/Max) + SciPy."""
    return sp.lambdify(xs, f_sym, modules=["scipy", "numpy"])
