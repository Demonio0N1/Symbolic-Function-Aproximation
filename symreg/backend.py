# Backend numérico (torch, CUDA si está disponible) y guardas numéricas.
#
# Decisión de la Fase 6 (con datos): se retiró el backend CuPy. torch cubre CPU
# y CUDA con el mismo código, y todo el camino rápido de la Fase 2 (intérprete
# plano por lotes, x15,8 por generación) es torch; la ruta CuPy solo disponía
# de la evaluación clásica recursiva (~x10 más lenta) y duplicaba cada helper.
import numpy as np
import torch

from . import config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if config.USE_TF32:
    torch.set_float32_matmul_precision("high")


def info():
    """Descripción del dispositivo activo (para logs)."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"


def to_device(arr):
    dt = torch.float64 if config.DTYPE == "float64" else torch.float32
    return torch.as_tensor(arr, dtype=dt, device=device)


def to_numpy(x):
    return x.detach().cpu().numpy()


# ===== Guardas numéricas =====
def finfo_bounds(x):
    f = torch.finfo(x.dtype)
    return f.min*0.25, f.max*0.25


def clip_safe(x, lo=None, hi=None):
    lo_f, hi_f = finfo_bounds(x)
    if lo is None: lo = lo_f
    if hi is None: hi = hi_f
    lo = max(lo_f, min(float(lo), hi_f)); hi = min(hi_f, max(float(hi), lo_f))
    return torch.clamp(x, min=float(lo), max=float(hi))


def nan_to_num_safe(x, pos=None, neg=None):
    lo_f, hi_f = finfo_bounds(x)
    if pos is None: pos = hi_f
    if neg is None: neg = lo_f
    return torch.nan_to_num(x, nan=0.0, posinf=float(pos), neginf=float(neg))


def exp_safe(x):
    max_in = 80.0 if x.dtype == torch.float32 else 700.0
    return clip_safe(nan_to_num_safe(torch.exp(clip_safe(x, -max_in, max_in))))


# ===== Operadores elementales =====
def ones_like(x): return torch.ones_like(x)
def abs_(x): return torch.abs(x)
def sin_(x): return torch.sin(x)
def cos_(x): return torch.cos(x)


def tan_(x):
    return clip_safe(nan_to_num_safe(torch.tan(clip_safe(x, -50.0, 50.0))))


def log_(x): return torch.log(abs_(x) + 1e-8)
def sqrt_(x): return torch.sqrt(abs_(x) + 1e-8)
def tanh_(x): return torch.tanh(x)
def sigmoid_(x): return torch.sigmoid(x)
def asin_(x): return torch.asin(clip_safe(x, -1+1e-6, 1-1e-6))
def acos_(x): return torch.acos(clip_safe(x, -1+1e-6, 1-1e-6))
def atan_(x): return torch.atan(x)


def add_(a, b): return a + b
def sub_(a, b): return a - b
def mul_(a, b): return a * b


def div_(a, b, eps=1e-8):
    b = torch.where(torch.abs(b) < eps, torch.sign(b)*eps + (b == 0)*eps, b)
    return clip_safe(nan_to_num_safe(a/b))


def pow_(a, b):
    a = clip_safe(a, -1e3, 1e3); b = clip_safe(b, -5, 5)
    return clip_safe(nan_to_num_safe(torch.pow(a, b)))


def eml_(a, b):
    # eml(a,b) = exp(a) - ln(b), con las mismas guardas que exp_safe y log_ (Fase 4)
    return exp_safe(a) - log_(b)


# ===== Especiales =====
def gamma_(x):
    z = clip_safe(x, -170.0, 170.0)
    return clip_safe(nan_to_num_safe(exp_safe(torch.lgamma(z))))


def beta_(a, b):
    a = clip_safe(a, 1e-6, 170.0); b = clip_safe(b, 1e-6, 170.0)
    return clip_safe(exp_safe(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b)), 0.0, None)


def erf_(x): return torch.special.erf(x)
def erfc_(x): return torch.special.erfc(x)


def _scipy_fallback(fn_name, x):
    import scipy.special as sc
    vals = getattr(sc, fn_name)(to_numpy(x))
    return to_device(np.nan_to_num(vals, nan=0.0))


def j0_(x):
    if hasattr(torch.special, "bessel_j0"):
        return torch.special.bessel_j0(x)
    return _scipy_fallback("j0", x)


def j1_(x):
    if hasattr(torch.special, "bessel_j1"):
        return torch.special.bessel_j1(x)
    return _scipy_fallback("j1", x)


def y0_(x):
    x = clip_safe(x, 1e-6, None)
    if hasattr(torch.special, "bessel_y0"):
        return torch.special.bessel_y0(x)
    return _scipy_fallback("y0", x)


def y1_(x):
    x = clip_safe(x, 1e-6, None)
    if hasattr(torch.special, "bessel_y1"):
        return torch.special.bessel_y1(x)
    return _scipy_fallback("y1", x)


def chebyshevT_(x, n):
    if n == 0: return ones_like(x)
    if n == 1: return x
    T0 = ones_like(x); T1 = x
    for k in range(2, n+1):
        T0, T1 = T1, 2.0*x*T1 - T0
    return T1


def legendreP_(x, n):
    if n == 0: return ones_like(x)
    if n == 1: return x
    P0 = ones_like(x); P1 = x
    for k in range(2, n+1):
        P0, P1 = P1, ((2*k-1)*x*P1 - (k-1)*P0)/k
    return P1
