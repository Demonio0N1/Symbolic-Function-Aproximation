# Ajuste fino de constantes (Adam sobre autograd) y reconocimiento de constantes.
import numpy as np
import torch

from . import config
from .backend import to_device
from .evolve import HallOfFame, fitness
from .tree import collect_constants


def _derivs_to_device():
    if config.DY is not None and isinstance(config.DY, np.ndarray):
        config.DY = to_device(config.DY.reshape(-1, 1))
    if config.DDY is not None and isinstance(config.DDY, np.ndarray):
        config.DDY = to_device(config.DDY.reshape(-1, 1))
    return config.DY, config.DDY


def _ft_loss(expr, params, x_t, y_t, dy_t=None, ddy_t=None):
    """Pérdida de fine-tuning, idéntica a la de la evolución (Fase 1.4):
    LAMBDA_Y*MSE(y) + LAMBDA_Yp*MSE(y') + LAMBDA_Ypp*MSE(y'') si hay derivadas."""
    yhat = expr.eval_backend(x_t, params)
    loss = config.LAMBDA_Y*torch.mean((yhat - y_t)**2)
    if (dy_t is not None) or (ddy_t is not None):
        x_req = x_t.clone().detach().requires_grad_(True)
        yhat2 = expr.eval_backend(x_req, params)
        (dyhat,) = torch.autograd.grad(yhat2, x_req, grad_outputs=torch.ones_like(yhat2),
                                       create_graph=True, retain_graph=True, allow_unused=True)
        if dyhat is None: dyhat = torch.zeros_like(yhat2)
        if dy_t is not None:
            loss = loss + config.LAMBDA_Yp*torch.mean((dyhat - dy_t)**2)
        if ddy_t is not None:
            (ddyhat,) = torch.autograd.grad(dyhat, x_req, grad_outputs=torch.ones_like(dyhat),
                                            create_graph=True, retain_graph=True, allow_unused=True)
            if ddyhat is None: ddyhat = torch.zeros_like(yhat2)
            loss = loss + config.LAMBDA_Ypp*torch.mean((ddyhat - ddy_t)**2)
    return loss


def fine_tune_constants(expr, X, Y, steps=300, lr=5e-3, verbose=True):
    """Ajusta las constantes de UNA expresión con la misma pérdida que la evolución."""
    nodes = collect_constants(expr)
    if not nodes:
        if verbose: print("No hay constantes que ajustar.")
        return expr
    Xa = np.asarray(X)
    x_t = to_device(Xa.reshape(len(Xa), -1)); y_t = to_device(np.asarray(Y).reshape(-1, 1))
    dy_t, ddy_t = _derivs_to_device() if config.USE_DERIV_LOSS else (None, None)
    params = {c._id: torch.tensor([c.value], dtype=x_t.dtype, device=x_t.device,
                                  requires_grad=True) for c in nodes}
    opt = torch.optim.Adam(params.values(), lr=lr)
    for i in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        loss = _ft_loss(expr, params, x_t, y_t, dy_t, ddy_t)
        loss.backward(); opt.step()
        if verbose and (i % 50 == 0 or i == 1):
            print(f"FT paso {i:4d}  loss={loss.item():.6g}")
    for c in nodes:
        v = float(params[c._id].detach().cpu().item())
        if np.isfinite(v): c.value = v   # si Adam divergió, conserva el valor previo
    return expr


def fine_tune_hof(hof, X, Y, steps=250, lr=5e-3, verbose=True, alpha=1e-3):
    """Fase 2.4: ajusta las constantes de TODAS las expresiones del Hall of Fame
    en paralelo con un único Adam (pérdida sumada, parámetros disjuntos)."""
    Xa = np.asarray(X)
    x_t = to_device(Xa.reshape(len(Xa), -1)); y_t = to_device(np.asarray(Y).reshape(-1, 1))
    dy_t, ddy_t = _derivs_to_device() if config.USE_DERIV_LOSS else (None, None)
    entries = []; all_params = []
    for f, e in hof.items:
        e = e.clone()
        nodes = collect_constants(e)
        params = {c._id: torch.tensor([c.value], dtype=x_t.dtype, device=x_t.device,
                                      requires_grad=True) for c in nodes}
        entries.append((e, nodes, params))
        all_params += list(params.values())
    if not all_params:
        return hof

    opt = torch.optim.Adam(all_params, lr=lr)
    for i in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        total = None
        for e, nodes, params in entries:
            l = _ft_loss(e, params, x_t, y_t, dy_t, ddy_t)
            total = l if total is None else total + l
        total.backward(); opt.step()
        if verbose and (i % 50 == 0 or i == 1):
            print(f"FT-HOF paso {i:4d}  loss_total={total.item():.6g}")

    new_hof = HallOfFame(k=hof.k)
    for e, nodes, params in entries:
        for c in nodes:
            v = float(params[c._id].detach().cpu().item())
            if np.isfinite(v): c.value = v
        new_hof.update(e, fitness(e, x_t, y_t, alpha=alpha))
    return new_hof


def _const_candidates(v):
    """Candidatos 'bonitos': entero cercano, fracción simple, múltiplos de pi/E."""
    import sympy as sp
    out = []
    r = round(v)
    if abs(v - r) < 0.05: out.append(float(r))
    try:
        fr = sp.Rational(v).limit_denominator(12)
        out.append(float(fr))
    except Exception:
        pass
    try:
        cand = sp.nsimplify(v, [sp.pi, sp.E], tolerance=0.01, rational=False)
        if cand.is_number: out.append(float(cand))
    except Exception:
        pass
    uniq = []
    for c in out:
        if np.isfinite(c) and abs(c - v) < 0.05*max(1.0, abs(v)) and all(c != u for u in uniq):
            uniq.append(c)
    return uniq


def recognize_constants(expr, X, Y, alpha=1e-3, verbose=False):
    """Fase 3.2: redondeo/nsimplify tras el fine-tuning (2.9999*sin(x) -> 3*sin(x));
    cada sustitución se acepta solo si el fitness (datos completos) NO empeora."""
    Xa = np.asarray(X)
    x_d = to_device(Xa.reshape(len(Xa), -1)); y_d = to_device(np.asarray(Y).reshape(-1, 1))
    e2 = expr.clone()
    base = fitness(e2, x_d, y_d, alpha=alpha)
    for c in collect_constants(e2):
        for cand in _const_candidates(c.value):
            if cand == c.value: continue
            old = c.value; c.value = cand
            f = fitness(e2, x_d, y_d, alpha=alpha)
            if f <= base + 1e-12:
                base = f
                if verbose: print(f"  constante {old:.8g} -> {cand:.8g}  (fit {f:.6g})")
                break
            c.value = old
    return e2
