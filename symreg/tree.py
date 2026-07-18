# Árboles de expresión: clases de nodo, utilidades, simplificación y
# serialización JSON (Fase 6.3).
import numpy as np
import sympy as sp
import torch

from . import config
from .backend import (abs_, acos_, add_, asin_, atan_, beta_, chebyshevT_,
                      cos_, div_, eml_, erf_, erfc_, gamma_, j0_, j1_,
                      legendreP_, log_, mul_, pow_, sigmoid_, sin_, sqrt_,
                      sub_, tan_, tanh_, to_device, to_numpy, y0_, y1_)


class Operation:
    def __init__(self, name=None, o1=None, o2=None):
        self.name = name; self.o1 = o1; self.o2 = o2
    def eval_backend(self, x, const_map=None): raise NotImplementedError
    def to_text(self): raise NotImplementedError
    def to_sympy(self, xs): raise NotImplementedError  # conversión recursiva nodo -> SymPy
    def clone(self): raise NotImplementedError


class Constant(Operation):
    def __init__(self, value=None):
        super().__init__(name="const")
        if value is None:
            import random
            value = random.uniform(-5, 5)
        self.value = float(value); self._id = id(self)

    def eval_backend(self, x, const_map=None):
        if const_map is not None and self._id in const_map:
            c = const_map[self._id]
            return c + (0*c)
        if getattr(x, "ndim", 1) == 2 and x.shape[1] > 1:
            return (x[:, :1]*0) + self.value   # multivariable: difunde a (n,1)
        return (x*0) + self.value

    def to_text(self):
        v = self.value
        if not np.isfinite(v): return f"{v}"
        return str(int(round(v))) if abs(v-round(v)) < 1e-10 else f"{v:.5g}"

    def to_sympy(self, xs):
        v = self.value
        if not np.isfinite(v): return sp.nan
        if abs(v - round(v)) < 1e-10: return sp.Integer(round(v))
        if config.PRETTY_CONSTANTS:
            # reconocimiento estricto (1e-9) de fracciones simples y múltiplos de pi/E
            try:
                fr = sp.Rational(v).limit_denominator(12)
                if abs(float(fr) - v) < 1e-9: return fr
                cand = sp.nsimplify(v, [sp.pi, sp.E], tolerance=1e-9, rational=False)
                if cand.is_number and abs(float(cand) - v) < 1e-9 and len(str(cand)) <= 12:
                    return cand
            except Exception:
                pass
        return sp.Float(v)

    def clone(self): return Constant(self.value)


class FloatInput(Operation):
    # multivariable: idx=0 es el caso clásico "x"
    def __init__(self, idx=0):
        super().__init__(name="x"); self.idx = int(idx)

    def eval_backend(self, x, const_map=None):
        if getattr(x, "ndim", 1) == 2 and x.shape[1] > 1:
            return x[:, self.idx:self.idx+1]
        return x

    def to_text(self): return "x" if self.idx == 0 else f"x{self.idx}"

    def to_sympy(self, xs):
        if isinstance(xs, (list, tuple)): return xs[self.idx]
        return xs

    def clone(self): return FloatInput(self.idx)


class Unary(Operation):
    def __init__(self, name, o1): super().__init__(name=name, o1=o1)


class Binary(Operation):
    def __init__(self, name, o1, o2): super().__init__(name=name, o1=o1, o2=o2)


def _unary(nombre, fn, texto, sympy_fn):
    """Fábrica de clases unarias: evita repetir 5 líneas por operador."""
    class _U(Unary):
        def __init__(self, o1=None): super().__init__(nombre, o1)
        def eval_backend(self, x, c=None): return fn(self.o1.eval_backend(x, c))
        def to_text(self): return texto.format(self.o1.to_text())
        def to_sympy(self, xs): return sympy_fn(self.o1.to_sympy(xs))
        def clone(self): return _U(self.o1.clone())
    _U.__name__ = _U.__qualname__ = nombre.capitalize()
    return _U


def _binary(nombre, fn, texto, sympy_fn):
    class _B(Binary):
        def __init__(self, o1=None, o2=None): super().__init__(nombre, o1, o2)
        def eval_backend(self, x, c=None):
            return fn(self.o1.eval_backend(x, c), self.o2.eval_backend(x, c))
        def to_text(self): return texto.format(self.o1.to_text(), self.o2.to_text())
        def to_sympy(self, xs): return sympy_fn(self.o1.to_sympy(xs), self.o2.to_sympy(xs))
        def clone(self): return _B(self.o1.clone(), self.o2.clone())
    _B.__name__ = _B.__qualname__ = nombre.capitalize()
    return _B


# Unarias (la semántica sympy refleja las guardas del motor: log(|u|), sqrt(|u|), |Gamma|)
Neg     = _unary("neg", lambda a: -a, "-({})", lambda u: -u)
Abs     = _unary("abs", abs_, "abs({})", sp.Abs)
Sin     = _unary("sin", sin_, "sin({})", sp.sin)
Cos     = _unary("cos", cos_, "cos({})", sp.cos)
Tan     = _unary("tan", tan_, "tan({})", sp.tan)
Log     = _unary("log", log_, "log(|{}|+eps)", lambda u: sp.log(sp.Abs(u)))
Sqrt    = _unary("sqrt", sqrt_, "sqrt(|{})", lambda u: sp.sqrt(sp.Abs(u)))
Tanh    = _unary("tanh", tanh_, "tanh({})", sp.tanh)
Sigmoid = _unary("sigmoid", sigmoid_, "sigmoid({})", lambda u: 1/(1 + sp.exp(-u)))
Asin    = _unary("asin", asin_, "asin({})", sp.asin)
Acos    = _unary("acos", acos_, "acos({})", sp.acos)
Atan    = _unary("atan", atan_, "atan({})", sp.atan)
Erf     = _unary("erf", erf_, "erf({})", sp.erf)
Erfc    = _unary("erfc", erfc_, "erfc({})", sp.erfc)
J0      = _unary("j0", j0_, "J0({})", lambda u: sp.besselj(0, u))
J1      = _unary("j1", j1_, "J1({})", lambda u: sp.besselj(1, u))
Y0      = _unary("y0", y0_, "Y0({})", lambda u: sp.bessely(0, u))
Y1      = _unary("y1", y1_, "Y1({})", lambda u: sp.bessely(1, u))
Gamma   = _unary("gamma", gamma_, "Gamma({})", lambda u: sp.Abs(sp.gamma(u)))


class ChebyshevT(Unary):
    def __init__(self, o1=None, n=2): super().__init__("chebyshevT", o1); self.n = int(n)
    def eval_backend(self, x, c=None): return chebyshevT_(self.o1.eval_backend(x, c), self.n)
    def to_text(self): return f"T_{self.n}({self.o1.to_text()})"
    def to_sympy(self, xs): return sp.chebyshevt(self.n, self.o1.to_sympy(xs))
    def clone(self): return ChebyshevT(self.o1.clone(), self.n)


class LegendreP(Unary):
    def __init__(self, o1=None, n=2): super().__init__("legendreP", o1); self.n = int(n)
    def eval_backend(self, x, c=None): return legendreP_(self.o1.eval_backend(x, c), self.n)
    def to_text(self): return f"P_{self.n}({self.o1.to_text()})"
    def to_sympy(self, xs): return sp.legendre(self.n, self.o1.to_sympy(xs))
    def clone(self): return LegendreP(self.o1.clone(), self.n)


# Binarias
Add = _binary("add", add_, "({} + {})", lambda u, v: u + v)
Sub = _binary("sub", sub_, "({} - {})", lambda u, v: u - v)
Mul = _binary("mul", mul_, "({} * {})", lambda u, v: u * v)
Div = _binary("div", div_, "({} / {})", lambda u, v: u / v)
Min = _binary("min", torch.minimum, "min({}, {})", sp.Min)
Max = _binary("max", torch.maximum, "max({}, {})", sp.Max)
Beta = _binary("beta", beta_, "Beta({}, {})", sp.beta)


class Pow(Binary):
    def __init__(self, o1=None, o2=None): super().__init__("pow", o1, o2)
    def eval_backend(self, x, c=None):
        return pow_(self.o1.eval_backend(x, c), self.o2.eval_backend(x, c))
    def to_text(self): return f"({self.o1.to_text()} ^ {self.o2.to_text()})"
    def to_sympy(self, xs):
        # el motor recorta el exponente a [-5,5]; si es constante, lo reflejamos
        e = self.o2.to_sympy(xs)
        if isinstance(self.o2, Constant):
            ev = min(5.0, max(-5.0, self.o2.value))
            e = sp.Integer(round(ev)) if abs(ev - round(ev)) < 1e-10 else sp.Float(ev)
        return self.o1.to_sympy(xs) ** e
    def clone(self): return Pow(self.o1.clone(), self.o2.clone())


class Eml(Binary):
    # Fase 4.1: eml(u,v) = exp(u) - ln(v) (Odrzywolek, arXiv:2603.21852)
    def __init__(self, o1=None, o2=None): super().__init__("eml", o1, o2)
    def eval_backend(self, x, c=None):
        return eml_(self.o1.eval_backend(x, c), self.o2.eval_backend(x, c))
    def to_text(self): return f"eml({self.o1.to_text()}, {self.o2.to_text()})"
    def to_sympy(self, xs):
        u = self.o1.to_sympy(xs); v = self.o2.to_sympy(xs)
        if config.EML_SYMPY_EXPAND:
            return sp.exp(u) - sp.log(sp.Abs(v))   # expandida a exp/ln (evaluable)
        return sp.Function("eml")(u, v)            # compacta, solo para display
    def clone(self): return Eml(self.o1.clone(), self.o2.clone())


# ===== Utilidades =====
def all_subnodes(node):
    out = []
    def rec(cur, parent=None, attr=None):
        out.append((cur, parent, attr))
        if cur.o1 is not None: rec(cur.o1, cur, "o1")
        if cur.o2 is not None: rec(cur.o2, cur, "o2")
    rec(node, None, None)
    return out


def collect_constants(expr):
    return [n for n, _, _ in all_subnodes(expr) if isinstance(n, Constant)]


def tree_depth(node):
    d1 = tree_depth(node.o1) if node.o1 is not None else 0
    d2 = tree_depth(node.o2) if node.o2 is not None else 0
    return 1 + max(d1, d2)


# ===== Simplificación algebraica ligera (Fase 2.5) =====
def _const_value(node):
    # Valor numérico de un subárbol sin x (respeta las guardas numéricas del motor)
    xtmp = to_device(np.zeros((1, 1), dtype=np.float32))
    return float(to_numpy(node.eval_backend(xtmp)).reshape(-1)[0])


def _is_const(node, val, tol=1e-12):
    return isinstance(node, Constant) and abs(node.value - val) <= tol


def _has_no_x(node):
    if isinstance(node, FloatInput): return False
    if node.o1 is not None and not _has_no_x(node.o1): return False
    if node.o2 is not None and not _has_no_x(node.o2): return False
    return True


def simplify_tree(node):
    """Simplificación ligera: plegado de constantes (sin(0.5), 2*3, ...) e
    identidades x*1, x+0, x-0, x/1, x^1, x^0, u*0, -(-u). Devuelve árbol nuevo."""
    node = node.clone()
    def rec(cur):
        if cur.o1 is not None: cur.o1 = rec(cur.o1)
        if cur.o2 is not None: cur.o2 = rec(cur.o2)
        if not isinstance(cur, (Constant, FloatInput)) and _has_no_x(cur):
            try:
                v = _const_value(cur)
                if np.isfinite(v) and abs(v) < 1e12:
                    return Constant(v)
            except Exception:
                pass
        name = cur.name
        if name == "add":
            if _is_const(cur.o1, 0.0): return cur.o2
            if _is_const(cur.o2, 0.0): return cur.o1
        elif name == "sub":
            if _is_const(cur.o2, 0.0): return cur.o1
        elif name == "mul":
            if _is_const(cur.o1, 1.0): return cur.o2
            if _is_const(cur.o2, 1.0): return cur.o1
            if _is_const(cur.o1, 0.0) or _is_const(cur.o2, 0.0): return Constant(0.0)
        elif name == "div":
            if _is_const(cur.o2, 1.0): return cur.o1
        elif name == "pow":
            if _is_const(cur.o2, 1.0): return cur.o1
            if _is_const(cur.o2, 0.0): return Constant(1.0)
        elif name == "neg" and cur.o1.name == "neg":
            return cur.o1.o1
        return cur
    return rec(node)


# ===== Serialización JSON (Fase 6.3) =====
def tree_to_json(node):
    """Serializa un árbol a un dict JSON-compatible."""
    d = {"op": node.name}
    if isinstance(node, Constant):
        d["value"] = node.value
    elif isinstance(node, FloatInput):
        d["idx"] = node.idx
    else:
        if getattr(node, "n", None) is not None and node.name in ("chebyshevT", "legendreP"):
            d["n"] = node.n
        if node.o1 is not None: d["o1"] = tree_to_json(node.o1)
        if node.o2 is not None: d["o2"] = tree_to_json(node.o2)
    return d


def tree_from_json(d):
    """Reconstruye un árbol desde el dict producido por tree_to_json."""
    from .ops import BINARY_MAP, UNARY_MAP
    op = d["op"]
    if op == "const":
        return Constant(d["value"])
    if op == "x":
        return FloatInput(d.get("idx", 0))
    if op in ("chebyshevT", "legendreP"):
        cls = UNARY_MAP[op]
        return cls(tree_from_json(d["o1"]), n=d.get("n", 2))
    if op in UNARY_MAP:
        return UNARY_MAP[op](tree_from_json(d["o1"]))
    if op in BINARY_MAP:
        return BINARY_MAP[op](tree_from_json(d["o1"]), tree_from_json(d["o2"]))
    raise ValueError(f"operador desconocido en JSON: {op!r}")


def save_tree(node, path):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree_to_json(node), f, indent=1)


def load_tree(path):
    import json
    with open(path, encoding="utf-8") as f:
        return tree_from_json(json.load(f))
