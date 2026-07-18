# Catálogos de operadores y mapeos nombre -> clase / función vectorizada.
from . import backend
from .tree import (Abs, Acos, Add, Asin, Atan, Beta, ChebyshevT, Cos, Div,
                   Eml, Erf, Erfc, Gamma, J0, J1, LegendreP, Log, Max, Min,
                   Mul, Neg, Pow, Sigmoid, Sin, Sqrt, Sub, Tan, Tanh, Y0, Y1)

SELECT_UN = ["sin", "cos", "log", "sqrt", "abs", "neg"]   # añade: "tan","tanh","sigmoid","asin","acos","atan","erf","erfc","gamma","j0","j1","y0","y1","chebyshevT","legendreP"
SELECT_BIN = ["add", "sub", "mul", "div", "pow"]          # añade: "min","max","beta","eml"
CHEB_DEGREES = [2, 3, 4]
LEG_DEGREES = [2, 3, 4]

UNARY_MAP = {"neg": Neg, "abs": Abs, "sin": Sin, "cos": Cos, "tan": Tan, "log": Log,
             "sqrt": Sqrt, "tanh": Tanh, "sigmoid": Sigmoid, "asin": Asin, "acos": Acos,
             "atan": Atan, "erf": Erf, "erfc": Erfc, "gamma": Gamma, "j0": J0, "j1": J1,
             "y0": Y0, "y1": Y1, "chebyshevT": ChebyshevT, "legendreP": LegendreP}
BINARY_MAP = {"add": Add, "sub": Sub, "mul": Mul, "div": Div, "pow": Pow,
              "min": Min, "max": Max, "beta": Beta, "eml": Eml}

# Funciones vectorizadas por nombre (para el intérprete plano por lotes)
import torch as _torch

VOPS_UN = {"neg": lambda a: -a, "abs": backend.abs_, "sin": backend.sin_, "cos": backend.cos_,
           "tan": backend.tan_, "log": backend.log_, "sqrt": backend.sqrt_,
           "tanh": backend.tanh_, "sigmoid": backend.sigmoid_, "asin": backend.asin_,
           "acos": backend.acos_, "atan": backend.atan_, "erf": backend.erf_,
           "erfc": backend.erfc_, "gamma": backend.gamma_, "j0": backend.j0_,
           "j1": backend.j1_, "y0": backend.y0_, "y1": backend.y1_}
VOPS_BIN = {"add": backend.add_, "sub": backend.sub_, "mul": backend.mul_,
            "div": backend.div_, "pow": backend.pow_, "min": _torch.minimum,
            "max": _torch.maximum, "beta": backend.beta_, "eml": backend.eml_}


def vop_unary(key):
    name, n = key
    if name == "chebyshevT": return (lambda a, n=n: backend.chebyshevT_(a, n))
    if name == "legendreP":  return (lambda a, n=n: backend.legendreP_(a, n))
    return VOPS_UN[name]


def build_catalogs(select_un, select_bin, cheb=CHEB_DEGREES, leg=LEG_DEGREES):
    cat_un = []; cat_bin = []
    for name in select_un:
        if name == "chebyshevT":
            for n in cheb: cat_un.append((f"chebyshevT_{n}", lambda o1, n=n: ChebyshevT(o1=o1, n=n)))
        elif name == "legendreP":
            for n in leg: cat_un.append((f"legendreP_{n}", lambda o1, n=n: LegendreP(o1=o1, n=n)))
        else:
            cat_un.append((name, lambda o1, cls=UNARY_MAP[name]: cls(o1=o1)))
    for name in select_bin:
        cat_bin.append((name, lambda o1, o2, cls=BINARY_MAP[name]: cls(o1=o1, o2=o2)))
    return cat_un, cat_bin
