# Test de ida y vuelta motor <-> SymPy (Fase 1.1): la expresión SymPy generada
# por to_sympy() debe coincidir numéricamente con la evaluación del motor.
import random

import pytest


def test_roundtrip_polinomio_seno(eng):
    # 2*x^3 + 3*sin(x) + 1 (el caso de prueba del proyecto)
    e = eng.Add(
        eng.Mul(eng.Constant(2.0), eng.Pow(eng.FloatInput(), eng.Constant(3.0))),
        eng.Add(eng.Mul(eng.Constant(3.0), eng.Sin(eng.FloatInput())), eng.Constant(1.0)),
    )
    assert eng.check_sympy_roundtrip(e, lo=-5.0, hi=5.0)


def test_roundtrip_trig_log_div(eng):
    # cos(x)/(x+2) + log(|x|)
    e = eng.Add(
        eng.Div(eng.Cos(eng.FloatInput()), eng.Add(eng.FloatInput(), eng.Constant(2.0))),
        eng.Log(eng.FloatInput()),
    )
    assert eng.check_sympy_roundtrip(e, lo=0.5, hi=5.0)


def test_roundtrip_sqrt_abs_neg(eng):
    # -sqrt(|x|) + |x - 3|
    e = eng.Add(
        eng.Neg(eng.Sqrt(eng.FloatInput())),
        eng.Abs(eng.Sub(eng.FloatInput(), eng.Constant(3.0))),
    )
    assert eng.check_sympy_roundtrip(e, lo=-4.0, hi=4.0)


def test_roundtrip_tanh_sigmoid_atan(eng):
    e = eng.Add(
        eng.Tanh(eng.FloatInput()),
        eng.Mul(eng.Sigmoid(eng.FloatInput()), eng.Atan(eng.FloatInput())),
    )
    assert eng.check_sympy_roundtrip(e, lo=-3.0, hi=3.0)


def test_roundtrip_especiales(eng):
    # erf(x) + J0(x) + Gamma en dominio positivo
    e = eng.Add(
        eng.Add(eng.Erf(eng.FloatInput()), eng.J0(eng.FloatInput())),
        eng.Gamma(eng.FloatInput()),
    )
    assert eng.check_sympy_roundtrip(e, lo=0.5, hi=4.0)


def test_roundtrip_chebyshev_legendre(eng):
    e = eng.Add(
        eng.ChebyshevT(eng.FloatInput(), n=3),
        eng.LegendreP(eng.FloatInput(), n=4),
    )
    assert eng.check_sympy_roundtrip(e, lo=-1.0, hi=1.0)


def test_roundtrip_min_max(eng):
    e = eng.Min(eng.FloatInput(), eng.Max(eng.Constant(0.5), eng.Sin(eng.FloatInput())))
    assert eng.check_sympy_roundtrip(e, lo=-3.0, hi=3.0)


def test_roundtrip_arboles_aleatorios(eng):
    # Fuzz con catálogo benigno: los árboles aleatorios también deben coincidir
    random.seed(123)
    cat_un, cat_bin = eng.build_catalogs(["sin", "cos", "abs", "neg"], ["add", "sub", "mul"])
    aos = eng.OperatorManager(cat_un, cat_bin)
    ok = 0
    for _ in range(20):
        tree, _ = eng.random_node(aos, max_depth=4)
        assert eng.check_sympy_roundtrip(tree, lo=-3.0, hi=3.0), tree.to_text()
        ok += 1
    assert ok == 20


def test_texto_prefijo_no_corrompe(eng):
    # El bug original: "pow" -> "**" via str.replace corrompía la notación.
    # Con la conversión recursiva la expresión resultante debe ser válida siempre.
    e = eng.Pow(eng.Add(eng.FloatInput(), eng.Constant(1.0)), eng.Constant(2.0))
    f_sym, xs = eng.to_sympy(e)
    import sympy as sp

    assert sp.simplify(f_sym - (xs + 1) ** 2) == 0
