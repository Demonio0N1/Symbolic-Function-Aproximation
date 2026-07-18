# symreg — regresión simbólica evolutiva acelerada en GPU (torch).
#
# API plana: `import symreg` expone las clases de nodo, evolve/evolve_islands,
# fine-tuning, exportación a SymPy, el operador EML y la serialización.
# Los flags de configuración viven en symreg.config (mutables en runtime).
from . import config
from .aos import OperatorManager
from .backend import device, info, to_device, to_numpy
from .benchmarks import TARGETS, correr_benchmark, tabla_markdown
from .checkpoint import load_checkpoint, save_checkpoint
from .eml import eml_pure_fit, eml_pure_train, _eml_full_structure, _eml_structure_from_tree
from .evolve import (HallOfFame, crossover, dedup_population, eval_population_batch,
                     eval_trees_matrix, evolve, evolve_islands, fitness,
                     inject_hint_population, make_hint_tree, mse_expr,
                     mse_expr_with_derivs, mutate, mutate_with_macros, nsga2_order,
                     random_node, random_terminal)
from .export import check_sympy_roundtrip, lambdify_expr, safe_derivative, to_sympy
from .finetune import fine_tune_constants, fine_tune_hof, recognize_constants
from .ops import (BINARY_MAP, SELECT_BIN, SELECT_UN, UNARY_MAP, build_catalogs)
from .tree import (Abs, Acos, Add, Asin, Atan, Beta, ChebyshevT, Constant, Cos,
                   Div, Eml, Erf, Erfc, FloatInput, Gamma, J0, J1, LegendreP,
                   Log, Max, Min, Mul, Neg, Operation, Pow, Sigmoid, Sin, Sqrt,
                   Sub, Tan, Tanh, Y0, Y1, all_subnodes, collect_constants,
                   load_tree, save_tree, simplify_tree, tree_depth,
                   tree_from_json, tree_to_json)

# Catálogos por defecto (equivalentes a los del notebook original)
CAT_UN, CAT_BIN = build_catalogs(SELECT_UN, SELECT_BIN)

__version__ = "1.0.0"
