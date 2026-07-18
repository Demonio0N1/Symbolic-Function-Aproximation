# Configuración global del motor (flags agrupados, en español como el notebook).
# Los módulos leen estos valores EN TIEMPO DE EJECUCIÓN (import symreg.config as config),
# así que pueden modificarse antes o entre corridas.

# ===== Dispositivo / precisión =====
DTYPE = "float32"      # "float32" o "float64" para la evaluación en GPU (float64: ~x1,26 más lento)
USE_TF32 = True        # TF32 en GPUs Ampere+ (solo afecta a matmul; inocuo en estas ops)

# ===== Rendimiento (Fase 2) =====
BATCH_EVAL = True      # evalúa la población por lotes en GPU (una sincronización por lote)
FITNESS_CACHE = True   # memoiza fitness por individuo y generación en la ruta clásica (BATCH_EVAL=False)
SIMPLIFY_EVERY = 0     # simplificación algebraica ligera de la población cada N generaciones (0 = off)
DEDUP_EVERY = 0        # deduplicación semántica cada N generaciones (0 = off)
AOS_UPDATE_CHUNK = 1   # hijos por sub-lote entre actualizaciones AOS.
                       #  1  = recompensas inmediatas por hijo, como el algoritmo original
                       #  16 = máxima velocidad (~x16): recompensas levemente diferidas

# ===== Calidad de búsqueda (Fase 3) =====
AOS_XOVER_REWARD = True   # recompensa AOS también al crossover (ops del subárbol donado)
AOS_REWARD_NORM  = True   # normaliza las recompensas por la varianza de Y (escala-invariante)
SELECTION = "alpha"       # "alpha": mse + alpha*n_nodos | "nsga2": frente de Pareto precisión-complejidad
PRETTY_CONSTANTS = True   # to_sympy reconoce enteros/fracciones/pi/E con tolerancia estricta

# ===== Operador EML (Fase 4) — eml(x,y) = exp(x) - ln(y), arXiv:2603.21852 =====
EML_MODE = "off"          # "off" | "operator" (eml en el catálogo/AOS) | "pure" (gramática solo-eml)
EML_SYMPY_EXPAND = True   # to_sympy: True = forma exp/ln expandida (evaluable); False = eml(u,v) compacta
EML_PURE_DEPTH  = 4       # profundidad de la estructura completa del modo puro (sin init evolutiva)
EML_PURE_STARTS = 256     # multi-starts entrenados en paralelo (dimensión extra del tensor)
EML_PURE_STEPS  = 1500
EML_PURE_LR     = 5e-3

# ===== Pérdida con derivadas (opcional) =====
USE_DERIV_LOSS = False
LAMBDA_Y, LAMBDA_Yp, LAMBDA_Ypp = 1.0, 0.5, 0.2
DY = None   # y'(x)  (np.ndarray; evolve/fine-tune lo llevan al dispositivo)
DDY = None  # y''(x)

# ===== Estado interno =====
N_VARS = 1  # nº de variables de entrada; lo fija evolve() según la forma de X


def set_seed(seed):
    """Reproducibilidad completa: random, numpy y torch (CPU y CUDA)."""
    import random as _random

    import numpy as _np
    import torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
