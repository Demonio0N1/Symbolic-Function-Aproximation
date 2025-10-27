===========================================================
 Function-Aprox — Búsqueda Simbólica de Funciones (GPU + Derivadas)
===========================================================

Este proyecto implementa un sistema evolutivo simbólico para aproximar una función matemática a partir de datos (X, Y),
generando una expresión analítica interpretable (por ejemplo: 2*x**3 + 3*sin(x) + 1).

Permite usar derivadas (f', f''), optimización fina de constantes, y aceleración GPU con CuPy o Torch CUDA.

-----------------------------------------------------------
1. DERIVADAS OPCIONALES (USE_DERIV_LOSS)
-----------------------------------------------------------

Si USE_DERIV_LOSS=True, el sistema incorpora derivadas reales de la función objetivo dentro de la función de pérdida.

Ejemplo:

if USE_DERIV_LOSS:
    DY  = (6*X**2 + 3*cos(X)).astype(np.float32)   # y' real
    DDY = (12*X - 3*sin(X)).astype(np.float32)     # y'' real

Descripción:
- DY: primera derivada real de la función.
- DDY: segunda derivada real.
Ambas se comparan con las derivadas del modelo simbólico encontrado.

Ventajas:
- Acelera la convergencia hacia funciones con la forma correcta.
- Reduce el sobreajuste numérico.
- Ideal si conoces parte del comportamiento físico (pendientes, curvaturas, etc.).

Recomendaciones:
- Activar si la función o sus derivadas son conocidas.
- Desactivar si los datos provienen de mediciones ruidosas o experimentales.

-----------------------------------------------------------
2. HIPERPARÁMETROS DE EVOLUCIÓN
-----------------------------------------------------------

POP=160; GEN=1000; DEPTH=5; ELITE=10; PMUT=0.6; PXOVER=0.3; ALPHA=1e-3
FINE_TUNE=True; FT_STEPS=250; FT_LR=5e-3

Descripción de parámetros:

POP      → Tamaño de la población (100–300 recomendado)
GEN      → Número de generaciones (200–2000)
DEPTH    → Profundidad máxima del árbol de expresión (3–6)
ELITE    → Número de individuos conservados sin mutación (5–15)
PMUT     → Probabilidad de mutación (0.4–0.7)
PXOVER   → Probabilidad de cruce entre individuos (0.2–0.5)
ALPHA    → Penalización de complejidad (1e-3–1e-2)
FINE_TUNE → Activa ajuste fino de constantes (True/False)
FT_STEPS → Iteraciones del ajuste fino (100–500)
FT_LR    → Tasa de aprendizaje (1e-3–1e-2)

Consejos:
- DEPTH controla la complejidad máxima de la función.
- Aumentar GEN mejora precisión pero incrementa tiempo.
- FINE_TUNE usa PyTorch para ajustar constantes tras la evolución.

-----------------------------------------------------------
3. AOS (ADAPTATIVE OPERATOR SELECTION)
-----------------------------------------------------------

Configuración típica:
AOS_CONF = dict(
    tau=0.8, lr=0.4, decay=0.98,
    prune_every=40, min_keep_un=3, min_keep_bin=3, prune_threshold=0.04
)

Significado de parámetros:
tau            → Peso de memoria del rendimiento de operadores (0.7–0.9)
lr             → Tasa de aprendizaje del AOS (0.2–0.5)
decay          → Atenuación progresiva (0.95–0.99)
prune_every    → Cada cuántas generaciones limpiar operadores (20–50)
min_keep_un    → Mínimo de operadores unarios a mantener (2–3)
min_keep_bin   → Mínimo de operadores binarios a mantener (2–3)
prune_threshold → Umbral de rendimiento mínimo (0.04)

Explicación:
AOS adapta dinámicamente qué operadores son más útiles (sin, log, pow, etc.).
Favorece los que mejoran el ajuste en generaciones recientes.

-----------------------------------------------------------
4. PRIORS (GUÍAS INICIALES DE OPERADORES)
-----------------------------------------------------------

unary_priors  = {"sin":0.6, "cos":0.1, "log":0.1}
binary_priors = {"add":0.3, "mul":0.1, "pow":0.4}

Los priors establecen un sesgo inicial para ciertos operadores.
Ayudan a guiar la búsqueda hacia formas que tienen sentido físico o matemático.

unary_priors  → Funciones unarias (sin, cos, log, sqrt, etc.)
binary_priors → Operaciones binarias (add, mul, pow, div, etc.)

Ejemplo:
Aumentar el peso de "pow" o "sin" puede favorecer funciones polinómicas u oscilatorias.

-----------------------------------------------------------
5. SUBMUESTREO POR GENERACIÓN
-----------------------------------------------------------

sample_schedule = [
    (40, 0.25), (90, 0.6), (130, 0.85), (GEN, 1.0)
]

Controla qué fracción de los datos se usa en cada etapa:

Generación ≤ 40  → 25% de datos
Generación ≤ 90  → 60% de datos
Generación ≤ 130 → 85% de datos
Final (GEN)      → 100% de datos

Ventajas:
- Etapas iniciales más rápidas.
- Gradualmente se aumenta la precisión.
- Reduce costo computacional en primeras generaciones.

-----------------------------------------------------------
6. CURRÍCULUM DE COMPLEJIDAD
-----------------------------------------------------------

curriculum = [
    {"until": 40, "depth": 3, "un": ["sin","cos","abs","neg","sqrt"], "bin": ["add","sub","mul"]},
    {"until": 90, "depth": 4, "un": ["sin","cos","log","sqrt","abs","neg"], "bin": ["add","sub","mul","div"]},
    {"until": GEN, "depth": 5, "un": ["sin","cos","log","sqrt","abs","neg"], "bin": ["add","sub","mul","div","pow"]},
]

Define qué operadores y profundidad se permiten en cada fase evolutiva.

Fase 1: funciones simples (add, mul, sin)
Fase 2: se incorporan log y div
Fase 3: se habilitan pow y profundidad máxima

Permite que el modelo aprenda primero estructuras simples y luego complejas.

-----------------------------------------------------------
7. BLOQUE PRINCIPAL DE EVOLUCIÓN
-----------------------------------------------------------

t0=time.time()
best, fit, hist, aos = evolve(
    X, Y, CAT_UN, CAT_BIN,
    pop_size=POP, generations=GEN, max_depth=DEPTH,
    elite=ELITE, p_mut=PMUT, p_xover=PXOVER, alpha=ALPHA,
    verbose=True, aos_params=AOS_CONF,
    use_macros=True, p_macro=0.25,
    use_hint=True, hint_fraction=0.35, hint_jitter=0.25,
    unary_priors=unary_priors, binary_priors=binary_priors,
    sample_schedule=sample_schedule, curriculum=curriculum,
    hill_climb_steps=3
)

Descripción de parámetros extra:
use_macros      → Permite usar combinaciones predefinidas (sin(log(x)), sqrt(abs(x)))
p_macro         → Probabilidad de usar macros (0.2–0.3)
use_hint        → Usa información auxiliar (derivadas o heurísticas)
hint_fraction   → Proporción inicial de individuos guiados (0.3–0.4)
hint_jitter     → Ruido agregado a los hints (0.2–0.3)
hill_climb_steps → Mini optimizaciones locales por generación (1–5)

El bloque evolve ejecuta todo el ciclo:
1. Genera la población inicial.
2. Evalúa cada individuo (MSE + penalización).
3. Aplica mutaciones y cruces.
4. Ajusta probabilidades de operadores (AOS).
5. Gradualmente incrementa complejidad (currículum).
6. Ajusta constantes (FINE_TUNE).

-----------------------------------------------------------
8. RESULTADOS
-----------------------------------------------------------

best → expresión simbólica final
fit  → error total
hist → evolución del error por generación
aos  → registro de operadores más efectivos

Ejemplo de visualización:

print("Expresión final:", best.to_text())

# Formato simbólico (SymPy)
f_sym, x = to_sympy(best)
display(sp.Eq(sp.Symbol("f(x)"), f_sym))

-----------------------------------------------------------
9. RECOMENDACIONES PRÁCTICAS
-----------------------------------------------------------

- Comenzar con DEPTH=4–5 y GEN=400–800.
- Si el resultado converge mal, subir POP o GEN.
- Si las expresiones se vuelven demasiado largas, subir ALPHA.
- Si la convergencia es lenta, activar derivadas (USE_DERIV_LOSS=True).
- Si esperas estructuras trigonométricas o polinomiales, ajustar priors y curriculum.
- Para máxima velocidad: activar GPU con CuPy o Torch.

-----------------------------------------------------------
10. DESCRIPCIÓN CONCEPTUAL
-----------------------------------------------------------

Este sistema combina:
- Programación genética (búsqueda estructural)
- Optimización simbólica (selección de operadores)
- Aprendizaje jerárquico (currículum de complejidad)
- Información física (derivadas)

El objetivo final es descubrir funciones matemáticas interpretables que describan correctamente los datos,
manteniendo un equilibrio entre precisión y simplicidad.

===========================================================
💡 Inspirado en: Gal-Lahat/Function-Aprox (GitHub), reimplementado y extendido con optimización GPU, derivadas simbólicas y AOS adaptativo.
===========================================================


# === Function-Aprox: CPU Requirements ===
# Compatible con Python >= 3.10
# Esta versión no requiere GPU (usa NumPy como backend)

numpy>=2.0
sympy>=1.12
matplotlib>=3.8
tqdm>=4.65
nbformat>=5.9
scipy>=1.11

pip install -r requirements.txt


# === Function-Aprox: GPU Requirements (CUDA 13.x / RTX 40xx) ===
# Compatible con Python >= 3.10 y NVIDIA RTX 4060 Ti / CUDA 13.0
# Usa PyTorch para fine-tuning y CuPy para evaluación simbólica masiva.

# Núcleo
numpy>=2.0
sympy>=1.12
matplotlib>=3.8
tqdm>=4.65
nbformat>=5.9
scipy>=1.11

# GPU acceleration
cupy-cuda13x==13.6.0   # CuPy para CUDA 13.x
torch==2.9.0           # PyTorch con soporte CUDA/cu12.x

pip install -r requirements-gpu.txt



######Conda yaml#####

name: function-aprox
channels:
  - conda-forge
dependencies:
  # --- Python base ---
  - python=3.10
  - pip

  # --- Núcleo científico ---
  - numpy>=2.0
  - sympy>=1.12
  - scipy>=1.11
  - matplotlib>=3.8
  - tqdm>=4.65

  # --- Notebooks ---
  - jupyter
  - ipykernel
  - nbformat>=5.9

  # --- Utilidades varias ---
  - packaging
  - typing_extensions

  # --- Instalar por pip (GPU/cuDNN y PyTorch/CuPy) ---
  - pip:
      - cupy-cuda13x==13.6.0   # CuPy para CUDA 13.x (driver 580.95.05)
      - torch==2.9.0           # PyTorch con runtime CUDA propio (cu12.x)


conda env create -f environment.yml
conda activate function-aprox



