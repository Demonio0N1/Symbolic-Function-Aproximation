# Límites del motor `symreg` — informe empírico

Batería de 64 corridas en CPU (4 núcleos, sin GPU) con **la misma configuración que
`symreg.benchmarks` en modo `catalogo`**: 4 islas × 160 individuos × 150
generaciones, curriculum de 3 fases, fine-tuning Adam del Hall of Fame (250
pasos) y reconocimiento de constantes. Salvo indicación, catálogo por defecto
(`sin cos log sqrt abs neg` / `add sub mul div pow`), `alpha = 1e-3`, semilla 0.
Script y resultados crudos: ver la sección *Reproducir* al final.

**NMSE** = MSE / var(Y) (error relativo a la varianza del objetivo; 1 = predecir
la media, 0 = perfecto). Estados: ✅ exacta (NMSE < 1e-9), 🟢 muy buena (< 1e-3),
🟡 aceptable (< 1e-2), 🟠 pobre (< 0.2), 🔴 falla.

## Resumen: qué recupera y qué no

| Funciona bien (exacto en 1–2 min) | Falla o degrada con la config por defecto |
|---|---|
| Polinomios de grado ≤ 3 con coeficientes enteros/simples (`2x³+3sin x+1`, `x²+x-1`) | Polinomios de grado ≥ 4 con varios términos (`x⁴-3x²+2`: NMSE 0,004–0,03) |
| Composición simple de operadores del catálogo (`sin(x²)`, `sin x·cos 2x`, `sin x + sin(x+x²)`) | Funciones que necesitan operadores fuera del catálogo: `exp(x)`, `x·e^(-x²)`, `tanh`, `floor`, `sign` |
| `1/x`, incluso con datos a ±0,1 del polo | Objetivos con **var(Y) ≲ 1** (nguyen5/7, keijzer1, feyn_gauss, feyn_rel): la penalización `alpha` absoluta gana a la exactitud |
| Multivariable separable, hasta 5 variables (`x·x1+sin x`, `x²+x1²`, `x·x1·x2`, `x·x1+x2-x3²+sin x4`) | Cocientes con denominador compuesto (`(x+x1)/(1+x2²)`) |
| Frecuencias hasta ~25 muestras por periodo (`sin 10x` con 400 puntos en [-5,5]) | Frecuencias con ≲ 5 muestras por periodo (`sin 50x`, `sin 200x` → devuelve 0) |
| Pocos datos: 15 puntos bastan para `2x³+3sin x+1` | Constantes "feas" dentro de funciones (`1,7·sin(2,3x+0,4)+0,37x` → NMSE 0,06, 114 nodos) |
| Ruido del 1 % (NMSE 8e-5, estructura reconocible) | Ruido ≥ 10 %: árboles de 390 nodos, 9 min, expresión ilegible; 50 %: > 25 min |
| Extrapolación cuando la forma es exacta | Extrapolación de ajustes aproximados (nguyen7: NMSE 1,3 fuera del rango) |
|  | Escalas: `|Y| ~ 1e-4` → devuelve 0; `Y ~ 1e4` y `x ∈ [1000, 2000]` → no converge o NMSE 0,02 |

## Los 10 límites, con causa y remedio

### 1. La penalización de complejidad es absoluta y domina cuando var(Y) es pequeña  ⟵ el más importante

`fitness = MSE + alpha·n_nodos` con `alpha = 1e-3` fijo. Si `var(Y) ≈ 0,03–0,9`,
cada nodo cuesta tanto como un 0,1–3 % de toda la señal, y **la solución exacta tiene
peor fitness que una aproximación de 3–5 nodos**. Fitness de la expresión verdadera
vs la hallada (datos completos, alpha = 1e-3):

| objetivo | var(Y) | nodos exacta | fit exacta | fit hallada | gana |
|---|---:|---:|---:|---:|---|
| propia | 8,8e3 | 14 | 0,014 | 0,015 | exacta |
| nguyen5 | 0,027 | 9 | **0,009** | **0,0055** | la simple (`-cos(√|x|)`) |
| nguyen7 | 0,67 | 11 | **0,011** | **0,0043** | la simple (`1,38·x`) |
| feyn_gauss | 0,019 | 11 | **0,011** | **0,0056** | la simple |
| keijzer1 | 0,012 | 8 | 0,008 | 0,0113 | exacta, por 0,003 |
| feyn_rel | 0,093 | 7 | 0,007 | 0,0091 | exacta, por 0,002 |

Esto explica **exactamente** la tabla de `BENCHMARKS.md`: los MSE "0/5" de GPU
(nguyen5 0,00151, nguyen7 0,00131, keijzer1 0,0063, feyn_gauss 0,000629, feyn_rel
0,00409) se reproducen al dígito en CPU porque el motor converge, de forma
determinista, al óptimo de su propio objetivo. No es un fallo de búsqueda.

Con `alpha` relativo a la varianza mejora mucho, a costa de árboles más grandes:

| objetivo | alpha = 1e-3 (default) | alpha = 1e-3·var(Y) | alpha = 1e-6·var(Y) |
|---|---:|---:|---:|
| nguyen5 | 0,056 | 0,0022 | 5e-5 (36 nodos) |
| nguyen7 | 0,002 | 0,002 | 2e-6 (48 nodos) |
| keijzer1 | 0,52 | 8,6e-4 | > 60 min, sin terminar |
| feyn_gauss | 0,032 | 5,9e-4 | 4,3e-5 (77 nodos) |
| feyn_rel | 0,044 | 3,4e-5 | 2,2e-4 (158 nodos) |

**Remedio:** pasar `alpha = k·var(Y)` con `k ≈ 1e-5…1e-4` (parámetro ya existente de
`evolve`/`evolve_islands`), o normalizar `Y` a varianza 1 antes y deshacer la
escala en la expresión. Sin presión de parsimonia (`1e-6·var`) aparece el bloat
del punto 6.

### 2. Solo recupera lo que el catálogo puede expresar

- `exp(x)`: no existe operador `exp` (solo se llega vía `pow(e, x)` o `eml`). Con el
  catálogo por defecto → 27 nodos y NMSE 2,7e-3; añadiendo `eml` al catálogo →
  **`exp(x)` exacto en 57 s**. `x·e^(-x²)` falla con las dos semillas (NMSE 0,04).
- `tanh(3x)`: sin `tanh` → `sin(0,61x + sin(x+sin x))`, NMSE 2,8e-4; con `tanh` →
  exacto.
- `sign(x)` → `sin(…·sin x/|x|)` (NMSE 1,5e-4, numéricamente buena pero no es la
  función); `floor(x)` → NMSE 0,016. No hay operadores de escalón ni redondeo.

**Remedio:** añadir `exp` al catálogo unario (hoy falta en `UNARY_MAP`), o activar
`eml`/`tanh`/`sigmoid` cuando el problema lo sugiera.

### 3. Polinomios de grado ≥ 4

`x²+x-1` exacto; `x⁴-3x²+2` → NMSE 0,028 (s0) / 0,004 (s1) con 47–54 nodos;
`x⁶-2x⁴+x²-1` → 0,027 / 0,007; `x⁸-x²` → 0,006. El curriculum de las primeras 40
generaciones solo ofrece `sin cos abs neg sqrt` + `add sub mul` (sin `pow`, que
entra en la gen 90), y el AOS se queda en composiciones trigonométricas.
Además `pow` recorta el exponente a [-5, 5], así que `x⁶` solo existe como cadena
de `mul`.

**Remedio:** curriculum con una fase solo-polinómica (`add sub mul pow`), activar
`chebyshevT`/`legendreP` (ya existen), o más generaciones.

### 4. Dependencia de la semilla: una corrida no es concluyente

`sin x·cos 2x`: semilla 0 → `-sin⁵x` (NMSE 0,16); semilla 1 → **exacto**.
`x⁴-3x²+2`: 0,028 vs 0,004. `sin 10x` exacto con alpha=1e-3 pero NMSE 0,57 con
alpha=5e-4. Tasa de recuperación por semilla es la métrica correcta (como hace
`BENCHMARKS.md` con 5 semillas); más islas (`evolve_islands_mpi`) o reinicios
compensan.

### 5. Frecuencia máxima ≈ 10–20 muestras por periodo

Con 400 puntos en [-5, 5] (paso 0,025): `sin 10x` (25 muestras/periodo) exacto;
`sin 50x` (5 muestras/periodo) y `sin 200x` → la constante 0. El límite no es
Nyquist sino la búsqueda: no hay gradiente útil hacia frecuencias altas.

### 6. Ruido y ausencia de solución exacta disparan el bloat y el tiempo

`2x³+3sin x+1` + ruido gaussiano (fracción de σ_Y): 0 % → 15 nodos, 91 s;
1 % → 51 nodos, 267 s, NMSE 8e-5 (la estructura sigue visible:
`x²(1,9999x+0,40)+2,93 sin x+…`); 10 % → **390 nodos, 554 s**, NMSE 0,008 (al
nivel del ruido, pero expresión ilegible); 50 % → **no terminó en 25 min**. Causa:
`alpha=1e-3` es despreciable frente a `var(Y)=8840`, así que no hay presión de
parsimonia y el tiempo por generación crece con el tamaño de los árboles.

**Remedio:** `val_fraction > 0` + `selection="nsga2"` (frente de Pareto) y alpha
relativo; o limitar `max_depth`.

### 7. Constantes lejos de O(1)

Las constantes nacen en U(-5, 5) y Adam (lr 5e-3, 250 pasos) solo las mueve un
poco: `(x-100)²` en [98, 102] → NMSE 0,0019 y 28 nodos (nunca aparece el 100);
`x²/1e6` en [1000, 2000] → NMSE 0,018 (además `pow` recorta la base a ±1e3);
`1e4·x²+1e3` → no terminó en 25 min; `1e-4·sin x` → devuelve **0** porque un solo
nodo (1e-3) cuesta más que toda la señal (var = 5e-9). Constantes "feas" dentro de
una función (`1,7·sin(2,3x+0,4)+0,37x`) → falla (NMSE 0,064, 114 nodos, 205 s);
fuera de ella (`0,123x²-4,567x+0,89`) → NMSE 4,6e-6 pero con estructura incorrecta.

**Remedio:** centrar y escalar `x` e `y` antes de evolucionar (z-score) y
deshacerlo en SymPy; es la práctica estándar y aquí no está automatizada.

### 8. Cocientes compuestos en multivariable

Lo separable sale exacto hasta 5 variables (y los dos CSV de ejemplo: `x·x1+sin x`
exacto; `tres_variables` → `x²+x1²+…` con NMSE 0,0088). `(x+x1)/(1+x2²)` →
`(x+x1)·|cos √|x2||`, NMSE 0,0039: el denominador `1+x2²` no aparece en 150
generaciones.

### 9. Extrapolación solo si la forma es exacta

Entrenando `2x³+3sin x+1` en [-2, 2]: NMSE 5,7e-5 dentro, pero la expresión es
`1,85x³+2,23x+0,5 sin 2x+1`, no la verdadera; fuera ([-5, 5]) NMSE 3e-4 (se salva
porque el cúbico domina). nguyen7 entrenado en [0, 2]: la recta `1,38x` da NMSE
**1,3** en [0, 8]. Un NMSE de 1e-3…1e-2 dentro del rango no garantiza nada fuera.

### 10. La expresión exportada puede no coincidir con el motor

2 de 61 expresiones fallan `check_sympy_roundtrip`: el motor explota las guardas
numéricas (`a/(x-x)` → valor enorme recortado, `log(|0|+1e-8)`, `sqrt(|·|+1e-8)`),
que en SymPy son `zoo`/0 (en `const_fea1` SymPy simplifica todo a `0`). Ocurre en
árboles inflados (> 50 nodos). **Siempre** validar con `check_sympy_roundtrip`
antes de usar la expresión fuera del motor.

### Otros detalles medidos

- **Tiempo (CPU, 4 núcleos):** mediana 90 s por corrida; 200–550 s si los árboles
  superan ~100 nodos; 3 corridas superaron 25 min. Depende del tamaño de los
  árboles, no del nº de datos (15 puntos tardaron 259 s).
- **Criterio "recuperada = MSE < 1e-6" de `BENCHMARKS.md` es absoluto:** da por
  recuperada la constante 0 para `1e-4·sin x` y una expresión de 77 nodos para
  feyn_gauss. Mejor NMSE < 1e-9 y nº de nodos ≤ 2× el objetivo.
- **Guardas de `pow`:** base recortada a ±1e3 y exponente a ±5; `log` y `sqrt` usan
  `|·| + 1e-8`; `div` sustituye NaN/Inf por el máximo finito/4. Son límites
  numéricos duros del motor.
- **Tests:** los 29 de `pytest tests/` pasan en CPU (1 omitido por falta de GPU).
- **Datos:** los tres CSV de `ejemplos/` cargan y corren sin cambios.

## Tablas completas (64 corridas)

### Benchmarks oficiales del repo (config 'catalogo', semilla 0, CPU)

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| propia |  | 1 | 2.8e-15 | ✅ exacta | 15 | 91.4 | `(((x + x) * (x * x)) - (cos(-3.1429) + (sin(x) * -3)))` |
| nguyen5 |  | 1 | 0.056 | 🟠 pobre | 4 | 42.9 | `-(cos(sqrt(\|x)))` |
| nguyen6 |  | 1 | 1.9e-15 | ✅ exacta | 9 | 52.5 | `(sin(x) + sin(((x * x) + x)))` |
| nguyen7 |  | 1 | 0.002 | 🟡 aceptable | 3 | 47.2 | `(1.3766 * x)` |
| keijzer1 |  | 1 | 0.52 | 🔴 falla | 5 | 42.1 | `log(\|sqrt(\|cos(sin(x)))\|+eps)` |
| feyn_gauss |  | 1 | 0.032 | 🟠 pobre | 5 | 52.7 | `(cos(sqrt(\|x)) * 0.43403)` |
| feyn_edens |  | 1 | 0 | ✅ exacta | 6 | 61.2 | `((x * x) * sqrt(\|-0.25))` |
| feyn_rel |  | 1 | 0.044 | 🟠 pobre | 5 | 55.5 | `(sqrt(\|cos(x)) ^ -2.941)` |

### Mismos benchmarks con alpha = 1e-3·var(Y)

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| nguyen5_alpharel | 200 | 1 | 0.0022 | 🟡 aceptable | 12 | 60.5 | `-cos(sin(x + sin(sin(x)))*Abs(x)**(1/4))` |
| nguyen7_alpharel | 200 | 1 | 0.002 | 🟡 aceptable | 3 | 50.5 | `1.37664532661438*x` |
| keijzer1_alpharel | 200 | 1 | 0.00086 | 🟢 muy buena | 37 | 101.7 | `0.29916880399624*cos(2.06320455828219*sqrt(Abs(x)) + 5.01475524902344*Abs(x) + 3` |
| feyn_gauss_alpharel | 300 | 1 | 0.00059 | 🟢 muy buena | 28 | 89.6 | `sin(cos(sqrt(Abs(log(Abs(0.86950296163559*x**2 + sin(2*Abs(x) + 3.87075328826904` |
| feyn_rel_alpharel | 200 | 1 | 3.4e-05 | 🟢 muy buena | 17 | 75.7 | `1.73078856995853**(x**2/cos(log(Abs(log(Abs(log(3.80478310585022/Abs(x))))))))` |

### Mismos benchmarks con alpha = 1e-6·var(Y)

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| nguyen5_alpha1e6 | 200 | 1 | 5e-05 | 🟢 muy buena | 36 | 116.9 | `-cos(sqrt(Abs(cos(4.1635594367981*sin(1.00592184066772*x**2 + cos(sqrt(Abs(sin(x` |
| nguyen7_alpha1e6 | 200 | 1 | 2e-06 | 🟢 muy buena | 48 | 121.2 | `0.952101588249207*x*sqrt(Abs(0.352348566055298*sin(x + Abs(sqrt(x)*(sin(1.735286` |
| keijzer1_alpha1e6 | 200 | 1 | – | ⏱ > 25 min sin terminar (bloat sin presión de parsimonia) | – | >1500 | – |
| feyn_gauss_alpha1e6 | 300 | 1 | 4.3e-05 | ⚠️ 'recuperada' por MSE<1e-6 pero NMSE alto | 77 | 145.1 | `Abs(0.700495380013329*cos(x) + 0.350247690006665*sqrt(Abs(cos(0.384196999450219*` |
| feyn_rel_alpha1e6 | 200 | 1 | 0.00022 | 🟢 muy buena | 158 | 217.4 | `(1.91869190523808*2**(1/4)*Abs(x**(1/4)*sin(0.753501718575538*Abs(x**(1/4)*(x + ` |

### Grado polinómico

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| poly2 | 300 | 1 | 7.7e-15 | ✅ exacta | 7 | 78.9 | `x**2 + x - 1` |
| poly4 | 300 | 1 | 0.028 | 🟠 pobre | 54 | 117.1 | `2.79334743356923**sin(Abs(1.13264441490173*x**2 - 1.20421399734914) - 1.23408690` |
| poly4@s1 | 300 | 1 | 0.0041 | 🟡 aceptable | 47 | 101.4 | `-sin(0.914834260940552*x**2)**9 + sqrt(Abs(2.43184208869934**Abs(x**2*cos(3.0595` |
| poly6 | 300 | 1 | 0.027 | 🟠 pobre | 19 | 72.6 | `-(Abs(log(Abs(cos(x)))) + 0.78551928670431)*cos(0.692390390403775*x**3)` |
| poly6@s1 | 300 | 1 | 0.0071 | 🟡 aceptable | 28 | 74.0 | `log((2.03866584983062*sqrt(Abs(cos(x))))**(cos(x**2) + cos(x**2 + cos(sqrt(Abs(x` |
| poly8 | 300 | 1 | 0.006 | 🟡 aceptable | 28 | 90.1 | `x**3*(2.35474634170532*x - 2.37534129032824e-8)*cos(4.6093373298645*x)*Abs(x - s` |

### Composición y operadores fuera del catálogo

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| sin_x2 | 400 | 1 | 6.4e-14 | ✅ exacta | 4 | 60.7 | `sin(x**2)` |
| sinx_cos2x | 400 | 1 | 0.16 | 🟠 pobre | 5 | 59.2 | `-sin(x)**5` |
| sinx_cos2x@s1 | 400 | 1 | 1.7e-14 | ✅ exacta | 7 | 69.1 | `sin(x)*cos(2*x)` |
| x_gauss | 400 | 1 | 0.046 | 🟠 pobre | 6 | 52.3 | `x*cos(x)**1.44444444444444` |
| x_gauss@s1 | 400 | 1 | 0.042 | 🟠 pobre | 7 | 56.4 | `0.384299964605552*sin(x + sin(x))` |
| exp_x | 300 | 1 | 0.0027 | 🟡 aceptable | 27 | 97.0 | `log(Abs((x + 4.8988995552063)**x + 1))/cos(sin(log(0.787205994129181*x**2 + 3.61` |
| exp_x_eml | 300 | 1 | 3.8e-15 | ✅ exacta | 3 | 56.8 | `exp(x)` |
| tanh3x | 300 | 1 | 0.00028 | 🟢 muy buena | 10 | 68.8 | `sin(0.608784556388855*x + sin(x + sin(x)))` |
| tanh3x_cat | 300 | 1 | 3.3e-16 | ✅ exacta | 4 | 81.8 | `tanh(3*x)` |

### Constantes no 'bonitas'

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| const_fea1 | 400 | 1 | 0.064 | 🟠 pobre | 114 | 204.6 | `0` |
| const_fea2 | 400 | 1 | 4.6e-06 | 🟢 muy buena | 27 | 130.9 | `-4.56699991226196*x + sqrt(0.950904409262954*x**2 + 4.73129255041225) - cos(cos(` |

### Singularidades, discontinuidades y frecuencia

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| recip_pos | 300 | 1 | 3.5e-15 | ✅ exacta | 3 | 95.2 | `1/x` |
| recip_pos_alpharel | 300 | 1 | 3.8e-15 | ✅ exacta | 3 | 62.8 | `1/x` |
| recip_sym | 286 | 1 | 1.5e-15 | ✅ exacta | 3 | 90.9 | `1/x` |
| sign | 300 | 1 | 0.00015 | 🟢 muy buena | 11 | 97.8 | `sin((0.740443869855444*Abs(x) + 1.3297563791275)*sin(x)/Abs(x))` |
| floor | 300 | 1 | 0.016 | 🟠 pobre | 12 | 73.9 | `x - 0.681253790855408*cos(cos(3.09603643417358*x - 7.0696496963501))` |
| sin10x | 400 | 1 | 1.2e-12 | ✅ exacta | 7 | 60.0 | `sin(9.99999962793288*x)` |
| sin10x_alpharel | 400 | 1 | 0.57 | 🔴 falla | 82 | 85.0 | `-sin(sin(297*cos(19*x*Abs(x*sqrt(log(0.238490271585506/Abs(sin(1.33088401811309/` |
| sin50x | 400 | 1 | 1 | 🔴 falla | 1 | 48.9 | `0` |
| sin200x | 400 | 1 | 1 | 🔴 falla | 1 | 64.5 | `0` |

### Ruido (fracción de la desviación típica de Y)

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| ruido_1pct | 400 | 1 | 8.3e-05 | 🟢 muy buena | 51 | 266.6 | `x**2*(1.9998539686203*x + 0.400960491380283) + 2.93329500406981*sin(x) + cos(cos` |
| ruido_10pct | 400 | 1 | 0.008 | 🟡 aceptable | 390 | 554.2 | `x*(24*x**2 + 17*(-sin(Abs(x - 3.03155689088838)))**(sqrt(Abs(x + 1))))/12` |
| ruido_50pct | 400 | 1 | – | ⏱ > 25 min sin terminar (bloat por ruido) | – | >1500 | – |

### Pocos datos

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| n15 | 15 | 1 | 8.2e-16 | ✅ exacta | 14 | 258.8 | `2*x**3 + 3*sin(x) + 1` |
| n50 | 50 | 1 | 1.2e-14 | ✅ exacta | 21 | 139.2 | `2*x**3 + 3*sin(x) + 1` |

### Escala y rango de x

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| escala_grande | 300 | 1 | – | ⏱ > 25 min sin terminar (Y ~ 1e4, alpha irrelevante → bloat) | – | >1500 | – |
| escala_chica | 300 | 1 | 1 | ⚠️ 'recuperada' por MSE<1e-6 pero NMSE alto | 1 | 28.8 | `0` |
| offset100 | 300 | 1 | 0.0019 | 🟡 aceptable | 28 | 123.4 | `0.289242338994669**cos(1.32288864254951*x - 0.0535335838794708) + sin(x - 8/5) +` |
| x_grande | 300 | 1 | 0.018 | 🟠 pobre | 21 | 114.2 | `log(Abs(-115*log(Abs(x)) + 5*sqrt(115)*sqrt(Abs(x)) + 92)/115) + sin(3.007597544` |

### Multivariable

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| mv_csv2 | 400 | 2 | 1.6e-13 | ✅ exacta | 6 | 97.7 | `x*x1 + sin(x)` |
| mv_csv3 | 500 | 3 | 0.0088 | 🟡 aceptable | 56 | 221.3 | `0.999781310558319*x**2 + 0.997486233711243*x1**2 + x2*log(Abs(x2**3*sqrt(Abs(x2)` |
| mv_ruido_csv | 300 | 1 | 0.0035 | 🟡 aceptable | 16 | 101.4 | `cos(cos(log(Abs(x)))) + 3*Abs(x) - sqrt(Abs(sin(x)))` |
| mv_x1x2_sin | 300 | 2 | 2.5e-15 | ✅ exacta | 6 | 95.3 | `x*x1 + sin(x)` |
| mv_sum_sq | 300 | 2 | 7.8e-15 | ✅ exacta | 7 | 107.6 | `x**2 + x1**2` |
| mv_sincos | 400 | 2 | 5.1e-15 | ✅ exacta | 5 | 94.0 | `sin(x)*cos(x1)` |
| mv_x1x2x3 | 400 | 3 | 3.1e-15 | ✅ exacta | 5 | 127.0 | `x*x1*x2` |
| mv_ratio | 400 | 3 | 0.0039 | 🟡 aceptable | 12 | 99.8 | `(x + x1)*Abs(cos(sqrt(Abs(x2))))` |
| mv_5vars | 500 | 5 | 3.5e-15 | ✅ exacta | 12 | 133.6 | `x*x1 + x2 - x3**2 + sin(x4)` |

### Extrapolación (entrena en rango corto, evalúa fuera)

| Caso | n | vars | NMSE | Estado | Nodos | t (s) | Expresión obtenida (SymPy) |
|---|---:|---:|---:|---|---:|---:|---|
| extrap_propia | 300 | 1 | 5.7e-05 | 🟢 muy buena · extrap NMSE=0.0003 | 24 | 133.1 | `1.85105238820697*x**3 + 2.22886660317308*x + 0.5*sin(2*x) + 0.99999988079071` |
| extrap_nguyen7 | 200 | 1 | 0.002 | 🟡 aceptable · extrap NMSE=1.3 | 3 | 59.0 | `1.37664532661438*x` |


## Reproducir

Las corridas usan `symreg.benchmarks.correr_benchmark(..., "catalogo", seed=0)` para
los 8 presets y, para el resto, una llamada idéntica a `evolve_islands` +
`fine_tune_hof` + `recognize_constants` con los mismos hiperparámetros (ver el
cuerpo de `correr_benchmark`), cambiando solo `X`, `Y`, el catálogo o `alpha`.
Entorno: Python 3.11, torch 2.14 CPU, numpy 2.4, sympy 1.14; `CUDA_VISIBLE_DEVICES=""`,
`OMP_NUM_THREADS=1`, 4 corridas en paralelo. Las corridas marcadas "> 25 min" se
cortaron con `timeout 1500`.
