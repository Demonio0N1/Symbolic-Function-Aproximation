# Interfaz web de symreg — guía de uso

Interfaz local para interactuar con el motor de regresión simbólica desde el
navegador. Corre **en tu máquina** (los cálculos usan tu GPU/CPU a través del
paquete `symreg`); nada sale de tu equipo.

## Arranque

```bash
./setup.sh          # una sola vez: instala y verifica todo
./webapp/run.sh     # arranca el servidor -> http://localhost:8000
```

Detén el servidor con `Ctrl+C`. Si cambias el código del backend
(`webapp/app.py`), reinícialo; si solo cambia la página, basta recargar el
navegador (`Ctrl+Shift+R` para saltarte la caché).

## 1 · Datos (¿qué función quieres recuperar?)

Tres pestañas:

| Pestaña | Uso |
|---|---|
| **Preset** | Los 8 objetivos de los benchmarks (propia, Nguyen 5/6/7, Keijzer-1 y 3 de Feynman). |
| **Expresión** | Escribe la función objetivo a mano. Hasta **6 variables** (`x, y, z, u, v, w`, o `x1..x5`), potencias con `^` o `**`, funciones numpy (`sin, cos, exp, log, sqrt, abs, pi, e`…). Acepta forma de ecuación: `x^2 + y^2 = 0`, `f(x,y) = x*y`, o `izq = der` (se usa `izq − der`). Con varias variables el dominio se muestrea uniforme en `[x mín, x máx]` por dimensión. |
| **CSV** | Tus propios datos, sin cabecera, columnas `x1[,x2,…],y` (la última columna es la salida). Multivariable automático. Hay tres archivos de prueba en [`ejemplos/`](../ejemplos): `univariable_ruido.csv` (0.5x²−2cos x+3 con ruido — prueba la validación), `dos_variables.csv` (x·y+sin x) y `tres_variables.csv` (x²+y²+z³ — actívale el fit inicial). |

Extras: `puntos` (tamaño de la muestra), `ruido gaussiano σ` (para probar
robustez) y `semilla` (reproducibilidad).

## 2 · Evolución (cómo buscar)

Pulsa **❓ Guía de parámetros** (cabecera) para ver la explicación gráfica de
cada control con diagramas e intuición. Resumen mínimo:

- **población/isla, generaciones, islas** — presupuesto de búsqueda. Con 4
  islas y 250 generaciones el caso de prueba se recupera 5/5 semillas en ~2 min
  (RTX 4090).
- **profundidad** — complejidad máxima de los sub-árboles nuevos (3–6).
- **alpha** — penalización por nodo: súbelo si salen fórmulas gigantes.
- **validación** — fracción de puntos reservada; el ganador se elige por su
  error ahí (anti-sobreajuste).
- **reinicio (gens)** — regenera la población (conservando el elite) tras N
  generaciones sin mejora.
- **selección** — `alpha·nodos` clásica o `NSGA-II` (devuelve además el frente
  de Pareto precisión-vs-complejidad completo).
- **operadores** — el vocabulario de las fórmulas; el AOS aprende cuáles usar.
  Enlaces rápidos: `[solo eml]` / `[por defecto]`.
- **modo EML** — `operador` añade `eml(a,b)=eᵃ−ln(b)` al catálogo; `puro`
  ajusta por gradiente una fórmula hecha SOLO de eml (aparecen los campos de
  reinicios en paralelo y pasos de Adam).
- **fit inicial (semilla polinomial)** — ajusta primero un polinomio por
  mínimos cuadrados (grado 2–5) y siembra la población con él: la evolución
  parte de algo razonable (en el caso de prueba: 30 s vs 100 s).
- **fine-tuning / reconocer constantes** — pulido final: Adam sobre todas las
  constantes del Hall of Fame y redondeo a enteros/π/e/fracciones si el error
  no empeora.

## 3 · Botones

- **▶ Iniciar** (verde) — corre con la configuración actual.
- **▶ Solo EML** (morado) — misma configuración pero con catálogo
  **únicamente-eml**: el resultado es un árbol 100 % `eml(·,·)` (funciona
  también multivariable). Ignora los checkboxes de operadores.
- **■ Cancelar** — la evolución va por bloques de generaciones, así que la
  cancelación tarda como mucho un bloque y conserva el resultado parcial.

Solo puede haber **una corrida a la vez** (la GPU es una). La barra muestra
generación, mejor fitness y tiempo en vivo.

## 4 · Resultados

- **Expresión** en tres formatos: motor (`(x*x)*(x+x)…`), SymPy y LaTeX
  copiable. Con varias variables se muestra con tus nombres (`y`, `z`) y la
  leyenda del mapeo interno (`x1 = y · x2 = z`).
- **MSE** de entrenamiento y de validación, nº de nodos, tiempo, y el test de
  ida y vuelta motor↔SymPy.
- **Gráficas**: ajuste (curva sobre los datos, o `y real vs y predicho` si es
  multivariable) y convergencia (mejor fitness por bloque, escala log).
- **Hall of Fame**: las k mejores expresiones de toda la corrida.
- **Frente de Pareto** (solo con NSGA-II): todas las fórmulas no dominadas
  precisión-vs-complejidad.
- **Descargas**: `⬇ árbol JSON` (recargable con `symreg.load_tree`) y
  `⬇ checkpoint` (población final + Hall of Fame + scores del AOS; sirve para
  continuar la corrida con `symreg.evolve(init_pop=…)`).

## API REST (para scripts)

La página usa una API que también puedes llamar directamente:

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/info` | dispositivo, versión y presets |
| POST | `/api/iniciar` | inicia una corrida (JSON con los mismos campos del formulario) |
| GET | `/api/estado` | `{corriendo, progreso, resultado, error}` — sondear cada 1–2 s |
| POST | `/api/cancelar` | cancela la corrida en curso |
| POST | `/api/csv` | sube un CSV (multipart, campo `archivo`) |
| GET | `/api/descargar/arbol` \| `/checkpoint` | descargas del último resultado |

Ejemplo mínimo:

```bash
curl -X POST http://localhost:8000/api/iniciar -H "Content-Type: application/json" \
  -d '{"fuente":"expr","expr":"x^2 + y^2","lo":-2,"hi":2,"npts":300,
       "pop":160,"gen":250,"islas":4,"warm_start":true}'
watch -n 2 'curl -s http://localhost:8000/api/estado | python3 -m json.tool | head -20'
```

## Problemas frecuentes

- **La página no carga** → ¿está el servidor arrancado? `./webapp/run.sh`.
  ¿Puerto ocupado? `./webapp/run.sh --port 8001`.
- **Cambié algo y no se refleja** → recarga con `Ctrl+Shift+R`; si tocaste
  `webapp/app.py`, reinicia el servidor.
- **"ya hay una corrida en marcha"** → espera o pulsa Cancelar; una a la vez.
- **Sin GPU** → todo funciona en CPU (más lento); instala con `./setup.sh --cpu`.
