# Benchmarks (Fase 5)

GPU: RTX 4090 — 5 semillas por celda; recuperación exacta = MSE < 1e-6 tras fine-tuning + reconocimiento de constantes.

| Objetivo | Config | Recuperación | MSE mediana | Nodos med. | t medio (s) |
|----------|--------|:------------:|------------:|-----------:|------------:|
| propia | catalogo | 4/5 | 2.85e-11 | 20 | 83 |
| propia | catalogo+eml | 2/5 | 0.00177 | 23 | 90 |
| propia | eml_puro | 0/5 | 1.59e+03 | 191 | 31 |
| nguyen5 | catalogo | 0/5 | 0.00151 | 4 | 33 |
| nguyen5 | catalogo+eml | 0/5 | 0.00151 | 4 | 34 |
| nguyen5 | eml_puro | 0/5 | 0.00291 | 95 | 6 |
| nguyen6 | catalogo | 5/5 | 1.87e-15 | 9 | 42 |
| nguyen6 | catalogo+eml | 2/5 | 0.00114 | 10 | 43 |
| nguyen6 | eml_puro | 0/5 | 0.0204 | 11 | 6 |
| nguyen7 | catalogo | 0/5 | 0.00131 | 3 | 42 |
| nguyen7 | catalogo+eml | 0/5 | 0.00131 | 3 | 41 |
| nguyen7 | eml_puro | 0/5 | 0.00799 | 95 | 5 |
| keijzer1 | catalogo | 0/5 | 0.0063 | 5 | 27 |
| keijzer1 | catalogo+eml | 0/5 | 0.00641 | 5 | 28 |
| keijzer1 | eml_puro | 0/5 | 0.00567 | 95 | 4 |
| feyn_gauss | catalogo | 0/5 | 0.000629 | 5 | 37 |
| feyn_gauss | catalogo+eml | 0/5 | 0.000629 | 6 | 38 |
| feyn_gauss | eml_puro | 0/5 | 0.562 | 95 | 5 |
| feyn_edens | catalogo | 4/5 | 1.23e-14 | 6 | 44 |
| feyn_edens | catalogo+eml | 5/5 | 0 | 5 | 44 |
| feyn_edens | eml_puro | 0/5 | 0.0596 | 95 | 6 |
| feyn_rel | catalogo | 0/5 | 0.00409 | 7 | 42 |
| feyn_rel | catalogo+eml | 0/5 | 0.00409 | 5 | 44 |
| feyn_rel | eml_puro | 0/5 | 0.0381 | 95 | 7 |
