# Suite de benchmarks (Fase 5.1) — wrapper de línea de comandos sobre
# symreg.benchmarks. Corre en GPU.
#
# Uso:
#   python benchmarks.py                        # suite completa (5 semillas)
#   python benchmarks.py --seeds 0,1 --targets propia,nguyen5 --configs catalogo
import argparse
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent


def main():
    import matplotlib
    matplotlib.use("Agg")
    import torch

    from symreg.benchmarks import CONFIGS, TARGETS, correr_benchmark, tabla_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--gen", type=int, default=150)
    ap.add_argument("--out", default="benchmarks_results.json")
    args = ap.parse_args()

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (fallback)")
    seeds = [int(s) for s in args.seeds.split(",")]
    resultados = []
    for tg in args.targets.split(","):
        for cf in args.configs.split(","):
            for sd in seeds:
                r = correr_benchmark(tg, cf, sd, gen=args.gen)
                resultados.append(r)
                print(f"[{tg:10s}|{cf:12s}|s{sd}] mse={r['mse']:.3g} rec={r['recuperada']} "
                      f"nodos={r['nodos']} t={r['t']}s", flush=True)
                json.dump(resultados, open(args.out, "w"), indent=1)

    tabla = tabla_markdown(resultados)
    print("\n" + tabla)
    (REPO / "BENCHMARKS.md").write_text(
        "# Benchmarks (Fase 5)\n\nGPU: RTX 4090 — 5 semillas por celda; "
        "recuperación exacta = MSE < 1e-6 tras fine-tuning + reconocimiento de constantes.\n\n"
        + tabla + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
