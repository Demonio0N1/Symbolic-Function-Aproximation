# Islas distribuidas con MPI (Fase 7) — línea de comandos sobre symreg.evolve_islands_mpi.
#
#   mpirun -np 4 python mpi_islands.py --target propia --gen 250      # 1 isla por rango, 1 GPU por rango
#   mpirun -np 2 python mpi_islands.py --islands 8 --expr "np.sin(x)*x"
#   mpirun -np 4 python mpi_islands.py --csv ejemplos/dos_variables.csv
#   python mpi_islands.py --target nguyen6                              # sin mpirun: 1 rango (serie)
#
# Cada rango usa cuda:<rango local % nº GPUs> si hay CUDA (varias GPUs por nodo,
# varios nodos) o la CPU. Solo el rango 0 lee los datos y escribe los resultados.
import argparse
import json
import pathlib
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--target", default="propia", help="preset de symreg.benchmarks.TARGETS")
    src.add_argument("--expr", help='expresión numpy en x, p.ej. "2*x**3+3*np.sin(x)+1"')
    src.add_argument("--csv", help="CSV sin cabecera: columnas x1..xk, y")
    ap.add_argument("--lo", type=float, default=-5.0); ap.add_argument("--hi", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--islands", type=int, default=None, help="nº total de islas (default: nº de rangos)")
    ap.add_argument("--gen", type=int, default=250); ap.add_argument("--pop", type=int, default=160)
    ap.add_argument("--migrate-every", type=int, default=50); ap.add_argument("--migrants", type=int, default=3)
    ap.add_argument("--val", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finetune-steps", type=int, default=250)
    ap.add_argument("--un", default="sin,cos,log,sqrt,abs,neg"); ap.add_argument("--bin", default="add,sub,mul,div,pow")
    ap.add_argument("--out", default="mpi_resultado", help="prefijo de salida (rango 0): .json árbol + _checkpoint.json")
    args = ap.parse_args()

    import symreg
    comm = symreg.get_comm(); rank, size = comm.Get_rank(), comm.Get_size()

    X = Y = None
    if rank == 0:
        if args.csv:
            A = np.loadtxt(args.csv, delimiter=",")
            X, Y = A[:, :-1].astype(np.float32), A[:, -1].astype(np.float32)
        elif args.expr:
            x = np.linspace(args.lo, args.hi, args.n)
            X = x.astype(np.float32); Y = eval(args.expr, {"np": np, "x": x}).astype(np.float32)
        else:
            fn, lo, hi, npts = symreg.TARGETS[args.target]
            x = np.linspace(lo, hi, npts)
            X = x.astype(np.float32); Y = fn(x).astype(np.float32)

    cat_un, cat_bin = symreg.build_catalogs(args.un.split(","), args.bin.split(","))
    t0 = time.time()
    best, fit, hist, aos, hof = symreg.evolve_islands_mpi(
        X, Y, cat_un, cat_bin, n_islands=args.islands, migrate_every=args.migrate_every,
        migrants=args.migrants, generations=args.gen, pop_size=args.pop, verbose=True,
        val_fraction=args.val, seed=args.seed, finetune_steps=args.finetune_steps,
        restart_patience=80, use_hint=False)
    comm.Barrier()
    if rank == 0:
        Xa = np.asarray(X).reshape(len(X), -1)
        best = symreg.recognize_constants(best, Xa, np.asarray(Y))   # con todos los datos (train+val)
        f_sym, xs = symreg.to_sympy(best)
        print(f"\n== {size} rangos, {args.islands or size} islas, {time.time()-t0:.1f} s ==")
        print("motor :", best.to_text())
        print("sympy :", f_sym)
        print(f"fit   : {fit:.6g}   (hist: {[round(h, 6) for h in hist[-5:]]})")
        symreg.save_tree(best, args.out + ".json")
        symreg.save_checkpoint(args.out + "_checkpoint.json", hof, aos,
                               extra=dict(rangos=size, islas=args.islands or size, gen=args.gen,
                                          seed=args.seed, t=round(time.time()-t0, 1)))
        print("guardado:", args.out + ".json", "y", args.out + "_checkpoint.json")


if __name__ == "__main__":
    main()
