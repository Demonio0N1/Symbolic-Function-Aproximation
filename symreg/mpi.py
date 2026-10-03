# Islas distribuidas con MPI (Fase 7): un proceso por rango, cada rango evoluciona
# un bloque de islas en SU dispositivo (cuda:<rango local> si hay GPUs, CPU si no)
# y la migración en anillo cruza los rangos con MPI_Sendrecv. El Hall of Fame se
# fusiona con allgather, así que TODOS los rangos devuelven el mismo resultado.
#
#   mpirun -np 4 python mpi_islands.py --target propia          # 4 islas, 1 por rango
#   mpirun -np 2 python mpi_islands.py --islands 8               # 4 islas por rango
#
# mpi4py es opcional: sin él (o fuera de mpirun) la función corre en serie con un
# comunicador de tamaño 1 y produce la misma dinámica que evolve_islands().
import os

import numpy as np

from . import backend, config
from .backend import to_device
from .evolve import HallOfFame, evolve, mse_expr
from .finetune import fine_tune_hof
from .tree import tree_from_json, tree_to_json


class SerialComm:
    """Comunicador de tamaño 1 con la API mínima de mpi4py que usa este módulo."""

    def Get_rank(self): return 0
    def Get_size(self): return 1
    def bcast(self, obj, root=0): return obj
    def gather(self, obj, root=0): return [obj]
    def allgather(self, obj): return [obj]
    def sendrecv(self, sendobj, dest, sendtag=0, source=0, recvtag=0, **kw): return sendobj
    def Barrier(self): pass


def get_comm():
    """COMM_WORLD de mpi4py si está disponible; si no, SerialComm()."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return SerialComm()


def local_rank(comm):
    """Rango dentro del nodo (para elegir GPU): variables del lanzador o Split_type."""
    for var in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID",
                "MV2_COMM_WORLD_LOCAL_RANK", "PMI_LOCAL_RANK"):
        if var in os.environ:
            return int(os.environ[var])
    try:
        from mpi4py import MPI
        if isinstance(comm, MPI.Comm):
            node = comm.Split_type(MPI.COMM_TYPE_SHARED)
            r = node.Get_rank(); node.Free()
            return r
    except Exception:
        pass
    return comm.Get_rank()


def rank_device(comm=None):
    """cuda:(rango_local % n_gpus) si hay CUDA; cpu si no."""
    import torch
    comm = comm or get_comm()
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank(comm) % torch.cuda.device_count())
    return torch.device("cpu")


def _island_block(n_islands, rank, size):
    base, rem = divmod(n_islands, size)
    start = rank*base + min(rank, rem)
    return start, start + base + (1 if rank < rem else 0)


def evolve_islands_mpi(X, Y, cat_un, cat_bin, n_islands=None, migrate_every=50, migrants=3,
                       generations=1000, pop_size=160, verbose=True, hof_k=10,
                       val_fraction=0.0, seed=0, comm=None, device=None,
                       finetune_steps=0, finetune_lr=5e-3, alpha=1e-3, **kwargs):
    """evolve_islands() distribuido: devuelve (best, fit, hist, aos_isla0, hof_global)
    en TODOS los rangos (mismo contenido).

    - `n_islands` (default = nº de rangos) se reparte en bloques contiguos; el
      rango r evoluciona sus islas en secuencia sobre `device` (default
      `rank_device`: una GPU por rango local, o CPU).
    - Los datos solo hacen falta en el rango 0 (los demás pueden pasar None):
      se difunden con bcast. El split de validación usa `seed`, idéntico en todos.
    - Cada rango siembra `seed + 1000*rank` para que sus islas sean distintas.
    - `finetune_steps > 0`: cada rango afina las constantes de SU Hall of Fame
      (Adam) antes del allgather, repartiendo también esa etapa entre GPUs.
    - `**kwargs` pasa a evolve() (curriculum, aos_params, restart_patience, ...)."""
    comm = comm or get_comm()
    rank, size = comm.Get_rank(), comm.Get_size()
    if n_islands is None:
        n_islands = size
    if n_islands < size:
        raise ValueError(f"n_islands={n_islands} < nº de rangos={size}: cada rango necesita >= 1 isla")

    backend.set_device(device if device is not None else rank_device(comm))

    X = comm.bcast(np.asarray(X) if rank == 0 else None, root=0)
    Y = comm.bcast(np.asarray(Y) if rank == 0 else None, root=0)
    X = X.reshape(len(X), -1)
    X_val = Y_val = None
    if val_fraction and val_fraction > 0.0:
        perm = np.random.default_rng(seed).permutation(len(X))
        n_val = max(4, int(len(X)*val_fraction))
        X_val, Y_val = X[perm[:n_val]], Y[perm[:n_val]]
        X, Y = X[perm[n_val:]], Y[perm[n_val:]]

    config.set_seed(seed + 1000*rank)
    start, end = _island_block(n_islands, rank, size)
    n_local = end - start
    prev_rank, next_rank = (rank - 1) % size, (rank + 1) % size

    pops = [None]*n_local; aoss = [None]*n_local
    hof_local = HallOfFame(k=hof_k); hist = []
    done = 0
    while done < generations:
        block = min(migrate_every, generations - done)
        for i in range(n_local):
            _b, _f, _h, aos_i, hof_i = evolve(
                X, Y, cat_un, cat_bin, pop_size=pop_size, generations=block,
                verbose=False, hof_k=hof_k, init_pop=pops[i], init_aos=aoss[i],
                gen_offset=done, alpha=alpha, **kwargs)
            pops[i] = hof_i.final_pop; aoss[i] = aos_i
            for f, e in hof_i.items:
                hof_local.update(e, f)
        # Migración en anillo global: la última isla de este rango alimenta a la
        # primera del siguiente (los árboles viajan serializados a JSON).
        outgoing = [tree_to_json(e) for e in pops[-1][:migrants]]
        incoming = comm.sendrecv(outgoing, dest=next_rank, sendtag=done,
                                 source=prev_rank, recvtag=done)
        incoming = [tree_from_json(d) for d in incoming]
        for i in range(n_local):
            src = incoming if i == 0 else pops[i-1]
            dst = pops[i]
            for k in range(min(migrants, len(src), len(dst))):
                dst[-(k+1)] = src[k].clone()
        done += block
        local_best = hof_local.items[0][0] if hof_local.items else np.inf
        hist.append(float(min(comm.allgather(local_best))))
        if verbose and rank == 0:
            print(f"[islas-mpi] gen {done:4d}  best_global={hist[-1]:.6g}  "
                  f"({size} rangos x {n_local} islas, {backend.info()})", flush=True)

    if finetune_steps and finetune_steps > 0:
        hof_local = fine_tune_hof(hof_local, X, Y, steps=finetune_steps, lr=finetune_lr,
                                  verbose=False, alpha=alpha)

    payload = [(float(f), tree_to_json(e)) for f, e in hof_local.items]
    hof_g = HallOfFame(k=hof_k)
    for items in comm.allgather(payload):
        for f, d in items:
            hof_g.update(tree_from_json(d), f)
    hof_g.final_pop = pops[0]
    if X_val is not None and hof_g.items:
        xv = to_device(X_val); yv = to_device(Y_val.reshape(-1, 1))
        scored = sorted(((mse_expr(e, xv, yv), f, e) for f, e in hof_g.items), key=lambda t: t[0])
        hof_g.val_mse = {e.to_text(): float(vm) for vm, f, e in scored}
        hof_g.items = [(f, e) for vm, f, e in scored]
    f0, e0 = hof_g.items[0]
    return e0.clone(), f0, hist, aoss[0], hof_g
