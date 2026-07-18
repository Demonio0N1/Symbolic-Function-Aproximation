# Checkpoints de corridas largas (Fase 6.3): guarda/carga el Hall of Fame,
# la población final y los scores del AOS en un JSON.
import json

from .tree import tree_from_json, tree_to_json


def save_checkpoint(path, hof, aos=None, extra=None):
    """Serializa el estado de una corrida: hof (items + final_pop) y scores AOS."""
    d = {
        "items": [{"fit": float(f), "tree": tree_to_json(e)} for f, e in hof.items],
        "final_pop": [tree_to_json(e) for e in (hof.final_pop or [])],
        "val_mse": hof.val_mse,
    }
    if aos is not None:
        d["aos"] = {"un_scores": aos.un_scores, "bin_scores": aos.bin_scores}
    if extra:
        d["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


def load_checkpoint(path):
    """Devuelve dict con 'items' [(fit, expr)], 'final_pop' [expr], 'aos', 'extra'.
    La población puede pasarse a evolve(init_pop=...) para continuar la corrida."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "items": [(it["fit"], tree_from_json(it["tree"])) for it in d.get("items", [])],
        "final_pop": [tree_from_json(t) for t in d.get("final_pop", [])],
        "val_mse": d.get("val_mse"),
        "aos": d.get("aos"),
        "extra": d.get("extra"),
    }
