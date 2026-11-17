"""Évaluation d'un modèle entraîné, avec les mêmes métriques que la baseline.

Usage :
    uv run python -m thermal_id.evaluate runs/cnn_cpu_400k/best.pt --name cnn_400k
    uv run python -m thermal_id.evaluate runs/cnn_cpu_400k/best.pt --name cnn_400k_round2 --round-t 2   # T à 0,01 °C
    uv run python -m thermal_id.evaluate runs/<modèle retenu>/best.pt --name final --split test   # une seule fois

Écrit ``results/<name>_<split>.json`` (MAE / MAPE / R² par cible, MAPE par famille d'excitation) et
``results/<name>_<split>_pred.pt`` (cibles réelles et prédites, pour les graphiques).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from thermal_id.baseline import quintiles
from thermal_id.data import CompactBatch, Normalizer
from thermal_id.metrics import mape_by_group, regression_report
from thermal_id.paths import PROCESSED_DIR, RESULTS_DIR
from thermal_id.train import load_subset, load_trained, predict


def evaluate_checkpoint(
    ckpt: Path, processed_dir: Path, split: str, threads: int | None = None, t_decimals: int | None = None
) -> dict:
    """Évalue ``ckpt`` sur un découpage ; métriques globales et par famille d'excitation.

    ``t_decimals`` : arrondit T_int avant la prédiction (ex. 2 = résolution 0,01 °C, comme dans
    l'exemple de chargement de l'énoncé) pour mesurer la sensibilité à la résolution du capteur.
    """
    if threads:
        torch.set_num_threads(threads)
    model, norm = load_trained(ckpt)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["splits"][split]
    data = load_subset(processed_dir, ids, None, norm, seed=0, with_features=model.config.n_extra > 0)
    # paramètres PRBS dans le même ordre (load_subset garde l'ordre quand n=None)
    meta = [CompactBatch.load(processed_dir / f"batch_{k:02d}.pt") for k in ids]
    p_on = torch.cat([b.p_on for b in meta])
    min_dur = torch.cat([b.min_duration_h for b in meta])
    del meta
    if t_decimals is not None:
        data.T = data.T.round(decimals=t_decimals)

    pred = norm.inverse(predict(model, data, norm))
    y = data.y
    has_heat = (data.U.sum(1) > 0).to(torch.int64)
    return {
        "checkpoint": str(ckpt),
        "split": split,
        "n": len(y),
        "t_decimals": t_decimals,
        "report": regression_report(y, pred),
        "mape_no_heat_vs_heat": mape_by_group(y, pred, has_heat),
        "mape_by_p_on_quintile": mape_by_group(y, pred, quintiles(p_on)),
        "mape_by_min_duration_quintile": mape_by_group(y, pred, quintiles(min_dur)),
        "_pred": {"y": y, "pred": pred},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--round-t", type=int, default=None, metavar="DECIMALES",
                   help="arrondir T_int avant la prédiction (2 = 0,01 °C, format de l'énoncé)")
    a = p.parse_args()
    res = evaluate_checkpoint(a.checkpoint, a.processed_dir, a.split, a.threads, a.round_t)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(res.pop("_pred"), a.out_dir / f"{a.name}_{a.split}_pred.pt")
    (a.out_dir / f"{a.name}_{a.split}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    rep = res["report"]
    print(f"{a.name} sur {a.split} ({res['n']} simulations)")
    print("MAPE (%) " + " ".join(f"{t} {v['mape']:.2f}" for t, v in rep.items()))
    print("R²       " + " ".join(f"{t} {v['r2']:.3f}" for t, v in rep.items()))


if __name__ == "__main__":
    main()
