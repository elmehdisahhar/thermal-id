"""Prépare les 20 batches bruts pour l'entraînement.

Usage :
    uv run python -m thermal_id.prepare            # data/raw/quentin_house -> data/processed
    uv run python -m thermal_id.prepare --round-t 2 --out-dir data/processed_t2   # T à 0,01 °C (énoncé)

Pour chaque batch (lu un par un pour limiter la mémoire) : contrôle d'intégrité, conversion au
format compact, sauvegarde. Ensuite : contrôle des bâtiments en double entre batches,
ajustement de la normalisation sur le train, écriture d'un manifeste.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from thermal_id.data import DEFAULT_SPLITS, CompactBatch, Normalizer, compact_batch, validate_batch
from thermal_id.paths import PROCESSED_DIR, RAW_DIR
from thermal_id.simul_io import batch_files, batch_number, load_single_batch


def prepare(
    raw_dir: Path, out_dir: Path, splits: dict[str, list[int]] = DEFAULT_SPLITS, t_decimals: int | None = None
) -> dict:
    """Exécute la préparation complète et renvoie le manifeste.

    ``t_decimals`` : arrondi de T_int (2 = 0,01 °C, format de l'énoncé) ; ``None`` = précision de stockage.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = batch_files(raw_dir)
    found = {batch_number(f) for f in files}
    wanted = {k for ids in splits.values() for k in ids}
    if missing := wanted - found:
        raise FileNotFoundError(f"batches manquants dans {raw_dir} : {sorted(missing)}")

    report: dict[int, dict] = {}
    all_targets: list[torch.Tensor] = []
    for f in files:
        k = batch_number(f)
        if k not in wanted:
            continue
        t0 = time.perf_counter()
        sim = load_single_batch(f)
        issues = validate_batch(sim)
        compact = compact_batch(sim, k, t_decimals)
        compact.save(out_dir / f"batch_{k:02d}.pt")
        all_targets.append(compact.y)
        report[k] = {
            "n": len(compact),
            "issues": issues,
            "hlc_mean": round(compact.y[:, 0].mean().item(), 2),
            "tau_b_mean": round(compact.y[:, 3].mean().item(), 2),
            "frac_sans_chauffage": round((compact.U_on.sum(1) == 0).float().mean().item(), 4),
        }
        status = "OK" if not issues else "ANOMALIES : " + " ; ".join(issues)
        print(f"batch {k:2d} : {len(compact)} simulations, {time.perf_counter() - t0:4.1f} s, {status}")
        del sim, compact
        gc.collect()

    targets = torch.cat(all_targets)
    n_duplicates = len(targets) - torch.unique(targets, dim=0).shape[0]
    print(f"vecteurs de cibles en double sur l'ensemble des batches : {n_duplicates}")

    normalizer = Normalizer.fit(_iter_batches(out_dir, splits["train"]))
    normalizer.save(out_dir / "normalizer.json")

    manifest = {
        "splits": splits,
        "t_decimals": t_decimals,
        "n_per_split": {s: sum(report[k]["n"] for k in ids) for s, ids in splits.items()},
        "duplicate_target_vectors": n_duplicates,
        "batches_with_issues": [k for k, r in report.items() if r["issues"]],
        "report": report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _iter_batches(out_dir: Path, ids: list[int]) -> Iterator[CompactBatch]:
    for k in ids:
        yield CompactBatch.load(out_dir / f"batch_{k:02d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="dossier des batch_*.pt bruts")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR, help="dossier de sortie")
    parser.add_argument("--round-t", type=int, default=None, metavar="DECIMALES",
                        help="arrondir T_int (2 = 0,01 °C, format de l'énoncé)")
    args = parser.parse_args()
    manifest = prepare(args.raw_dir, args.out_dir, t_decimals=args.round_t)
    print("simulations par découpage :", manifest["n_per_split"])
    print("batches avec anomalies :", manifest["batches_with_issues"] or "aucun")
    print("normalisation :", json.loads((args.out_dir / "normalizer.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
