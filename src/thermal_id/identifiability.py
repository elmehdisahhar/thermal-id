"""Test d'identifiabilité : des courbes presque identiques ont-elles des cibles proches ?

Usage :
    uv run python -m thermal_id.identifiability --batches 11 12 13 14 15 16 --threads 2

Principe. Le simulateur est déterministe et les signaux U se répètent (graine PRBS fixe). Parmi les
simulations qui partagent exactement le même signal U, la seule différence est le bâtiment. Pour chacune,
on cherche la plus proche voisine au sens de la courbe T (écart quadratique moyen, °C) et on compare
leurs cibles. Si deux courbes quasi identiques ont des cibles très différentes, aucun modèle ne peut les
départager à partir de la courbe seule.

Portée : c'est un ordre de grandeur, pas le plancher exact de l'erreur. La distance à la voisine dépend
aussi de la densité de l'échantillon (environ 130 bâtiments par signal U dans le train), et la tranche
des courbes quasi identiques (< 0,01 °C) contient surtout des simulations sans chauffage.

Estimation de la borne : pour deux exemples d'entrées identiques, le meilleur prédicteur donne la moyenne
conditionnelle ; si la dispersion conditionnelle est gaussienne, l'erreur absolue attendue de ce
prédicteur vaut E|y1 - y2| / sqrt(2).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from thermal_id.data import RAW_TARGETS, CompactBatch
from thermal_id.paths import PROCESSED_DIR, RESULTS_DIR

DIST_EDGES_C: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, math.inf)


def nearest_neighbours(T: torch.Tensor, U: torch.Tensor, chunk: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    """Pour chaque simulation : indice de la plus proche voisine de même signal U, et distance RMS (°C).

    Les simulations seules dans leur groupe reçoivent l'indice -1 et une distance infinie.
    """
    _, gid = torch.unique(U, dim=0, return_inverse=True)
    order = torch.argsort(gid)
    counts = torch.bincount(gid)
    nn_idx = torch.full((len(T),), -1, dtype=torch.long)
    nn_dist = torch.full((len(T),), math.inf)
    start = 0
    for c in counts.tolist():
        members = order[start : start + c]
        start += c
        if c < 2:
            continue
        Tg = T[members]
        for i in range(0, c, chunk):
            d = torch.cdist(Tg[i : i + chunk], Tg) / math.sqrt(T.shape[1])
            rows = torch.arange(d.shape[0])
            d[rows, rows + i] = math.inf                 # exclure soi-même
            best, j = d.min(dim=1)
            nn_dist[members[i : i + chunk]] = best
            nn_idx[members[i : i + chunk]] = members[j]
    return nn_idx, nn_dist


def analyse(processed_dir: Path, batches: list[int]) -> dict:
    """Écart de cibles entre plus proches voisines, par tranche de distance ; renvoie un dictionnaire JSON."""
    parts = [CompactBatch.load(processed_dir / f"batch_{k:02d}.pt") for k in batches]
    T = torch.cat([b.T for b in parts])
    U = torch.cat([b.U_on for b in parts])
    y = torch.cat([b.y for b in parts]).double()
    heated = U.sum(1) > 0
    del parts

    nn_idx, nn_dist = nearest_neighbours(T, U)
    ok = nn_idx >= 0
    # écart relatif entre voisines : |log y_i - log y_nn| (≈ écart relatif pour de petits écarts)
    rel = (y[ok].log() - y[nn_idx[ok]].log()).abs()
    dist = nn_dist[ok]
    heat_ok = heated[ok]

    bins = []
    for lo, hi in zip(DIST_EDGES_C[:-1], DIST_EDGES_C[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < 30:
            continue
        floor = 100 * rel[m].mean(0) / math.sqrt(2)
        bins.append({
            "dist_min_C": lo, "dist_max_C": hi, "n": int(m.sum()),
            "frac_chauffees": round(heat_ok[m].double().mean().item(), 3),
            "ecart_relatif_median_pct": {t: round(100 * rel[m, i].median().item(), 2) for i, t in enumerate(RAW_TARGETS)},
            "borne_mape_pct": {t: round(floor[i].item(), 2) for i, t in enumerate(RAW_TARGETS)},
        })
    return {
        "batches": batches,
        "n_simulations": len(T),
        "n_avec_voisine": int(ok.sum()),
        "distance_voisine_quantiles_C": {
            q: round(torch.quantile(dist.double(), q).item(), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "par_distance": bins,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--batches", type=int, nargs="+", default=[11, 12, 13, 14, 15, 16])
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "identifiability.json")
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    res = analyse(a.processed_dir, a.batches)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{res['n_simulations']} simulations, {res['n_avec_voisine']} avec une voisine de même U")
    print("distance à la voisine (°C) :", res["distance_voisine_quantiles_C"])
    print(f"\n{'distance (°C)':<16}{'n':>8}  " + "".join(f"{t:>9}" for t in RAW_TARGETS) + "   (borne MAPE %)")
    for b in res["par_distance"]:
        print(f"[{b['dist_min_C']:.2f}, {b['dist_max_C']:.2f}){'':<3}{b['n']:>8}  "
              + "".join(f"{b['borne_mape_pct'][t]:9.2f}" for t in RAW_TARGETS))


if __name__ == "__main__":
    main()
