"""Baseline : régression linéaire (moindres carrés) des cibles normalisées sur les descripteurs.

Usage :
    uv run python -m thermal_id.baseline           # data/processed -> results/baseline_val.json

Ajustement sur le train, évaluation sur la validation. Le test (batches 19-20) n'est pas touché :
il est réservé à l'évaluation finale, une seule fois, du modèle retenu.
L'ajustement passe par les équations normales accumulées batch par batch (X^T X est petit), pour
ne jamais charger tout le train en mémoire.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from thermal_id.data import RAW_TARGETS, CompactBatch, Normalizer
from thermal_id.features import ALL_FEATURE_NAMES as FEATURE_NAMES
from thermal_id.features import all_features
from thermal_id.metrics import mape_by_group, regression_report
from thermal_id.paths import PROCESSED_DIR, RESULTS_DIR


@dataclass
class LinearBaseline:
    """z = [1, (X - x_mean) / x_std, (termes quadratiques)] @ W, avec z les cibles normalisées."""

    x_mean: torch.Tensor   # (k,)
    x_std: torch.Tensor    # (k,)
    W: torch.Tensor        # (p, 5)
    degree: int = 1

    @staticmethod
    def design(xs: torch.Tensor, degree: int) -> torch.Tensor:
        """Matrice de régression à partir des descripteurs standardisés (N, k) -> (N, p)."""
        cols = [torch.ones(len(xs), 1, dtype=xs.dtype), xs]
        if degree == 2:
            i, j = torch.triu_indices(xs.shape[1], xs.shape[1])
            cols.append(xs[:, i] * xs[:, j])
        return torch.cat(cols, dim=1)

    @classmethod
    def fit(
        cls,
        chunks: Iterable[tuple[torch.Tensor, torch.Tensor]],
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        degree: int = 1,
        ridge: float = 1e-6,
    ) -> LinearBaseline:
        """chunks : couples (descripteurs bruts (n, k), cibles normalisées (n, 5))."""
        xtx = xty = None
        n = 0
        for X, Z in chunks:
            A = cls.design(((X - x_mean) / x_std).double(), degree)
            z = Z.double()
            xtx = A.T @ A if xtx is None else xtx + A.T @ A
            xty = A.T @ z if xty is None else xty + A.T @ z
            n += len(A)
        assert xtx is not None and xty is not None, "aucune donnée"
        reg = ridge * n * torch.eye(xtx.shape[0], dtype=xtx.dtype)
        reg[0, 0] = 0.0                                      # pas de pénalité sur la constante
        W = torch.linalg.solve(xtx + reg, xty)
        return cls(x_mean=x_mean, x_std=x_std, W=W, degree=degree)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Descripteurs bruts (N, k) -> cibles normalisées prédites (N, 5)."""
        A = self.design(((X - self.x_mean) / self.x_std).double(), self.degree)
        return (A @ self.W).float()


# --------------------------------------------------------------------------- pipeline
def batch_features(processed_dir: Path, k: int) -> dict[str, torch.Tensor]:
    """Descripteurs d'un batch compact, mis en cache sur disque (features_XX.pt)."""
    cache = processed_dir / f"features_{k:02d}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True)
    b = CompactBatch.load(processed_dir / f"batch_{k:02d}.pt")
    out = {"X": all_features(b.T, b.U_on), "y": b.y, "min_duration_h": b.min_duration_h, "p_on": b.p_on}
    torch.save(out, cache)
    return out


def quintiles(v: torch.Tensor) -> torch.Tensor:
    """Numéro de quintile (0 à 4) de chaque valeur."""
    edges = torch.quantile(v.double(), torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64))
    return torch.bucketize(v.double(), edges)


def run(processed_dir: Path, out_dir: Path) -> dict:
    """Ajuste les baselines (degrés 1 et 2) sur le train, les évalue sur la validation, sauvegarde modèles et JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    norm = Normalizer.load(processed_dir / "normalizer.json")
    train_ids, val_ids = manifest["splits"]["train"], manifest["splits"]["val"]

    t0 = time.perf_counter()
    feats = {k: batch_features(processed_dir, k) for k in train_ids + val_ids}
    print(f"descripteurs : {time.perf_counter() - t0:.1f} s")

    # statistiques des descripteurs sur le train seulement
    X_train = torch.cat([feats[k]["X"] for k in train_ids]).double()
    x_mean, x_std = X_train.mean(0), X_train.std(0).clamp(min=1e-12)
    del X_train

    def chunks() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        for k in train_ids:
            yield feats[k]["X"], norm.targets(feats[k]["y"])

    val = {key: torch.cat([feats[k][key] for k in val_ids]) for key in ("X", "y", "min_duration_h", "p_on")}
    y_mean_pred = norm.inverse(torch.zeros(len(val["y"]), 5))   # référence triviale : la moyenne du train

    results: dict = {
        "features": list(FEATURE_NAMES),
        "val_batches": val_ids,
        "mean_predictor": regression_report(val["y"], y_mean_pred),
    }
    for degree in (1, 2):
        model = LinearBaseline.fit(chunks(), x_mean, x_std, degree=degree)
        pred = norm.inverse(model.predict(val["X"]))
        results[f"linear_deg{degree}"] = {
            "n_params": int(model.W.numel()),
            "val": regression_report(val["y"], pred),
            "val_mape_by_p_on_quintile": mape_by_group(val["y"], pred, quintiles(val["p_on"])),
            "val_mape_by_min_duration_quintile": mape_by_group(val["y"], pred, quintiles(val["min_duration_h"])),
            "val_mape_no_heat_vs_heat": mape_by_group(val["y"], pred, val["X"][:, FEATURE_NAMES.index("has_heat")]),
        }
        torch.save({"model": model.__dict__, "features": FEATURE_NAMES}, out_dir / f"baseline_deg{degree}.pt")

    (out_dir / "baseline_val.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def print_summary(results: dict) -> None:
    rows = [("moyenne du train", results["mean_predictor"])]
    rows += [(f"linéaire degré {d}", results[f"linear_deg{d}"]["val"]) for d in (1, 2)]
    print(f"\nMAPE validation (%)  {'':<4}" + "".join(f"{t:>9}" for t in RAW_TARGETS))
    for name, rep in rows:
        print(f"{name:<24}" + "".join(f"{rep[t]['mape']:9.2f}" for t in RAW_TARGETS))
    print(f"\nR² validation {'':<10}" + "".join(f"{t:>9}" for t in RAW_TARGETS))
    for name, rep in rows[1:]:
        print(f"{name:<24}" + "".join(f"{rep[t]['r2']:9.3f}" for t in RAW_TARGETS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print_summary(run(args.processed_dir, args.out))


if __name__ == "__main__":
    main()
