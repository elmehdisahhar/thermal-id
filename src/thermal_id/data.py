"""Préparation des données : contrôle, format compact, découpage, normalisation, Dataset PyTorch.

Décisions issues de l'exploration (notebook, section 7) :
- entrée du modèle : les deux séries (T_int, U) sur 480 pas de 1 min ;
- U ne prend que les valeurs 0 et 7000 W : stockée comme indicateur marche/arrêt (uint8) ;
- sorties du modèle : HLC_mit, HLC_ext, tau_b, tau_n, tau_inf ; HLC = HLC_mit + HLC_ext par construction ;
- normalisation : standardisation de H_b, log puis standardisation des tau, statistiques du seul train ;
- découpage par batch : aucun bâtiment n'est commun à deux batches.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from thermal_id.simul_io import ToSaveSimul

N_STEPS = 480
P_ON_W = 7000.0

# cibles brutes, dans l'ordre stocké : HLC, Hb (mit, ext), CDT (tau_b, tau_n, tau_inf)
RAW_TARGETS: tuple[str, ...] = ("HLC", "HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf")
# sorties prédites par le modèle (HLC est déduit par somme)
MODEL_TARGETS: tuple[str, ...] = ("HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf")
LOG_TARGETS: tuple[str, ...] = ("tau_b", "tau_n", "tau_inf")

DEFAULT_SPLITS: dict[str, list[int]] = {
    "train": list(range(1, 17)),
    "val": [17, 18],
    "test": [19, 20],
}


# --------------------------------------------------------------------------- contrôles
def validate_batch(sim: ToSaveSimul, hlc_tol: float = 1e-3) -> list[str]:
    """Contrôles d'intégrité d'un batch brut. Renvoie la liste des anomalies (vide si tout va bien)."""
    issues: list[str] = []
    n = sim.indoor_temperature.shape[0]
    expected = {
        "indoor_temperature": (n, N_STEPS),
        "indoor_U": (n, N_STEPS),
        "HLC": (n,),
        "Hb": (n, 2),
        "CDT": (n, 3),
    }
    for name, shape in expected.items():
        tensor = getattr(sim, name)
        if tuple(tensor.shape) != shape:
            issues.append(f"{name} : forme {tuple(tensor.shape)} au lieu de {shape}")
        elif not torch.isfinite(tensor).all():
            issues.append(f"{name} : valeurs non finies")
    if issues:
        return issues
    if list(sim.Hb_labels) != ["HLC_mit", "HLC_ext"]:
        issues.append(f"Hb_labels inattendus : {sim.Hb_labels}")
    gap = (sim.HLC - sim.Hb.sum(1)).abs().max().item()
    if gap > hlc_tol:
        issues.append(f"HLC != somme(Hb) (écart max {gap:.2e})")
    levels = torch.unique(sim.indoor_U)
    if not torch.isin(levels, torch.tensor([0.0, P_ON_W])).all():
        issues.append(f"U prend d'autres valeurs que 0 et {P_ON_W:.0f} W : {levels[:10].tolist()}")
    if (sim.HLC <= 0).any() or (sim.CDT <= 0).any():
        issues.append("cibles négatives ou nulles")
    return issues


# --------------------------------------------------------------------------- format compact
@dataclass
class CompactBatch:
    """Batch allégé : ce dont l'entraînement et l'évaluation ont besoin."""

    T: torch.Tensor          # (N, 480) float32, température intérieure (°C)
    U_on: torch.Tensor       # (N, 480) uint8, chauffage en marche (1) ou à l'arrêt (0)
    y: torch.Tensor          # (N, 6) float32, cibles brutes dans l'ordre RAW_TARGETS
    min_duration_h: torch.Tensor  # (N,) paramètre PRBS (rythme), pour l'analyse d'erreur
    p_on: torch.Tensor       # (N,) paramètre PRBS (quantité de chauffage)
    batch_id: int

    def __len__(self) -> int:
        return self.T.shape[0]

    def save(self, path: str | Path) -> None:
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> CompactBatch:
        return cls(**torch.load(path, weights_only=True))


def compact_batch(sim: ToSaveSimul, batch_id: int, t_decimals: int | None = None) -> CompactBatch:
    """Convertit un batch brut au format compact.

    ``t_decimals`` : arrondit T_int (2 = résolution 0,01 °C, comme l'exemple de chargement de
    l'énoncé, ``indoor_temperature.round(decimals=2)``). ``None`` garde la précision de stockage.
    U n'a pas besoin d'être arrondi : il ne prend que les valeurs 0 et 7000 W.
    """
    fields = list(sim.solicitation_field_names)
    T = sim.indoor_temperature.to(torch.float32)
    if t_decimals is not None:
        T = T.round(decimals=t_decimals)
    return CompactBatch(
        T=T.contiguous(),
        U_on=(sim.indoor_U > 0).to(torch.uint8),
        y=torch.cat([sim.HLC.unsqueeze(1), sim.Hb, sim.CDT], dim=1).to(torch.float32),
        min_duration_h=sim.solicitation[:, fields.index("prbs_min_duration_hours")].clone(),
        p_on=sim.solicitation[:, fields.index("prbs_probability_on")].clone(),
        batch_id=batch_id,
    )


def concat_batches(batches: list[CompactBatch]) -> CompactBatch:
    """Concatène plusieurs batches compacts (batch_id = -1 si plusieurs)."""
    ids = {b.batch_id for b in batches}
    return CompactBatch(
        T=torch.cat([b.T for b in batches]),
        U_on=torch.cat([b.U_on for b in batches]),
        y=torch.cat([b.y for b in batches]),
        min_duration_h=torch.cat([b.min_duration_h for b in batches]),
        p_on=torch.cat([b.p_on for b in batches]),
        batch_id=ids.pop() if len(ids) == 1 else -1,
    )


def batch_ids_column(batches: list[CompactBatch]) -> torch.Tensor:
    """Numéro de batch de chaque simulation, aligné sur ``concat_batches``."""
    return torch.cat([torch.full((len(b),), b.batch_id, dtype=torch.int16) for b in batches])


# --------------------------------------------------------------------------- normalisation
@dataclass
class Normalizer:
    """Normalisation des entrées et des cibles, ajustée sur le train uniquement.

    Entrées : canal 0 = (T - t_mean) / t_std ; canal 1 = U en marche (0 ou 1).
    Cibles : z = (f(y) - mean) / std avec f = log pour les tau, identité pour H_b.
    """

    t_mean: float
    t_std: float
    y_mean: list[float]  # dans l'ordre MODEL_TARGETS, espace transformé
    y_std: list[float]

    # -- ajustement
    @classmethod
    def fit(cls, batches: Iterable[CompactBatch]) -> Normalizer:
        """Statistiques calculées batch par batch (sommes en float64) pour limiter la mémoire."""
        n = 0
        t_sum = t_sq = 0.0
        y_sum = torch.zeros(len(MODEL_TARGETS), dtype=torch.float64)
        y_sq = torch.zeros_like(y_sum)
        for b in batches:
            t = b.T.double()
            t_sum += t.sum().item()
            t_sq += (t * t).sum().item()
            z = _transform_targets(b.y).double()
            y_sum += z.sum(0)
            y_sq += (z * z).sum(0)
            n += len(b)
        n_t_all = n * N_STEPS
        t_mean = t_sum / n_t_all
        t_std = (t_sq / n_t_all - t_mean**2) ** 0.5
        y_mean = y_sum / n
        y_std = (y_sq / n - y_mean**2).sqrt()
        return cls(t_mean=t_mean, t_std=t_std, y_mean=y_mean.tolist(), y_std=y_std.tolist())

    # -- entrées
    def inputs(self, T: torch.Tensor, U_on: torch.Tensor) -> torch.Tensor:
        """(N, 480) x 2 -> (N, 2, 480) float32 normalisé."""
        t = (T.to(torch.float32) - self.t_mean) / self.t_std
        return torch.stack([t, U_on.to(torch.float32)], dim=1)

    # -- cibles
    def targets(self, y_raw: torch.Tensor) -> torch.Tensor:
        """Cibles brutes (N, 6) -> cibles normalisées (N, 5) dans l'ordre MODEL_TARGETS."""
        z = _transform_targets(y_raw)
        return (z - self._mean(z)) / self._std(z)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Sorties normalisées (N, 5) -> cibles brutes (N, 6), HLC = HLC_mit + HLC_ext."""
        t = z * self._std(z) + self._mean(z)
        hb = t[:, :2]
        taus = t[:, 2:].exp()
        hlc = hb.sum(1, keepdim=True)
        return torch.cat([hlc, hb, taus], dim=1)

    def _mean(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.y_mean, dtype=ref.dtype, device=ref.device)

    def _std(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.y_std, dtype=ref.dtype, device=ref.device)

    # -- persistance
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _transform_targets(y_raw: torch.Tensor) -> torch.Tensor:
    """(N, 6) brutes -> (N, 5) : H_b inchangé, log pour les tau."""
    hb = y_raw[:, 1:3]
    taus = y_raw[:, 3:6].log()
    return torch.cat([hb, taus], dim=1)


# --------------------------------------------------------------------------- Dataset
class ThermalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simulations d'un découpage, normalisées à la volée.

    Les données compactes restent en mémoire (T float32, U uint8) ; la normalisation est
    appliquée par élément dans ``__getitem__`` pour éviter une copie float32 de tout le jeu.
    """

    def __init__(self, data: CompactBatch, normalizer: Normalizer) -> None:
        self.data = data
        self.norm = normalizer
        self.z = normalizer.targets(data.y)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm.inputs(self.data.T[i : i + 1], self.data.U_on[i : i + 1])[0]
        return x, self.z[i]


def load_split(processed_dir: str | Path, split: str) -> tuple[CompactBatch, torch.Tensor]:
    """Charge et concatène les batches compacts d'un découpage. Renvoie aussi le n° de batch par ligne."""
    processed_dir = Path(processed_dir)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    batches = [CompactBatch.load(processed_dir / f"batch_{k:02d}.pt") for k in manifest["splits"][split]]
    return concat_batches(batches), batch_ids_column(batches)
