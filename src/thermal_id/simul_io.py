"""Lecture des fichiers ``batch_*.pt`` produits par le simulateur.

Reprise du fichier fourni ``util_assessement.py`` (même classe ``ToSaveSimul`` et même
faux package ``thermal_dynamics_sim``), intégrée au package pour qu'elle soit importable
depuis n'importe quel répertoire de travail (scripts, notebooks, tests).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import torch

DTYPE = torch.float32


@dataclass
class ToSaveSimul:
    """Contenu d'un batch de simulations (structure imposée par les fichiers fournis)."""

    HLC: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=DTYPE))
    CDT: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=DTYPE))
    indoor_temperature: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=DTYPE))
    indoor_U: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=DTYPE))
    solicitation: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=DTYPE))
    solicitation_field_names: list[str] = field(default_factory=list)
    Hb: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=DTYPE))
    Hb_labels: list[str] = field(default_factory=list)


def _register_mock_package() -> None:
    """Enregistre ``thermal_dynamics_sim.utils.data_to_save`` pour que pickle retrouve ``ToSaveSimul``."""
    if "thermal_dynamics_sim.utils.data_to_save" in sys.modules:
        return  # déjà enregistré (par exemple par util_assessement.py dans le même processus)
    main = ModuleType("thermal_dynamics_sim")
    main.__path__ = []  # type: ignore[attr-defined]
    utils = ModuleType("thermal_dynamics_sim.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    data_to_save = ModuleType("thermal_dynamics_sim.utils.data_to_save")
    for module in (main, utils, data_to_save):
        module.ToSaveSimul = ToSaveSimul  # type: ignore[attr-defined]
    main.utils = utils  # type: ignore[attr-defined]
    utils.data_to_save = data_to_save  # type: ignore[attr-defined]
    sys.modules["thermal_dynamics_sim"] = main
    sys.modules["thermal_dynamics_sim.utils"] = utils
    sys.modules["thermal_dynamics_sim.utils.data_to_save"] = data_to_save


_register_mock_package()


def load_single_batch(file_path: str | Path) -> ToSaveSimul:
    """Charge un fichier ``batch_*.pt``.

    ``weights_only=False`` est nécessaire car le fichier contient un objet Python picklé :
    à n'utiliser que sur des fichiers de confiance.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    data = torch.load(file_path, weights_only=False)
    if type(data).__name__ != "ToSaveSimul":  # la classe peut venir de ce module ou de util_assessement
        raise ValueError(f"ToSaveSimul attendu, obtenu {type(data)} dans {file_path}")
    return data


def batch_files(raw_dir: str | Path) -> list[Path]:
    """Liste les ``batch_<k>.pt`` d'un dossier, triés par numéro de batch."""
    files = list(Path(raw_dir).glob("batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"Aucun batch_*.pt dans {raw_dir}")
    return sorted(files, key=batch_number)


def batch_number(path: str | Path) -> int:
    """Numéro de batch extrait du nom de fichier (``batch_12.pt`` -> 12)."""
    return int(Path(path).stem.split("_")[1])
