"""Chemins du projet, définis une seule fois.

Les commandes (``uv run python -m thermal_id.<étape>``) et les notebooks utilisent ces chemins par
défaut : ils fonctionnent quel que soit le dossier courant (racine du projet ou ``notebooks/``).

    data/raw/quentin_house/   batches bruts fournis (batch_1.pt ... batch_20.pt), non modifiés
    data/processed/           batches compacts, normalisation, manifeste, descripteurs en cache
    runs/                     entraînements (poids, journal, historique)
    results/                  métriques (JSON), prédictions, figures
"""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
"""Racine du projet (le paquet est installé en mode éditable par ``uv sync``)."""

RAW_DIR: Path = ROOT / "data" / "raw" / "quentin_house"
PROCESSED_DIR: Path = ROOT / "data" / "processed"
RUNS_DIR: Path = ROOT / "runs"
RESULTS_DIR: Path = ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
