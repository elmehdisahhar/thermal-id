"""Graphiques du rapport (notebook ``04_resultats.ipynb``).

Conventions :
- une couleur par modèle, fixe d'un graphique à l'autre (la couleur suit le modèle, pas son rang) ;
- traits fins, grille discrète, un seul axe des ordonnées par graphique ;
- les textes restent dans les encres neutres ; la couleur ne sert qu'à identifier les séries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from thermal_id.data import RAW_TARGETS

# --------------------------------------------------------------------------- style
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

MODEL_COLORS: dict[str, str] = {
    "CNN 1D (modèle retenu)": "#2a78d6",          # bleu : le modèle retenu
    "MLP sur descripteurs": "#eb6834",            # orange
    "régression linéaire (degré 2)": "#1baf7a",   # aqua
    "moyenne du train": "#b5b4ad",                # gris : référence triviale
}
"""Couleur de chaque modèle (ordre fixe de la palette catégorielle, gris pour la référence triviale)."""

VARIANT_STYLES: tuple[dict[str, str], ...] = (
    {"color": "#8a8983"},                                     # gris moyen : variante de comparaison
    {"color": "#d6d5cf", "hatch": "////", "edgecolor": "#8a8983"},  # gris clair hachuré : variante dégradée
    {"color": "#2a78d6"},                                     # bleu : configuration retenue
)
"""Styles des variantes d'un même modèle (le bleu reste réservé au modèle retenu ; la dernière variante l'utilise).

Gris neutres et bleu se distinguent par la luminosité : lisible aussi pour les daltoniens et à l'impression."""

TARGET_LABELS: dict[str, str] = {
    "HLC": "HLC", "HLC_mit": "HLC$_{mit}$", "HLC_ext": "HLC$_{ext}$",
    "tau_b": r"$\tau_b$", "tau_n": r"$\tau_n$", "tau_inf": r"$\tau_\infty$",
}
TARGET_UNITS: dict[str, str] = {"HLC": "W/K", "HLC_mit": "W/K", "HLC_ext": "W/K",
                                "tau_b": "h", "tau_n": "h", "tau_inf": "h"}


def apply_style() -> None:
    """Style matplotlib commun à toutes les figures du rapport."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "font.size": 10,
    })


# --------------------------------------------------------------------------- figures
def learning_curves(histories: Mapping[str, Sequence[Mapping]], baseline_loss: float | None = None) -> Figure:
    """Perte de validation (gauche) et MAPE HLC de validation (droite) par epoch, pour chaque entraînement.

    ``histories`` : nom de l'entraînement -> liste des enregistrements de ``history.json``.
    Les pertes d'entraînement sont tracées en pointillés, de la même couleur.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.2))
    styles = VARIANT_STYLES[-len(histories):]   # la dernière série (configuration retenue) en bleu
    for style, (name, hist) in zip(styles, histories.items()):
        color = style["color"] if "hatch" not in style else style["edgecolor"]
        ep = [h["epoch"] for h in hist]
        ax_l.plot(ep, [h["val_loss"] for h in hist], "-o", color=color, label=f"{name} : validation")
        ax_l.plot(ep, [h["train_loss"] for h in hist], "--", color=color, lw=1.4, label=f"{name} : train")
        ax_r.plot(ep, [h["val_mape"]["HLC"] for h in hist], "-o", color=color, label=name)
    if baseline_loss is not None:
        ax_l.axhline(baseline_loss, color=INK_MUTED, ls=":", lw=1.4, label="régression linéaire (validation)")
    ax_l.set(xlabel="epoch", ylabel="perte MSE (cibles normalisées)", title="Perte au fil de l'entraînement")
    ax_r.set(xlabel="epoch", ylabel="MAPE HLC en validation (%)", title="Erreur sur HLC au fil de l'entraînement")
    ax_l.legend()
    ax_r.legend()
    fig.tight_layout()
    return fig


def metric_by_model(reports: Mapping[str, Mapping[str, Mapping[str, float]]], metric: str = "mape",
                    targets: Sequence[str] = RAW_TARGETS, title: str = "") -> Figure:
    """Barres groupées : une métrique (``mape`` ou ``r2``) par cible et par modèle.

    ``reports`` : nom du modèle -> rapport ``regression_report`` (cible -> {mae, mape, r2}).
    Les modèles connus prennent leur couleur de ``MODEL_COLORS`` ; sinon, les styles de ``VARIANT_STYLES``
    (la dernière variante, en bleu, est la configuration retenue).
    """
    names = list(reports)
    variant_styles = VARIANT_STYLES[-len(names):]
    x = np.arange(len(targets))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, name in enumerate(names):
        vals = [reports[name][t][metric] for t in targets]
        style = {"color": MODEL_COLORS[name]} if name in MODEL_COLORS else variant_styles[i]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width * 0.92, label=name, **style)
    ax.set_xticks(x, [TARGET_LABELS[t] for t in targets])
    ax.set(ylabel={"mape": "MAPE (%) : plus bas = mieux", "r2": "R² : plus haut = mieux"}[metric], title=title)
    ax.grid(axis="x", visible=False)
    top = max(reports[n][t][metric] for n in names for t in targets)
    ax.set_ylim(top=top * 1.25)                     # marge pour que la légende ne recouvre aucune barre
    ax.legend(ncol=len(names), loc="upper left")
    fig.tight_layout()
    return fig


def predicted_vs_true(y: np.ndarray, pred: np.ndarray, targets: Sequence[str] = RAW_TARGETS,
                      title: str = "") -> Figure:
    """Prédit contre réel pour chaque cible (densité hexagonale, une seule teinte), avec la diagonale y = x."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, t in zip(axes.ravel(), targets):
        j = RAW_TARGETS.index(t)
        yt, yp = y[:, j], pred[:, j]
        lo, hi = np.quantile(yt, [0.005, 0.995])
        ax.hexbin(yt, yp, gridsize=45, extent=(lo, hi, lo, hi), cmap="Blues", mincnt=1, bins="log", linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", label="prédiction parfaite")
        ax.axhline(yt.mean(), color=INK_MUTED, lw=1, ls=":", label="moyenne des cibles")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"réel ({TARGET_UNITS[t]})", ylabel=f"prédit ({TARGET_UNITS[t]})",
               title=TARGET_LABELS[t])
        ax.set_aspect("equal")
        ax.grid(False)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    if title:
        fig.suptitle(title, color=INK)
    fig.tight_layout()
    return fig


def mape_by_group(models: Mapping[str, Mapping[str, Mapping]], key: str, target: str,
                  group_labels: Sequence[str], xlabel: str, title: str = "") -> Figure:
    """MAPE d'une cible par groupe d'excitation (ex. quintiles de ``min_duration``), une ligne par modèle."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, m in models.items():
        groups = m[key]
        vals = [groups[k][target] for k in sorted(groups, key=int)]
        ax.plot(range(len(vals)), vals, "-o", color=MODEL_COLORS.get(name, INK_MUTED), label=name)
    ax.set_xticks(range(len(group_labels)), group_labels)
    ax.set(xlabel=xlabel, ylabel=f"MAPE {TARGET_LABELS[target]} (%)", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def arx_order_scan(arx_order: Mapping[str, Mapping[str, float]], resolution_c: float) -> Figure:
    """Erreur à un pas de l'ARX selon l'ordre (médiane et intervalle p10-p90), échelle log, avec la précision float32."""
    orders = sorted(arx_order, key=int)
    o = np.array([int(k) for k in orders])
    med = np.array([arx_order[k]["median"] for k in orders])
    p10 = np.array([arx_order[k]["p10"] for k in orders])
    p90 = np.array([arx_order[k]["p90"] for k in orders])
    fig, ax = plt.subplots(figsize=(7, 4))
    blue = VARIANT_STYLES[-1]["color"]
    ax.fill_between(o, p10, p90, color=blue, alpha=0.18, lw=0, label="10 %–90 % des simulations")
    ax.plot(o, med, "-o", color=blue, label="médiane")
    ax.axhline(resolution_c, color=INK_MUTED, ls=":", lw=1.4, label="précision de stockage de T (float32)")
    ax.axhline(0.01, color=INK_SECONDARY, ls="--", lw=1.2, label="résolution d'un capteur (0,01 °C)")
    ax.set_yscale("log")
    ax.set(xlabel="ordre du modèle ARX", ylabel="erreur RMS à un pas (°C)",
           title="Un modèle linéaire d'ordre 4 reproduit chaque courbe")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
