"""Métriques d'évaluation, communes à la baseline et au réseau, en unités physiques."""

from __future__ import annotations

import torch

from thermal_id.data import RAW_TARGETS


def regression_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, dict[str, float]]:
    """Cibles brutes (N, 6) dans l'ordre RAW_TARGETS -> {cible: {mae, mape, r2}}.

    mae dans l'unité de la cible (W/K ou h), mape en %, r2 sans unité.
    """
    y_true = y_true.double()
    y_pred = y_pred.double()
    err = y_pred - y_true
    mae = err.abs().mean(0)
    mape = 100 * (err.abs() / y_true.abs()).mean(0)
    ss_res = (err**2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    return {
        name: {"mae": mae[i].item(), "mape": mape[i].item(), "r2": r2[i].item()}
        for i, name in enumerate(RAW_TARGETS)
    }


def mape_by_group(y_true: torch.Tensor, y_pred: torch.Tensor, groups: torch.Tensor) -> dict[str, dict[str, float]]:
    """MAPE (%) par cible pour chaque valeur de ``groups`` (N,), par exemple une famille d'excitation."""
    ape = 100 * ((y_pred.double() - y_true.double()).abs() / y_true.double().abs())
    out: dict[str, dict[str, float]] = {}
    for g in torch.unique(groups).tolist():
        m = groups == g
        out[str(g)] = {"n": int(m.sum())} | {
            name: ape[m, i].mean().item() for i, name in enumerate(RAW_TARGETS)
        }
    return out
