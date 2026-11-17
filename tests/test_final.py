"""Arrondi de la température, évaluation finale de bout en bout et graphiques du rapport (données synthétiques)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage pendant les tests

import torch
from test_data import fake_sim
from test_train import make_processed

from thermal_id import baseline, final, mlp_features, plots
from thermal_id.data import RAW_TARGETS, compact_batch
from thermal_id.model import CNNConfig
from thermal_id.train import TrainConfig, train


def test_compact_batch_rounds_temperature() -> None:
    sim = fake_sim(n=8)
    exact = compact_batch(sim, batch_id=1)
    rounded = compact_batch(sim, batch_id=1, t_decimals=2)
    assert torch.equal(exact.T, sim.indoor_temperature)                      # sans option : rien ne change
    assert torch.allclose(rounded.T, torch.round(sim.indoor_temperature * 100) / 100)
    assert (rounded.T - exact.T).abs().max() <= 0.005 + 1e-6                  # erreur d'arrondi au plus 0,005 °C
    assert torch.equal(rounded.U_on, exact.U_on) and torch.equal(rounded.y, exact.y)


def test_final_evaluates_all_models_on_the_same_test_set(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    processed = make_processed(tmp_path / "data")
    results = tmp_path / "results"
    baseline.run(processed, results)
    mlp_features.run(processed, results / "mlp_features_val.json", epochs=1, batch_size=16, max_lr=1e-3, seed=0)
    cfg = TrainConfig(processed_dir=str(processed), out_dir=str(tmp_path / "run"), epochs=1, batch_size=16,
                      log_every=0, device="cpu", model=CNNConfig(widths=(8, 8, 16), head_hidden=16))
    train(cfg)

    res = final.run(processed, results, tmp_path / "run" / "best.pt")
    assert list(res["models"]) == ["moyenne du train", "régression linéaire (degré 2)",
                                   "MLP sur descripteurs", "CNN 1D (modèle retenu)"]
    n = res["n"]
    assert n == 48                                                            # un batch de test synthétique
    for name in res["models"]:
        assert set(res["models"][name]["report"]) == set(RAW_TARGETS)
        assert res["_pred"][name].shape == (n, len(RAW_TARGETS))
    # HLC = HLC_mit + HLC_ext pour chaque modèle (contrainte respectée par construction)
    for name in res["models"]:
        p = res["_pred"][name]
        assert torch.allclose(p[:, 0], p[:, 1] + p[:, 2], rtol=1e-5)


def test_report_figures_build_without_error() -> None:
    plots.apply_style()
    rep = {t: {"mae": 1.0, "mape": 10.0, "r2": 0.3} for t in RAW_TARGETS}
    hist = [{"epoch": e, "train_loss": 1 / e, "val_loss": 1.1 / e, "val_mape": {"HLC": 10 - e}} for e in (1, 2, 3)]
    groups = {"mape_by_p_on_quintile": {str(k): {"HLC": 8.0 + k} for k in range(5)}}
    y = torch.rand(200, len(RAW_TARGETS)).numpy() * 100 + 1
    figs = [
        plots.learning_curves({"a": hist, "b": hist}, baseline_loss=0.8),
        plots.metric_by_model({"CNN 1D (modèle retenu)": rep, "variante": rep}, "mape"),
        plots.predicted_vs_true(y, y * 0.9),
        plots.mape_by_group({"CNN 1D (modèle retenu)": groups}, "mape_by_p_on_quintile", "HLC",
                            ["Q1", "Q2", "Q3", "Q4", "Q5"], "quintile"),
        plots.arx_order_scan({"1": {"p10": 1e-3, "median": 1e-2, "p90": 1e-1},
                              "4": {"p10": 1e-6, "median": 5e-6, "p90": 1e-5}}, 5e-6),
    ]
    assert all(f.axes for f in figs)
