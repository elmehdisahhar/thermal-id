"""Mini-entraînement de bout en bout sur des batches synthétiques."""

import json
from pathlib import Path

import torch
from test_data import fake_sim

from thermal_id.data import Normalizer, compact_batch
from thermal_id.model import CNNConfig
from thermal_id.train import TrainConfig, load_trained, train


def make_processed(root: Path) -> Path:
    splits = {"train": [1, 2], "val": [3], "test": [4]}
    batches = [compact_batch(fake_sim(n=48, seed=k), batch_id=k) for k in range(1, 5)]
    for b in batches:
        b.save(root / f"batch_{b.batch_id:02d}.pt")
    Normalizer.fit(batches[:2]).save(root / "normalizer.json")
    (root / "manifest.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return root


def test_train_end_to_end_and_resume(tmp_path: Path) -> None:
    processed = make_processed(tmp_path)
    cfg = TrainConfig(
        processed_dir=str(processed), out_dir=str(tmp_path / "run"), epochs=2, batch_size=16,
        n_train=80, log_every=0, device="cpu", model=CNNConfig(widths=(8, 8, 16), head_hidden=16),
    )
    history = train(cfg)
    assert len(history) == 2 and all(set(h["val_mape"]) == {"HLC", "HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf"}
                                     for h in history)
    model, norm = load_trained(tmp_path / "run" / "best.pt")
    assert model(torch.randn(3, 2, 480)).shape == (3, 5)

    # reprise du même entraînement (même calendrier one-cycle) : l'historique est rechargé, rien à refaire
    cfg.resume = True
    history = train(cfg)
    assert [h["epoch"] for h in history] == [1, 2]


def test_hybrid_variant_trains_and_reloads(tmp_path: Path) -> None:
    processed = make_processed(tmp_path)
    cfg = TrainConfig(
        processed_dir=str(processed), out_dir=str(tmp_path / "hyb"), epochs=1, batch_size=16,
        log_every=0, device="cpu", use_features=True, model=CNNConfig(widths=(8, 8, 16), head_hidden=16),
    )
    train(cfg)
    model, _ = load_trained(tmp_path / "hyb" / "best.pt")
    assert model.config.n_extra == 36
    assert not torch.allclose(model.extra_std, torch.ones(36))      # statistiques du train bien sauvegardées
    out = model(torch.randn(2, 2, 480), torch.randn(2, 36))
    assert out.shape == (2, 5)
