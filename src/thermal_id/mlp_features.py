"""MLP sur les 36 descripteurs de la baseline : la non-linéarité seule suffit-elle ?

Usage :
    uv run python -m thermal_id.mlp_features --epochs 15 --threads 2

Même entrée que la régression linéaire (descripteurs en cache, ``features_XX.pt``), mais un réseau
dense non linéaire. Comparaison :
- MLP ≈ régression degré 2  -> les descripteurs ont épuisé leur information ;
- CNN > MLP                 -> l'accès à la courbe complète apporte de l'information.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from thermal_id.baseline import batch_features
from thermal_id.data import RAW_TARGETS, Normalizer
from thermal_id.metrics import regression_report
from thermal_id.paths import PROCESSED_DIR, RESULTS_DIR


def make_mlp(n_in: int, hidden: int = 256, depth: int = 3, n_out: int = 5) -> nn.Sequential:
    """MLP dense : ``depth`` couches cachées (Linear + GELU) de largeur ``hidden``, puis sortie linéaire."""
    layers: list[nn.Module] = []
    for i in range(depth):
        layers += [nn.Linear(n_in if i == 0 else hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, n_out))
    return nn.Sequential(*layers)


def run(processed_dir: Path, out: Path, epochs: int, batch_size: int, max_lr: float, seed: int) -> dict:
    """Entraîne le MLP sur le train (one-cycle AdamW), évalue sur la validation, écrit ``out`` (JSON)."""
    torch.manual_seed(seed)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    norm = Normalizer.load(processed_dir / "normalizer.json")

    def load(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        f = [batch_features(processed_dir, k) for k in ids]
        return torch.cat([d["X"] for d in f]), torch.cat([d["y"] for d in f])

    X_tr, y_tr = load(manifest["splits"]["train"])
    X_va, y_va = load(manifest["splits"]["val"])
    mean, std = X_tr.mean(0), X_tr.std(0).clamp(min=1e-6)
    X_tr, X_va = (X_tr - mean) / std, (X_va - mean) / std
    z_tr, z_va = norm.targets(y_tr), norm.targets(y_va)

    model = make_mlp(X_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
    steps = len(X_tr) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=epochs * steps)
    history = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        tot = 0.0
        for s in range(steps):
            j = perm[s * batch_size : (s + 1) * batch_size]
            loss = nn.functional.mse_loss(model(X_tr[j]), z_tr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            pred = model(X_va)
        val_loss = nn.functional.mse_loss(pred, z_va).item()
        history.append({"epoch": ep + 1, "train_loss": tot / steps, "val_loss": val_loss})
        print(f"epoch {ep + 1}/{epochs} | perte train {tot / steps:.4f} | perte val {val_loss:.4f}"
              f" | {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    report = regression_report(y_va, norm.inverse(pred))
    res = {"epochs": epochs, "batch_size": batch_size, "max_lr": max_lr, "history": history, "val": report}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    # poids et standardisation des descripteurs (statistiques du train), pour l'évaluation finale
    torch.save({"state_dict": model.state_dict(), "n_in": X_tr.shape[1], "x_mean": mean, "x_std": std},
               out.with_suffix(".pt"))
    print("MAPE val (%) " + " ".join(f"{t} {report[t]['mape']:.2f}" for t in RAW_TARGETS))
    print("R² val       " + " ".join(f"{t} {report[t]['r2']:.3f}" for t in RAW_TARGETS))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "mlp_features_val.json")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    run(a.processed_dir, a.out, a.epochs, a.batch_size, a.max_lr, a.seed)


if __name__ == "__main__":
    main()
