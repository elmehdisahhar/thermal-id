"""Entraînement du CNN 1D, identique sur CPU (local) et GPU (Kaggle).

Usage :
    # essai court en local (~15 min sur CPU)
    uv run python -m thermal_id.train --n-train 50000 --n-val 20000 --epochs 2 --out-dir runs/essai
    # modèle retenu (400 000 simulations, 10 epochs, ~1 h sur CPU)
    uv run python -m thermal_id.train --n-train 400000 --epochs 10 --log-every 300 --out-dir runs/cnn_cpu_400k
    # reprise après interruption (même --epochs)
    uv run python -m thermal_id.train --n-train 400000 --epochs 10 --out-dir runs/cnn_cpu_400k --resume

Méthode :
- perte : MSE sur les 5 cibles normalisées (H_b, log tau) ;
- optimiseur : AdamW, taux d'apprentissage « one-cycle », écrêtage du gradient ;
- à chaque epoch : perte et MAPE par cible sur la validation (unités physiques, HLC = somme des H_b),
  comparés à la baseline ; sauvegarde du dernier état (reprise possible) et du meilleur modèle ;
- journal lisible (train_log.txt) et structuré (history.json), avec temps restant estimé.

Les données d'un découpage sont chargées une fois en mémoire (sur le GPU si disponible) et les lots
sont tirés par indices : pas de DataLoader, la normalisation des entrées est faite à la volée.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch import nn

from thermal_id.baseline import batch_features
from thermal_id.data import RAW_TARGETS, CompactBatch, Normalizer
from thermal_id.features import ALL_FEATURE_NAMES
from thermal_id.metrics import regression_report
from thermal_id.model import CNNConfig, ThermalCNN, count_parameters
from thermal_id.paths import PROCESSED_DIR, RESULTS_DIR, RUNS_DIR


@dataclass
class TrainConfig:
    processed_dir: str = str(PROCESSED_DIR)
    out_dir: str = str(RUNS_DIR / "cnn")
    epochs: int = 10
    batch_size: int = 256
    max_lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    n_train: int | None = None        # None : tout le train ; sinon sous-échantillon réparti sur les batches
    n_val: int | None = None
    seed: int = 0
    device: str = "auto"              # "auto", "cpu" ou "cuda"
    amp: bool = True                  # précision mixte (GPU seulement)
    log_every: int = 200              # pas entre deux lignes de progression
    resume: bool = False
    use_features: bool = False        # variante hybride : ajouter les 36 descripteurs d'identification
    model: CNNConfig = field(default_factory=CNNConfig)


# --------------------------------------------------------------------------- données
@dataclass
class TensorSplit:
    """Un découpage chargé en mémoire : entrées compactes, cibles brutes et normalisées."""

    T: torch.Tensor   # (N, 480) float32
    U: torch.Tensor   # (N, 480) uint8
    y: torch.Tensor   # (N, 6) cibles brutes
    z: torch.Tensor   # (N, 5) cibles normalisées
    F: torch.Tensor | None = None   # (N, 36) descripteurs bruts (variante hybride)

    def __len__(self) -> int:
        return self.T.shape[0]

    def to(self, device: torch.device) -> TensorSplit:
        return TensorSplit(*(None if t is None else t.to(device) for t in (self.T, self.U, self.y, self.z, self.F)))

    def extra(self, idx: torch.Tensor | slice) -> torch.Tensor | None:
        return None if self.F is None else self.F[idx]


def load_subset(
    processed_dir: Path, ids: list[int], n: int | None, norm: Normalizer, seed: int, with_features: bool = False
) -> TensorSplit:
    """Charge les batches ``ids`` un par un et garde au plus ``n`` simulations, réparties également.

    Avec ``with_features``, ajoute les descripteurs (cache ``features_XX.pt``, calculé si absent) des mêmes lignes.
    """
    g = torch.Generator().manual_seed(seed)
    per_batch = None if n is None else math.ceil(n / len(ids))
    parts: list[tuple[torch.Tensor, ...]] = []
    for k in ids:
        b = CompactBatch.load(processed_dir / f"batch_{k:02d}.pt")
        cols = [b.T, b.U_on, b.y]
        if with_features:
            cols.append(batch_features(processed_dir, k)["X"])
        if per_batch is not None and per_batch < len(b):
            idx = torch.randperm(len(b), generator=g)[:per_batch]
            cols = [c[idx] for c in cols]
        parts.append(tuple(cols))
        del b
    cat = [torch.cat(p) for p in zip(*parts)]
    if n is not None:
        cat = [c[:n] for c in cat]
    T, U, y = cat[:3]
    return TensorSplit(T=T, U=U, y=y, z=norm.targets(y), F=cat[3] if with_features else None)


# --------------------------------------------------------------------------- évaluation
@torch.no_grad()
def predict(model: nn.Module, split: TensorSplit, norm: Normalizer, batch_size: int = 4096) -> torch.Tensor:
    """Sorties normalisées (N, 5) du modèle en mode évaluation."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(split), batch_size):
        sl = slice(i, i + batch_size)
        x = norm.inputs(split.T[sl], split.U[sl]).to(device)
        extra = split.extra(sl)
        if extra is not None:
            extra = extra.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x, extra).float())
    return torch.cat(out)


def evaluate(model: nn.Module, split: TensorSplit, norm: Normalizer) -> tuple[float, dict[str, dict[str, float]]]:
    """(perte MSE normalisée, rapport MAE / MAPE / R² en unités physiques)."""
    z_pred = predict(model, split, norm)
    loss = nn.functional.mse_loss(z_pred, split.z.to(z_pred.device)).item()
    report = regression_report(split.y.cpu(), norm.inverse(z_pred.cpu()))
    return loss, report


# --------------------------------------------------------------------------- entraînement
def resolve_device(name: str) -> torch.device:
    """« auto » : GPU si disponible, sinon CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Logger:
    """Écrit chaque message à l'écran et dans un fichier journal."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def train(cfg: TrainConfig) -> list[dict]:
    """Entraîne le CNN selon ``cfg`` ; écrit best.pt, last.pt, history.json et train_log.txt dans ``out_dir``.

    Renvoie l'historique par epoch (pertes, MAPE et R² de validation).
    """
    torch.manual_seed(cfg.seed)
    if cfg.use_features:
        cfg.model = replace(cfg.model, n_extra=len(ALL_FEATURE_NAMES))
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    processed, out = Path(cfg.processed_dir), Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "train_log.txt")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    norm = Normalizer.load(processed / "normalizer.json")
    t0 = time.perf_counter()
    train_set = load_subset(processed, manifest["splits"]["train"], cfg.n_train, norm, cfg.seed,
                            cfg.use_features).to(device)
    val_set = load_subset(processed, manifest["splits"]["val"], cfg.n_val, norm, cfg.seed + 1, cfg.use_features)
    log(f"appareil : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                  f" ({torch.get_num_threads()} threads)"))
    log(f"données : train {len(train_set)}, validation {len(val_set)} ({time.perf_counter() - t0:.0f} s)")

    model = ThermalCNN(cfg.model).to(device)
    if train_set.F is not None:   # standardisation des descripteurs : statistiques du train
        model.set_extra_stats(train_set.F.mean(0), train_set.F.std(0))
        log(f"variante hybride : {train_set.F.shape[1]} descripteurs ajoutés avant la tête")
    log(f"modèle : {count_parameters(model)} paramètres")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_set) // cfg.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.max_lr, total_steps=cfg.epochs * steps_per_epoch)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_val = math.inf
    start_epoch = 0
    if cfg.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_val, start_epoch = state["history"], state["best_val"], state["epoch"] + 1
        log(f"reprise après l'epoch {start_epoch}")

    baseline = _baseline_mape()
    t_train = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t_epoch = time.perf_counter()
        perm = torch.randperm(len(train_set), device=device)
        running, n_seen = 0.0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            x = norm.inputs(train_set.T[idx], train_set.U[idx])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x, train_set.extra(idx))
            loss = nn.functional.mse_loss(pred.float(), train_set.z[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_seen += 1
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                # temps restant estimé sur le rythme de l'epoch en cours (insensible à une mise en veille passée)
                rate = (time.perf_counter() - t_epoch) / (step + 1)
                left = (steps_per_epoch - step - 1) + (cfg.epochs - epoch - 1) * steps_per_epoch
                eta = rate * left
                log(f"  epoch {epoch + 1} pas {step + 1}/{steps_per_epoch} | perte train {running / n_seen:.4f}"
                    f" | lr {sched.get_last_lr()[0]:.2e} | reste ~{eta / 60:.0f} min")

        val_loss, report = evaluate(model, val_set, norm)
        eta = (time.perf_counter() - t_epoch) * (cfg.epochs - epoch - 1)
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(n_seen, 1),
            "val_loss": val_loss,
            "val_mape": {t: report[t]["mape"] for t in RAW_TARGETS},
            "val_r2": {t: report[t]["r2"] for t in RAW_TARGETS},
            "epoch_min": (time.perf_counter() - t_epoch) / 60,
        }
        history.append(record)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(), "model_config": asdict(cfg.model), "normalizer": asdict(norm),
                 "epoch": epoch + 1, "val_loss": val_loss, "val_report": report},
                out / "best.pt",
            )
        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "scaler": scaler.state_dict(), "epoch": epoch, "best_val": best_val, "history": history},
            out / "last.pt",
        )
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        log(f"epoch {epoch + 1}/{cfg.epochs} | perte train {record['train_loss']:.4f} | perte val {val_loss:.4f}"
            f"{' (meilleur)' if improved else ''} | {record['epoch_min']:.1f} min | reste ~{eta / 60:.0f} min")
        log("  MAPE val (%) " + " ".join(
            f"{t} {report[t]['mape']:.2f}" + (f" [base {baseline[t]:.2f}]" if baseline else "") for t in RAW_TARGETS))

    log(f"terminé en {(time.perf_counter() - t_train) / 60:.1f} min ; meilleure perte val {best_val:.4f}")
    return history


def _baseline_mape(path: Path = RESULTS_DIR / "baseline_val.json") -> dict[str, float] | None:
    """MAPE de la baseline (degré 2) pour comparaison dans le journal, si le fichier existe."""
    if not path.exists():
        return None
    val = json.loads(path.read_text(encoding="utf-8"))["linear_deg2"]["val"]
    return {t: val[t]["mape"] for t in RAW_TARGETS}


def load_trained(path: str | Path, device: str = "cpu") -> tuple[ThermalCNN, Normalizer]:
    """Recharge un modèle sauvegardé (best.pt) et sa normalisation."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mc = dict(ckpt["model_config"])
    mc["widths"] = tuple(mc["widths"])
    model = ThermalCNN(CNNConfig(**mc))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device), Normalizer(**ckpt["normalizer"])


# --------------------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = TrainConfig()
    p.add_argument("--processed-dir", default=d.processed_dir)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--max-lr", type=float, default=d.max_lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--use-features", action="store_true", help="variante hybride : CNN + 36 descripteurs")
    p.add_argument("--norm", default=d.model.norm, choices=["group", "batch"])
    p.add_argument("--dropout", type=float, default=d.model.dropout)
    a = p.parse_args()
    cfg = TrainConfig(
        processed_dir=a.processed_dir, out_dir=a.out_dir, epochs=a.epochs, batch_size=a.batch_size,
        max_lr=a.max_lr, weight_decay=a.weight_decay, n_train=a.n_train, n_val=a.n_val, seed=a.seed,
        device=a.device, amp=not a.no_amp, log_every=a.log_every, resume=a.resume, use_features=a.use_features,
        model=CNNConfig(norm=a.norm, dropout=a.dropout),
    )
    train(cfg)


if __name__ == "__main__":
    main()
