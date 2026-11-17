"""Évaluation finale sur le test (batches 19-20), à lancer une seule fois.

Usage :
    uv run python -m thermal_id.final --processed-dir data/processed_t2 --results-dir results/t2 \
        --cnn runs/cnn_t2_400k/best.pt

Tous les modèles sont figés avant cette étape (ajustés sur le train, choisis sur la validation) ;
le test ne sert qu'à mesurer. Modèles évalués sur les mêmes 100 000 simulations :
- moyenne du train (référence triviale) ;
- régression linéaire de degré 2 sur les descripteurs (``baseline_deg2.pt``) ;
- MLP sur les descripteurs (``mlp_features_val.pt``) ;
- CNN 1D retenu (``best.pt``).

Écrit ``<results-dir>/final_test.json`` et ``<results-dir>/final_test_pred.pt`` (cibles et prédictions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from thermal_id.baseline import LinearBaseline, batch_features, quintiles
from thermal_id.data import RAW_TARGETS, Normalizer
from thermal_id.evaluate import evaluate_checkpoint
from thermal_id.features import ALL_FEATURE_NAMES
from thermal_id.metrics import mape_by_group, regression_report
from thermal_id.mlp_features import make_mlp
from thermal_id.paths import PROCESSED_DIR, RESULTS_DIR


def run(processed_dir: Path, results_dir: Path, cnn_ckpt: Path) -> dict:
    """Évalue les quatre modèles sur le test ; renvoie les rapports et les prédictions."""
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    norm = Normalizer.load(processed_dir / "normalizer.json")
    test_ids = manifest["splits"]["test"]

    feats = [batch_features(processed_dir, k) for k in test_ids]
    X = torch.cat([f["X"] for f in feats])
    y = torch.cat([f["y"] for f in feats])
    has_heat = X[:, ALL_FEATURE_NAMES.index("has_heat")].to(torch.int64)
    p_on = torch.cat([f["p_on"] for f in feats])
    min_dur = torch.cat([f["min_duration_h"] for f in feats])

    preds: dict[str, torch.Tensor] = {"moyenne du train": norm.inverse(torch.zeros(len(y), 5))}

    lin = LinearBaseline(**torch.load(results_dir / "baseline_deg2.pt", weights_only=True)["model"])
    preds["régression linéaire (degré 2)"] = norm.inverse(lin.predict(X))

    saved = torch.load(results_dir / "mlp_features_val.pt", weights_only=True)
    mlp = make_mlp(saved["n_in"])
    mlp.load_state_dict(saved["state_dict"])
    mlp.eval()
    with torch.no_grad():
        preds["MLP sur descripteurs"] = norm.inverse(mlp((X - saved["x_mean"]) / saved["x_std"]))

    cnn = evaluate_checkpoint(cnn_ckpt, processed_dir, "test")
    if not torch.equal(cnn["_pred"]["y"], y):
        raise RuntimeError("ordre des simulations différent entre descripteurs et CNN")
    preds["CNN 1D (modèle retenu)"] = cnn["_pred"]["pred"]

    results: dict = {"test_batches": test_ids, "n": len(y), "t_decimals": manifest.get("t_decimals"),
                     "cnn_checkpoint": str(cnn_ckpt), "models": {}}
    for name, pred in preds.items():
        results["models"][name] = {
            "report": regression_report(y, pred),
            "mape_no_heat_vs_heat": mape_by_group(y, pred, has_heat),
            "mape_by_p_on_quintile": mape_by_group(y, pred, quintiles(p_on)),
            "mape_by_min_duration_quintile": mape_by_group(y, pred, quintiles(min_dur)),
        }
    results["_pred"] = {"y": y, **preds}
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--cnn", type=Path, required=True, help="checkpoint du CNN retenu (best.pt)")
    a = p.parse_args()
    res = run(a.processed_dir, a.results_dir, a.cnn)
    torch.save(res.pop("_pred"), a.results_dir / "final_test_pred.pt")
    (a.results_dir / "final_test.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Test final : {res['n']} simulations (batches {res['test_batches']}), T arrondi à "
          f"{res['t_decimals']} décimales\n")
    print(f"{'MAPE (%)':<32}" + "".join(f"{t:>9}" for t in RAW_TARGETS))
    for name, m in res["models"].items():
        print(f"{name:<32}" + "".join(f"{m['report'][t]['mape']:9.2f}" for t in RAW_TARGETS))
    print(f"\n{'R²':<32}" + "".join(f"{t:>9}" for t in RAW_TARGETS))
    for name, m in res["models"].items():
        print(f"{name:<32}" + "".join(f"{m['report'][t]['r2']:9.3f}" for t in RAW_TARGETS))
    print(f"\n{'MAE (W/K ou h)':<32}" + "".join(f"{t:>9}" for t in RAW_TARGETS))
    for name, m in res["models"].items():
        print(f"{name:<32}" + "".join(f"{m['report'][t]['mae']:9.2f}" for t in RAW_TARGETS))


if __name__ == "__main__":
    main()
