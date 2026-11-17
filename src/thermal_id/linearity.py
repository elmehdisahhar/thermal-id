"""Le simulateur est-il linéaire et sans bruit ? Peut-on identifier HLC et tau simulation par simulation ?

Usage :
    uv run python -m thermal_id.linearity                     # tests ARX (≈ 1 min)
    uv run python -m thermal_id.linearity --modes 2 3 4       # + ajustement à N modes, tau libres (≈ 5 min)

Deux tests, sur des simulations du train (jamais sur le test) :

1. Ordre ARX. Pour chaque simulation, régression (float64)
       T[k] = c + sum_{i=1..n} a_i T[k-i] + sum_{i=0..n} b_i U[k-i]
   Si le simulateur est linéaire invariant d'ordre n et sans bruit, l'erreur à un pas tombe à la
   précision de stockage de T (float32 : ≈ 4e-6 °C) dès l'ordre n, puis ne baisse plus.

2. Identification directe. Le gain statique de l'ARX, G = sum(b) / (1 - sum(a)) en °C par unité
   de U, donne HLC = P_ON / G ; les pôles donnent les constantes de temps. On compare aux vraies cibles.
   Variante « erreur de sortie » (``--modes``) : T = c + réponses libres + réponses forcées de N
   premiers ordres à tau libres, optimisés par descente de gradient.

Lecture : un ajustement parfait (test 1) qui ne retrouve pas HLC (test 2) signifie que plusieurs
bâtiments différents produisent des courbes indiscernables sur 8 h : la limite vient de la fenêtre
d'observation, pas du bruit ni du modèle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from thermal_id.data import P_ON_W, RAW_TARGETS, CompactBatch
from thermal_id.paths import PROCESSED_DIR, RESULTS_DIR

DT_MIN = 1.0
LAG = 0
"""Retard entre U et T, en pas. Mesuré sur les données brutes : quand le chauffage s'allume au pas k,
le saut de pente de T a lieu entre k-1 et k, donc T[k] répond déjà à U[k]."""

ARX_ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10)


# --------------------------------------------------------------------------- test 1 : ordre ARX
def arx_fit(T: torch.Tensor, U: torch.Tensor, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Moindres carrés ARX par simulation (float64).

    Renvoie (coefficients (S, 2 + 2*order) dans l'ordre [c, a_1..a_n, b_0..b_n], erreur RMS à un pas (S,)).
    """
    n = order
    T, U = T.double(), U.double()
    cols = [torch.ones_like(T[:, n:])]
    cols += [T[:, n - i : T.shape[1] - i] for i in range(1, n + 1)]
    cols += [U[:, n - i : U.shape[1] - i] for i in range(0, n + 1)]
    X = torch.stack(cols, dim=2)
    target = T[:, n:]
    coef = torch.linalg.lstsq(X, target[..., None]).solution[..., 0]
    rms = (target - (X @ coef[..., None])[..., 0]).pow(2).mean(1).sqrt()
    return coef, rms


def arx_static_gain(coef: torch.Tensor, order: int) -> torch.Tensor:
    """Gain statique (°C par unité de U_on) : sum(b) / (1 - sum(a))."""
    a, b = coef[:, 1 : order + 1], coef[:, order + 1 :]
    return b.sum(1) / (1 - a.sum(1))


def arx_time_constants_h(coef: torch.Tensor, order: int) -> torch.Tensor:
    """Constantes de temps (h) des pôles de l'ARX, triées par ordre décroissant (partie réelle des pôles)."""
    a = coef[:, 1 : order + 1]
    comp = torch.zeros(len(a), order, order, dtype=a.dtype)
    comp[:, 0, :] = a
    if order > 1:
        comp[:, 1:, :-1] = torch.eye(order - 1, dtype=a.dtype)
    p = torch.linalg.eigvals(comp).real.clamp(1e-12, 1 - 1e-12)
    return (-DT_MIN / torch.log(p) / 60).sort(dim=1, descending=True).values


# --------------------------------------------------------------------------- test 2 bis : N modes, tau libres
def forced_response(log_tau: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """(S, M) log tau (min), (S, n) U -> (S, n, M) réponses de premiers ordres de gain 1 (convolution FFT).

    Filtre x[k] = a x[k-1] + (1 - a) u[k-LAG], a = exp(-dt / tau), soit h[k] = (1 - a) a^(k-LAG), k >= LAG.
    """
    S, n = U.shape
    a = torch.exp(-DT_MIN / log_tau.exp())[..., None]                      # (S, M, 1)
    k = torch.arange(n, dtype=U.dtype)
    h = (1 - a) * a ** (k - LAG).clamp(min=0) * (k >= LAG)                  # (S, M, n)
    nfft = 2 * n
    y = torch.fft.irfft(torch.fft.rfft(h, nfft) * torch.fft.rfft(U, nfft)[:, None, :], nfft)[..., :n]
    return y.transpose(1, 2)


def design(log_tau: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Matrice de régression (S, n, 2 + 2M) : constante, M réponses libres, M réponses forcées, terme direct."""
    S, n = U.shape
    t = torch.arange(n, dtype=U.dtype)
    free = torch.exp(-t[None, :, None] / log_tau.exp()[:, None, :])
    return torch.cat([torch.ones(S, n, 1, dtype=U.dtype), free, forced_response(log_tau, U), U[..., None]], dim=2)


def solve_linear(log_tau: torch.Tensor, T: torch.Tensor, U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Coefficients linéaires exacts à tau fixés (colonnes mises à l'échelle). Renvoie (coefficients, résidus)."""
    X = design(log_tau, U)
    s = X.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    Xs = X / s
    A = Xs.transpose(1, 2) @ Xs + 1e-12 * torch.eye(X.shape[2], dtype=X.dtype)
    w = torch.linalg.solve(A, Xs.transpose(1, 2) @ T[..., None])[..., 0] / s[:, 0, :]
    return w, T - (X @ w[..., None])[..., 0]


INITS_MIN: dict[int, list[list[float]]] = {
    1: [[60.0], [600.0]],
    2: [[30.0, 600.0], [60.0, 3000.0]],
    3: [[20.0, 300.0, 3000.0], [60.0, 700.0, 2000.0]],
    4: [[10.0, 60.0, 600.0, 3000.0], [30.0, 200.0, 1000.0, 4000.0]],
    5: [[5.0, 15.0, 60.0, 600.0, 3000.0], [10.0, 35.0, 200.0, 1000.0, 4000.0]],
    6: [[3.0, 10.0, 35.0, 200.0, 1000.0, 4000.0], [5.0, 15.0, 60.0, 400.0, 1500.0, 6000.0]],
}
"""Initialisations des tau (minutes) ; le meilleur ajustement est retenu par simulation."""


def fit_modes(T: torch.Tensor, U: torch.Tensor, n_modes: int, iters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ajustement à ``n_modes`` modes, tau libres (Adam sur log tau). Renvoie (RMS, tau en h, coefficients)."""
    best_rms = torch.full((len(T),), float("inf"), dtype=T.dtype)
    best_lt = torch.zeros(len(T), n_modes, dtype=T.dtype)
    for init in INITS_MIN[n_modes]:
        log_tau = torch.log(torch.tensor(init, dtype=T.dtype)).repeat(len(T), 1).requires_grad_(True)
        opt = torch.optim.Adam([log_tau], lr=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for _ in range(iters):
            opt.zero_grad()
            solve_linear(log_tau.clamp(-2.0, 12.0), T, U)[1].pow(2).mean(1).sum().backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            lt = log_tau.clamp(-2.0, 12.0)
            rms = solve_linear(lt, T, U)[1].pow(2).mean(1).sqrt()
            better = rms < best_rms
            best_rms[better], best_lt[better] = rms[better], lt[better]
    with torch.no_grad():
        w, _ = solve_linear(best_lt, T, U)
    return best_rms, best_lt.exp() / 60, w


# --------------------------------------------------------------------------- analyse complète
def _q(x: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(x.double(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return {"p10": float(q[0]), "median": float(q[1]), "p90": float(q[2])}


def _hlc_error(hlc_hat: torch.Tensor, hlc: torch.Tensor) -> dict[str, float]:
    err = 100 * (hlc_hat - hlc).abs() / hlc
    return {"mape": float(err.mean()), **{f"err_{k}": v for k, v in _q(err).items()}}


def run(processed_dir: Path, batch: int, n: int, modes: list[int], iters: int, chunk: int = 100) -> dict:
    """Exécute les deux tests sur ``n`` simulations tirées du batch ``batch`` (graine fixe)."""
    b = CompactBatch.load(processed_dir / f"batch_{batch:02d}.pt")
    idx = torch.randperm(len(b.T), generator=torch.Generator().manual_seed(0))[:n]
    T, U, y = b.T[idx].double(), b.U_on[idx].double(), b.y[idx].double()
    heated = U.sum(1) > 0
    hlc = y[:, RAW_TARGETS.index("HLC")]
    out: dict = {"batch": batch, "n": n, "n_heated": int(heated.sum()),
                 "float32_resolution_C": float(torch.finfo(torch.float32).eps * T.abs().max()),
                 "arx_order": {}, "arx_identification": {}, "modes": {}}

    print(f"{n} simulations du batch {batch} ({int(heated.sum())} chauffées) ; "
          f"précision float32 de T ≈ {out['float32_resolution_C']:.1e} °C\n")
    print("Test 1 : erreur RMS à un pas de l'ARX (°C)")
    for order in ARX_ORDERS:
        coef, rms = arx_fit(T, U, order)
        out["arx_order"][order] = _q(rms)
        print(f"  ordre {order:2d} : médiane {out['arx_order'][order]['median']:.2e}"
              f" (p10 {out['arx_order'][order]['p10']:.1e}, p90 {out['arx_order'][order]['p90']:.1e})")

    print("\nTest 2 : identification directe depuis l'ARX (simulations chauffées)")
    for order in (2, 3, 4):
        coef, _ = arx_fit(T[heated], U[heated], order)
        hlc_hat = P_ON_W / arx_static_gain(coef, order)
        taus = arx_time_constants_h(coef, order)
        res = {"hlc": _hlc_error(hlc_hat, hlc[heated]),
               "tau_slowest_h": _q(taus[:, 0])}
        for j, t in enumerate(RAW_TARGETS):
            if t.startswith("tau"):
                res[f"corr_log_tau_slowest_{t}"] = float(torch.corrcoef(
                    torch.stack([taus[:, 0].log(), y[heated, j].log()]))[0, 1])
        out["arx_identification"][order] = res
        print(f"  ordre {order} : HLC MAPE {res['hlc']['mape']:.1f} % | tau le plus lent (médiane) "
              f"{res['tau_slowest_h']['median']:.2f} h | corr(log tau, log tau_b) {res['corr_log_tau_slowest_tau_b']:+.2f}")
    print("  cibles (médianes, h) : " + ", ".join(
        f"{t} {float(y[heated, j].median()):.1f}" for j, t in enumerate(RAW_TARGETS) if t.startswith("tau")))

    if modes:
        print("\nTest 2 bis : erreur de sortie à N modes, tau libres (simulations chauffées)")
    Th, Uh, hlch = T[heated], U[heated], hlc[heated]
    for m in modes:
        t0 = time.time()
        parts = [fit_modes(Th[i : i + chunk], Uh[i : i + chunk], m, iters) for i in range(0, len(Th), chunk)]
        rms = torch.cat([p[0] for p in parts])
        taus = torch.cat([p[1] for p in parts]).sort(dim=1, descending=True).values
        w = torch.cat([p[2] for p in parts])
        gain = w[:, 1 + m : 1 + 2 * m].sum(1) + w[:, -1]
        res = {"rms_C": _q(rms), "hlc": _hlc_error(P_ON_W / gain, hlch),
               "tau_median_h": [float(v) for v in taus.median(0).values], "seconds": round(time.time() - t0, 1)}
        out["modes"][m] = res
        print(f"  {m} mode(s) : RMS médian {res['rms_C']['median']:.2e} °C | HLC MAPE {res['hlc']['mape']:.1f} % | "
              f"tau médians (h) {[round(v, 2) for v in res['tau_median_h']]} | {res['seconds']} s")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--batch", type=int, default=1, help="batch d'entraînement (jamais 19-20)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--modes", type=int, nargs="*", default=[], help="ex. 2 3 4 (plus lent)")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "linearity.json")
    a = p.parse_args()
    if a.batch in (19, 20):
        raise SystemExit("les batches 19-20 sont réservés au test final")
    res = run(a.processed_dir, a.batch, a.n, a.modes, a.iters)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nrésultats : {a.out}")


if __name__ == "__main__":
    main()
