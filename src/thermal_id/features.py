"""Descripteurs de la réponse thermique, pour la baseline linéaire.

Chaque simulation (T_int, U) de 480 pas d'une minute est résumée en quelques nombres guidés par
la physique d'un modèle RC du premier ordre :

    C dT/dt = -H (T - T_ext) + P(t)

discrétisé au pas Δt = 1 min :  T[k+1] - T[k] = a (T[k] - T[0]) + b u[k] + c,
avec a = -H Δt / C, b = P_on Δt / C. On en tire deux estimations « physiques » par simulation :
    H ≈ -a P_on / b        (déperditions, W/K)
    tau ≈ -Δt / a          (constante de temps, min)
Ce modèle à une seule constante de temps est volontairement trop simple : il sert de descripteur,
pas de prédiction. Les autres descripteurs résument la forme de la courbe (montées, pentes).
"""

from __future__ import annotations

import torch

from thermal_id.data import P_ON_W

FEATURE_NAMES: tuple[str, ...] = (
    "frac_on",          # part du temps chauffage en marche
    "n_switch_on",      # nombre d'allumages
    "has_heat",         # 1 si au moins un pas chauffé
    "rise_60",          # T[60] - T[0]
    "rise_end",         # moyenne des 10 derniers pas - T[0]
    "rise_mean",        # moyenne de T - T[0]
    "rise_max",         # max de T - T[0]
    "drop_min",         # min de T - T[0]
    "t_std",            # écart-type de T
    "slope_on",         # pente moyenne (°C/min) quand le chauffage est en marche
    "slope_off",        # pente moyenne quand il est à l'arrêt
    "slope_last60",     # pente moyenne sur la dernière heure
    "rise_per_heat",    # rise_end / (frac_on + 0.05) : montée par quantité de chauffage
    "arx_a",            # coefficient a du modèle discret
    "arx_b",            # coefficient b
    "arx_c",            # coefficient c
    "log_tau_arx",      # log(-1/a) en minutes, borné
    "log_hlc_arx",      # log(-a P_on / b), borné, 0 si pas de chauffage
)

# Constantes de temps (min) de la base de modes pour la réponse indicielle, de l'air (minutes)
# aux parois (dizaines d'heures). Choix log-régulier ; le plus lent dépasse la fenêtre de 8 h.
MODE_TAUS_MIN: tuple[float, ...] = (5.0, 15.0, 45.0, 135.0, 400.0, 1200.0)
STEP_TIMES_MIN: tuple[int, ...] = (15, 60, 240, 480)

STEP_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"gain_u_tau{int(t)}" for t in MODE_TAUS_MIN)        # contribution de chaque mode à la réponse à U
    + tuple(f"free_tau{int(t)}" for t in MODE_TAUS_MIN)        # réponse libre depuis l'état initial
    + tuple(f"step_{t}min" for t in STEP_TIMES_MIN)            # réponse indicielle reconstruite (°C)
    + ("step_gain_total", "fit_rmse")
)
ALL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + STEP_FEATURE_NAMES


def response_features(T: torch.Tensor, U_on: torch.Tensor) -> torch.Tensor:
    """(N, 480) float, (N, 480) 0/1 -> (N, len(FEATURE_NAMES)) float32. Calcul vectorisé en float64."""
    t = T.to(torch.float64)
    u = U_on.to(torch.float64)
    t0 = t[:, :1]
    x = t - t0                                # écart à la température initiale
    dt = x[:, 1:] - x[:, :-1]                 # variation par pas (°C/min)
    u_k = u[:, :-1]

    frac_on = u.mean(1)
    n_switch_on = ((u[:, 1:] - u[:, :-1]) > 0).sum(1).to(torch.float64) + u[:, 0]
    has_heat = (frac_on > 0).to(torch.float64)

    n_on = u_k.sum(1)
    n_off = (1 - u_k).sum(1)
    slope_on = (dt * u_k).sum(1) / n_on.clamp(min=1)
    slope_off = (dt * (1 - u_k)).sum(1) / n_off.clamp(min=1)

    rise_end = x[:, -10:].mean(1)
    a, b, c = _fit_arx(x, u)
    tau = (-1.0 / a).where(a < 0, torch.full_like(a, 1e5))
    log_tau = tau.clamp(1.0, 1e5).log()
    hlc = (-a * P_ON_W / b).where((b > 1e-9) & (a < 0), torch.ones_like(a))
    log_hlc = hlc.clamp(1.0, 1e4).log() * has_heat

    feats = torch.stack(
        [
            frac_on,
            n_switch_on,
            has_heat,
            x[:, 60],
            rise_end,
            x.mean(1),
            x.max(1).values,
            x.min(1).values,
            t.std(1),
            slope_on,
            slope_off,
            (x[:, -1] - x[:, -61]) / 60.0,
            rise_end / (frac_on + 0.05),
            a,
            b,
            c,
            log_tau,
            log_hlc,
        ],
        dim=1,
    )
    return feats.to(torch.float32)


def step_response_features(T: torch.Tensor, U_on: torch.Tensor, chunk: int = 5000) -> torch.Tensor:
    """Identification linéaire par simulation, indépendante du scénario de chauffage.

    Modèle (moindres carrés par simulation, p = 12 paramètres) :
        T[k] - T[0] = sum_m g_m * f_m[k]  +  sum_m r_m * (1 - exp(-k / tau_m))
    où f_m = u filtré par un passe-bas du premier ordre de constante tau_m (gain statique 1).
    g_m est la part du mode m dans la réponse à un échelon de chauffage (°C pour P_on) : c'est une
    propriété du bâtiment, pas du signal U. r_m décrit la réponse libre depuis l'état initial.

    Renvoie (N, len(STEP_FEATURE_NAMES)) float32. Les simulations sans chauffage ont g = 0.
    """
    out = [
        _step_chunk(T[i : i + chunk], U_on[i : i + chunk]) for i in range(0, T.shape[0], chunk)
    ]
    return torch.cat(out).to(torch.float32)


def _step_chunk(T: torch.Tensor, U_on: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    t = T.to(torch.float64)
    u = U_on.to(torch.float64)
    x = t - t[:, :1]
    n, n_steps = x.shape
    taus = torch.tensor(MODE_TAUS_MIN, dtype=torch.float64)
    alpha = torch.exp(-1.0 / taus)                                       # (M,)

    # u filtré par chaque mode : f[k] = alpha f[k-1] + (1 - alpha) u[k-1]  (effet au pas suivant)
    filt = torch.zeros(n, n_steps, len(taus), dtype=torch.float64)
    for k in range(1, n_steps):
        filt[:, k] = alpha * filt[:, k - 1] + (1 - alpha) * u[:, k - 1, None]
    k = torch.arange(n_steps, dtype=torch.float64)
    free = (1 - torch.exp(-k[:, None] / taus)).expand(n, -1, -1)         # (n, 480, M)

    A = torch.cat([filt, free], dim=2)                                   # (n, 480, 2M)
    ata = A.transpose(1, 2) @ A
    aty = (A * x.unsqueeze(2)).sum(1)
    ata = ata + ridge * n_steps * torch.eye(A.shape[2], dtype=torch.float64)
    coef = torch.linalg.solve(ata, aty)                                  # (n, 2M)
    resid = x - (A @ coef.unsqueeze(2)).squeeze(2)

    g = coef[:, : len(taus)]
    times = torch.tensor(STEP_TIMES_MIN, dtype=torch.float64)
    step = g @ (1 - torch.exp(-times[None, :] / taus[:, None]))          # (n, len(STEP_TIMES_MIN))
    return torch.cat(
        [coef, step, g.sum(1, keepdim=True), resid.pow(2).mean(1, keepdim=True).sqrt()], dim=1
    )


def all_features(T: torch.Tensor, U_on: torch.Tensor) -> torch.Tensor:
    """Descripteurs de forme + descripteurs d'identification : (N, len(ALL_FEATURE_NAMES))."""
    return torch.cat([response_features(T, U_on), step_response_features(T, U_on)], dim=1)


def _fit_arx(x: torch.Tensor, u: torch.Tensor, ridge: float = 1e-8) -> tuple[torch.Tensor, ...]:
    """Moindres carrés par simulation de dx[k] = a x[k] + b u[k] + c (équations normales 3x3 en lot).

    Le petit terme ridge rend le système inversible quand u est identiquement nul (pas de chauffage) :
    b vaut alors 0.
    """
    y = x[:, 1:] - x[:, :-1]
    reg = torch.stack([x[:, :-1], u[:, :-1], torch.ones_like(y)], dim=2)   # (N, 479, 3)
    xtx = reg.transpose(1, 2) @ reg                                        # (N, 3, 3)
    xty = (reg * y.unsqueeze(2)).sum(1)                                    # (N, 3)
    xtx = xtx + ridge * torch.eye(3, dtype=x.dtype)
    coef = torch.linalg.solve(xtx, xty)
    return coef[:, 0], coef[:, 1], coef[:, 2]
