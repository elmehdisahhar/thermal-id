"""Les descripteurs d'identification retrouvent-ils les paramètres d'un système connu ?"""

import math

import torch

from thermal_id.baseline import LinearBaseline
from thermal_id.data import N_STEPS, P_ON_W
from thermal_id.features import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES,
    STEP_FEATURE_NAMES,
    all_features,
)


def first_order(gain: float, tau_min: float, n: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Système du 1er ordre discret, chauffage aléatoire : T[k+1] = T0 + a x[k] + (1 - a) gain u[k]."""
    g = torch.Generator().manual_seed(seed)
    u = (torch.rand(n, N_STEPS // 20, generator=g) > 0.5).repeat_interleave(20, dim=1).to(torch.uint8)
    alpha = math.exp(-1 / tau_min)
    x = torch.zeros(n, N_STEPS, dtype=torch.float64)
    for k in range(1, N_STEPS):
        x[:, k] = alpha * x[:, k - 1] + (1 - alpha) * gain * u[:, k - 1]
    return (24.624 + x).float(), u


def test_shapes() -> None:
    T, u = first_order(10.0, 45.0)
    F = all_features(T, u)
    assert F.shape == (8, len(ALL_FEATURE_NAMES))
    assert torch.isfinite(F).all()


def test_identification_recovers_known_system() -> None:
    gain, tau = 10.0, 45.0                     # 10 °C par 7000 W => H = 700 W/K
    T, u = first_order(gain, tau)
    F = all_features(T, u).double()
    col = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
    # modèle ARX : constante de temps et H exacts
    assert torch.allclose(F[:, col["log_tau_arx"]].exp(), torch.tensor(tau, dtype=torch.float64), rtol=2e-2)
    hlc = F[:, col["log_hlc_arx"]].exp()
    assert torch.allclose(hlc, torch.tensor(P_ON_W / gain, dtype=torch.float64), rtol=2e-2)
    # base de modes : gain total retrouvé à quelques % près. Il n'est pas exact car la réponse
    # moyenne à U ressemble à une réponse libre (1 - exp) : la base libre en absorbe une partie.
    # C'est la même confusion qui rend les modes lents mal identifiables sur 8 h.
    assert torch.allclose(F[:, col["step_gain_total"]], torch.tensor(gain, dtype=torch.float64), rtol=8e-2)
    assert (F[:, col["fit_rmse"]] < 1e-2).all()


def test_no_heating_is_handled() -> None:
    T, u = first_order(10.0, 45.0)
    u.zero_()
    F = all_features(T + 0.01 * torch.arange(N_STEPS), u)
    assert torch.isfinite(F).all()
    assert (F[:, FEATURE_NAMES.index("log_hlc_arx")] == 0).all()
    assert (F[:, len(FEATURE_NAMES) + STEP_FEATURE_NAMES.index("step_gain_total")].abs() < 1e-6).all()


def test_linear_baseline_recovers_linear_relation() -> None:
    g = torch.Generator().manual_seed(1)
    X = torch.randn(2000, 4, generator=g)
    W_true = torch.randn(5, 5, generator=g, dtype=torch.float64)
    Z = (torch.cat([torch.ones(2000, 1), X], 1).double() @ W_true).float()
    model = LinearBaseline.fit([(X[:1000], Z[:1000]), (X[1000:], Z[1000:])], X.mean(0), X.std(0), ridge=0.0)
    assert torch.allclose(model.predict(X), Z, atol=1e-3)
