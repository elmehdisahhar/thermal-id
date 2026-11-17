"""CNN 1D résiduel : (T_int, U) sur 480 pas -> 5 cibles normalisées.

Architecture :
    entrée (B, 2, 480)
    -> stem : Conv1d k=7, pas 2                         (B, w0, 240)
    -> 6 étages résiduels, chacun divise la longueur par 2 (sauf le premier)
                                                         (B, w5, 8)
    -> pooling global (moyenne et max sur le temps)      (B, 2 * w5)
    -> tête MLP                                          (B, 5)

Choix :
- Conv1d : un filtre qui glisse sur le temps est l'analogue d'une réponse impulsionnelle ;
  empiler des filtres permet d'apprendre la réponse du bâtiment à U.
- Sous-échantillonnage progressif (pas 2) : réduit le coût sur CPU et agrandit le champ réceptif ;
  le dernier étage « voit » toute la fenêtre de 480 min (vérifié par ``receptive_field``).
- Connexions résiduelles + GroupNorm : entraînement stable d'un réseau profond. La GroupNorm se comporte
  de la même façon en entraînement et en évaluation ; la BatchNorm, essayée d'abord (``norm="batch"``),
  rendait la perte d'évaluation instable.
- Pooling global : la sortie ne dépend pas de la position d'un motif dans le temps,
  seulement de sa présence et de son intensité.
- Sorties normalisées (HLC_mit, HLC_ext, log tau_b, log tau_n, log tau_inf) ; HLC est obtenu par
  somme au moment de la dénormalisation (``Normalizer.inverse``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from thermal_id.data import MODEL_TARGETS, N_STEPS


@dataclass(frozen=True)
class CNNConfig:
    in_channels: int = 2
    stem_width: int = 32
    widths: tuple[int, ...] = (32, 32, 64, 64, 128, 128)   # un étage résiduel par largeur
    kernel_size: int = 5
    head_hidden: int = 128
    dropout: float = 0.1
    n_outputs: int = len(MODEL_TARGETS)
    norm: str = "group"   # "group" (GroupNorm, identique en entraînement et en évaluation) ou "batch"
    n_extra: int = 0      # descripteurs scalaires ajoutés avant la tête (variante hybride) ; 0 = CNN seul


def make_norm(kind: str, channels: int) -> nn.Module:
    """Couche de normalisation. GroupNorm ne dépend pas des statistiques du lot : pas d'écart train / éval."""
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"normalisation inconnue : {kind}")


class ResBlock1d(nn.Module):
    """Deux convolutions (conv -> norm -> GELU) avec un raccourci ; ``stride=2`` divise la longueur par 2."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, norm: str = "group") -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = make_norm(norm, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, padding=pad, bias=False)
        self.bn2 = make_norm(norm, c_out)
        self.act = nn.GELU()
        if c_in == c_out and stride == 1:
            self.shortcut: nn.Module = nn.Identity()
        else:  # adapter la forme du raccourci (largeur et longueur)
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), make_norm(norm, c_out)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class ThermalCNN(nn.Module):
    def __init__(self, config: CNNConfig = CNNConfig()) -> None:
        super().__init__()
        self.config = config
        c = config
        self.stem = nn.Sequential(
            nn.Conv1d(c.in_channels, c.stem_width, 7, stride=2, padding=3, bias=False),
            make_norm(c.norm, c.stem_width),
            nn.GELU(),
        )
        blocks = []
        c_in = c.stem_width
        for i, w in enumerate(c.widths):
            blocks.append(ResBlock1d(c_in, w, c.kernel_size, stride=1 if i == 0 else 2, norm=c.norm))
            c_in = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(2 * c_in + c.n_extra, c.head_hidden),  # (n_extra descripteurs concaténés au pooling)
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.head_hidden, c.n_outputs),
        )
        if c.n_extra:
            # standardisation des descripteurs, stockée dans le modèle (sauvegardée avec les poids)
            self.register_buffer("extra_mean", torch.zeros(c.n_extra))
            self.register_buffer("extra_std", torch.ones(c.n_extra))

    def set_extra_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Fixe la standardisation des descripteurs (statistiques du train)."""
        self.extra_mean.copy_(mean)
        self.extra_std.copy_(std.clamp(min=1e-6))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, 480) -> carte de caractéristiques (B, C, L) avant le pooling."""
        return self.blocks(self.stem(x))

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """(B, 2, 480) [+ (B, n_extra) descripteurs bruts] -> (B, 5) cibles normalisées."""
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        if self.config.n_extra:
            if extra is None:
                raise ValueError(f"ce modèle attend {self.config.n_extra} descripteurs supplémentaires")
            z = (extra.float() - self.extra_mean) / self.extra_std
            pooled = torch.cat([pooled, z.to(pooled.dtype)], dim=1)
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def stage_shapes(model: ThermalCNN, n_steps: int = N_STEPS) -> list[tuple[str, tuple[int, ...]]]:
    """Forme de la sortie du stem et de chaque étage, pour un exemple."""
    model.eval()
    x = torch.zeros(1, model.config.in_channels, n_steps)
    out = [("entrée", tuple(x.shape))]
    h = model.stem(x)
    out.append(("stem", tuple(h.shape)))
    for i, block in enumerate(model.blocks):
        h = block(h)
        out.append((f"étage {i}", tuple(h.shape)))
    return out


def receptive_field(model: ThermalCNN, n_steps: int = N_STEPS) -> tuple[int, int]:
    """Champ réceptif mesuré : pas de temps d'entrée qui influencent une position centrale du dernier étage.

    Méthode empirique (gradient de la sortie par rapport à l'entrée), indépendante du calcul théorique.
    Renvoie (premier pas, dernier pas) influents.
    """
    was_training = model.training
    model.eval()
    x = torch.randn(1, model.config.in_channels, n_steps, requires_grad=True)
    h = model.features(x)
    h[0, :, h.shape[2] // 2].sum().backward()
    influent = (x.grad[0].abs().sum(0) > 0).nonzero().squeeze(1)
    model.train(was_training)
    return int(influent.min()), int(influent.max())
