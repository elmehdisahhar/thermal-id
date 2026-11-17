"""Tests du CNN : formes, champ réceptif, capacité à apprendre."""

import torch

from thermal_id.data import N_STEPS
from thermal_id.model import CNNConfig, ThermalCNN, receptive_field, stage_shapes


def test_output_shape() -> None:
    model = ThermalCNN()
    assert model(torch.randn(4, 2, N_STEPS)).shape == (4, 5)


def test_last_stage_sees_whole_window() -> None:
    first, last = receptive_field(ThermalCNN())
    assert first == 0 and last == N_STEPS - 1


def test_stage_lengths_halve() -> None:
    lengths = [s[-1] for _, s in stage_shapes(ThermalCNN())]
    assert lengths == [480, 240, 240, 120, 60, 30, 15, 8]


def test_can_overfit_tiny_batch() -> None:
    torch.manual_seed(0)
    model = ThermalCNN(CNNConfig(dropout=0.0))
    x, z = torch.randn(32, 2, N_STEPS), torch.randn(32, 5)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), z)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
