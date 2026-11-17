"""Tests du module de données, sur des batches synthétiques (pas besoin des vrais fichiers)."""

from pathlib import Path

import torch

from thermal_id.data import (
    N_STEPS,
    P_ON_W,
    CompactBatch,
    Normalizer,
    ThermalDataset,
    compact_batch,
    concat_batches,
    validate_batch,
)
from thermal_id.simul_io import ToSaveSimul

FIELDS = [
    "max_power_wat", "start_amplitude_wat", "end_amplitude_wat", "start_period_hours", "end_period_hours",
    "transition_speed", "phase_shift", "average_fluctuation_wat", "prbs_min_duration_hours",
    "prbs_max_duration_hours", "prbs_probability_on", "prbs_seed",
]


def fake_sim(n: int = 64, seed: int = 0) -> ToSaveSimul:
    g = torch.Generator().manual_seed(seed)
    hb = 50 + 150 * torch.rand(n, 2, generator=g)
    cdt = 5 + 100 * torch.rand(n, 3, generator=g)
    U = P_ON_W * (torch.rand(n, N_STEPS, generator=g) > 0.5).float()
    return ToSaveSimul(
        HLC=hb.sum(1),
        CDT=cdt,
        indoor_temperature=24.6 + 10 * torch.rand(n, N_STEPS, generator=g),
        indoor_U=U,
        solicitation=torch.rand(n, len(FIELDS), generator=g),
        solicitation_field_names=FIELDS,
        Hb=hb,
        Hb_labels=["HLC_mit", "HLC_ext"],
    )


def test_valid_batch_has_no_issue() -> None:
    assert validate_batch(fake_sim()) == []


def test_validation_detects_problems() -> None:
    sim = fake_sim()
    sim.HLC = sim.HLC + 1.0
    sim.indoor_U[0, 0] = 1234.0
    issues = validate_batch(sim)
    assert any("HLC" in m for m in issues)
    assert any("U prend" in m for m in issues)


def test_compact_roundtrip(tmp_path: Path) -> None:
    c = compact_batch(fake_sim(), batch_id=3)
    assert c.U_on.dtype == torch.uint8 and set(c.U_on.unique().tolist()) <= {0, 1}
    c.save(tmp_path / "b.pt")
    d = CompactBatch.load(tmp_path / "b.pt")
    assert d.batch_id == 3 and torch.equal(c.T, d.T) and torch.equal(c.y, d.y)


def test_normalizer_inverse_and_hlc_sum(tmp_path: Path) -> None:
    batches = [compact_batch(fake_sim(seed=s), batch_id=s) for s in range(3)]
    norm = Normalizer.fit(batches)
    data = concat_batches(batches)
    z = norm.targets(data.y)
    assert torch.allclose(z.mean(0), torch.zeros(5), atol=1e-4)
    assert torch.allclose(z.std(0, unbiased=False), torch.ones(5), atol=1e-3)
    back = norm.inverse(z)
    assert torch.allclose(back, data.y, rtol=1e-4)
    assert torch.allclose(back[:, 0], back[:, 1] + back[:, 2])
    norm.save(tmp_path / "n.json")
    assert Normalizer.load(tmp_path / "n.json") == norm


def test_dataset_item_shapes() -> None:
    c = compact_batch(fake_sim(), batch_id=1)
    ds = ThermalDataset(c, Normalizer.fit([c]))
    x, z = ds[0]
    assert x.shape == (2, N_STEPS) and z.shape == (5,)
    assert x.dtype == torch.float32
