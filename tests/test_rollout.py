import pytest
import torch
from datetime import timedelta

aurora = None
Batch = None
Metadata = None
rollout = None


def make_dummy_batch(H=8, W=8, T=2, L=3):
    surf = {
        "2t": torch.randn(1, T, H, W),
        "10u": torch.randn(1, T, H, W),
        "10v": torch.randn(1, T, H, W),
        "msl": torch.randn(1, T, H, W),
    }
    static = {
        "lsm": torch.rand(H, W),
        "z": torch.randn(H, W),
        "slt": torch.rand(H, W),
    }
    atmos = {
        "z": torch.randn(1, T, L, H, W),
        "u": torch.randn(1, T, L, H, W),
        "v": torch.randn(1, T, L, H, W),
        "t": torch.randn(1, T, L, H, W),
        "q": torch.rand(1, T, L, H, W),
    }
    meta = Metadata(
        lat=torch.linspace(-2.0, -1.0, H),
        lon=torch.linspace(28.8, 30.9, W),
        time=tuple([torch.tensor(0), torch.tensor(1)]),
        atmos_levels=tuple([50, 100, 150]),
        rollout_step=0,
    )
    return Batch(surf_vars=surf, static_vars=static, atmos_vars=atmos, metadata=meta)


@pytest.mark.cpu
def test_rollout_two_steps():
    try:
        import aurora as _aurora
        from aurora import Batch as _Batch, Metadata as _Metadata, rollout as _rollout
    except Exception:
        pytest.skip("aurora dependencies not available")
    global aurora, Batch, Metadata, rollout
    aurora, Batch, Metadata, rollout = _aurora, _Batch, _Metadata, _rollout
    model = aurora.AuroraSmallPretrained(use_lora=False, timestep=timedelta(hours=6))
    batch = make_dummy_batch()
    steps = list(rollout(model, batch, steps=2))
    assert len(steps) == 2
    pred = steps[-1]
    assert pred.surf_vars["2t"].shape[1] == 1
    assert pred.atmos_vars["t"].shape[1] == 1
