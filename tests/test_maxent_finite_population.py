"""Unconditional exchange chains for finite-population designs."""

import numpy as np
import pytest

from graphical_sampling.maxent.finite_population import (
    design_probability_mass,
    run_design_exchanges,
)
from prescribed_sip.methods.package_gfs import initial_design


PI = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])


def make_design():
    coords = np.column_stack([np.arange(6, dtype=float), np.zeros(6)])
    return initial_design(PI, coords)


def fip_from_mass(mass):
    result = np.zeros(6)
    for sample, probability in mass.items():
        result[list(sample)] += probability
    return result


def test_probability_aggregation_is_read_only():
    design = make_design()
    before = [
        [(sample.prob, sample.ids) for sample in heap]
        for heap in design._heaps
    ]
    mass = design_probability_mass(design)
    after = [
        [(sample.prob, sample.ids) for sample in heap]
        for heap in design._heaps
    ]
    assert before == after
    assert sum(mass.values()) == pytest.approx(1.0)
    assert all(len(sample) == 3 for sample in mass)


def test_exchange_runner_preserves_fixed_size_mass_and_fips():
    result = run_design_exchanges(
        make_design(),
        PI,
        exchanges=25,
        seed=20260924,
        checkpoint_every=7,
    )
    mass = design_probability_mass(result.final_design)
    assert sum(mass.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(len(sample) == 3 for sample in mass)
    np.testing.assert_allclose(fip_from_mass(mass), PI, atol=1e-12, rtol=0.0)
    assert [row.exchange for row in result.trace] == [0, 7, 14, 21, 25]


def test_exchange_runner_replays_exactly():
    kwargs = dict(
        prescribed_fip=PI,
        exchanges=40,
        seed=20260924,
        checkpoint_every=9,
    )
    first = run_design_exchanges(make_design(), **kwargs)
    second = run_design_exchanges(make_design(), **kwargs)
    assert design_probability_mass(first.final_design) == pytest.approx(
        design_probability_mass(second.final_design), abs=0.0, rel=0.0
    )
    assert first.trace == second.trace


def test_every_requested_exchange_is_called_even_when_diagnostics_fall(monkeypatch):
    calls = 0
    design_class = type(make_design())
    original = design_class.exchange

    def counted_exchange(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(design_class, "exchange", counted_exchange)
    result = run_design_exchanges(
        make_design(), PI, exchanges=11, seed=9, checkpoint_every=3
    )
    assert calls == 11
    assert result.trace[-1].exchange == 11


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"exchanges": -1}, "nonnegative"),
        ({"checkpoint_every": 0}, "positive"),
        ({"partitions": 0}, "positive"),
        ({"exchange_coef": 0.0}, "exchange_coef"),
    ],
)
def test_exchange_runner_rejects_invalid_controls(overrides, message):
    settings = dict(exchanges=2, seed=1, checkpoint_every=1)
    settings.update(overrides)
    with pytest.raises(ValueError, match=message):
        run_design_exchanges(make_design(), PI, **settings)
