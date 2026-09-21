"""Contract checks for the standalone MU284 final-study launcher."""
from pathlib import Path


def test_mu284_final_config_has_the_prescribed_full_budget():
    from prescribed_sip.mu284_final import study_config

    config = study_config()
    assert (config["N"], config["n"], config["sip_pairs"]) == (281, 15, 39340)
    assert config["target_checkpoints"] == [1000, 5000, 10000, 25000, 50000, 100000]
    assert config["directed_checkpoints"] == [10000, 25000, 50000, 100000, 250000,
                                                500000, 1000000, 1500000, 2000000]
    assert config["directed_evaluations"] == 2_000_000
    assert config["directed_batch"] == 100
    assert config["directed_repetitions"] == 10
    assert config["methods"] == ["Initial GFS baseline", "Original GFS + restarts",
                                 "GFS-Directed", "DSD — projection/order fit"]


def test_mu284_loader_preserves_cleaned_csv_order_and_uses_p75():
    import numpy as np
    from prescribed_sip.mu284_final import load_mu284

    repo = Path(__file__).resolve().parents[2]
    unit_ids, sizes, pi = load_mu284(repo)
    assert len(unit_ids) == len(sizes) == len(pi) == 281
    assert len(set(unit_ids.tolist())) == 281
    assert np.all(sizes > 0)
    assert np.all((0 < pi) & (pi < 1))
    np.testing.assert_allclose(pi.sum(), 15, atol=1e-10)
