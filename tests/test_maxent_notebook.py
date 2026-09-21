"""Reduced execution contract for the graphical MaxEnt die notebook."""
from pathlib import Path

import nbformat
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "simulations/jupyters/17_graphical_maxent_die.ipynb"


def test_wide_two_part_exchange_moves_six_percent_and_shades_both_regions():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__main__"}

    for index in (23, 24, 25):
        cell = notebook.cells[index]
        exec(compile(cell.source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    density_reference = namespace["density_curve"](namespace["q_uniform"])
    np.testing.assert_allclose(density_reference, 0.1, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        namespace["distribution_moments"](namespace["q_uniform"]),
        (5.0, 25.0 / 3.0),
        rtol=0.0,
        atol=1e-12,
    )

    donor_bins = np.arange(10, 20)
    receiver_bins = np.arange(65, 75)
    unchanged_bins = np.setdiff1d(np.arange(100), np.r_[donor_bins, receiver_bins])
    q_uniform = namespace["q_uniform"]
    q_two_part = namespace["q_two_part"]
    np.testing.assert_allclose(
        q_uniform[donor_bins] - q_two_part[donor_bins],
        0.006,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        q_two_part[receiver_bins] - q_uniform[receiver_bins],
        0.006,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_array_equal(q_two_part[unchanged_bins], q_uniform[unchanged_bins])
    assert q_two_part[donor_bins].sum() == pytest.approx(0.04, abs=1e-14)
    assert q_two_part[receiver_bins].sum() == pytest.approx(0.16, abs=1e-14)
    assert q_two_part.sum() == pytest.approx(1.0, abs=1e-14)
    assert all(len(axis.patches) >= 2 for axis in namespace["axes"])


def test_mean_only_microscopic_start_approaches_exact_target_without_entropy():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__main__"}

    for index in (23, 24, 28):
        cell = notebook.cells[index]
        exec(compile(cell.source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    atom_count = int(namespace["D_EFF"])
    assert namespace["starting_signature_mean"] == 100 * atom_count
    assert namespace["target_signature_mean"] == 40 * atom_count
    assert namespace["current_signature"] == namespace["target_signature_mean"]

    history = np.asarray(namespace["approach_history"])
    np.testing.assert_allclose(history[[0, -1]], [5.0, 2.0], rtol=0.0, atol=1e-12)
    assert np.all(np.diff(np.abs(history - 2.0)) < 0.0)
    assert namespace["accepted_approach"] == len(history) - 1
    assert namespace["rejected_approach"] >= 0

    initial_counts_mean = namespace["initial_counts_mean"]
    assert initial_counts_mean.dtype == np.int64
    assert initial_counts_mean.sum() == atom_count
    assert np.all(initial_counts_mean >= 0)
    np.testing.assert_array_equal(
        initial_counts_mean @ namespace["mean_signature"],
        namespace["target_signature_mean"],
    )

    mean0, _ = namespace["distribution_moments"](namespace["q0_mean"])
    np.testing.assert_allclose(
        mean0,
        namespace["MU_MEAN_ONLY"],
        rtol=0.0,
        atol=1e-12,
    )
    assert "entropy" not in namespace["approach_acceptance_rule"].lower()


def test_notebook_executes_reduced_study_and_exposes_expected_results(tmp_path):
    assert NOTEBOOK.exists(), "notebook 17 has not been created"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameter_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameter_cells) == 1

    reduced_settings = {
        "write_outputs": False,
        "output_dir": str(tmp_path),
        "replay_iterations": 80,
        "experiments": {
            "unconstrained": {
                "D": 4_000,
                "iterations": 60_000,
                "burn_in": 30_000,
                "checkpoint_every": 3_000,
                "seed": 20_260_921,
            },
            "mean": {
                "D": 2_000,
                "iterations": 120_000,
                "burn_in": 60_000,
                "checkpoint_every": 6_000,
                "seed": 20_260_922,
            },
            "two_moment": {
                "D": 4_000,
                "iterations": 180_000,
                "burn_in": 90_000,
                "checkpoint_every": 9_000,
                "seed": 20_260_923,
            },
        },
        "finite_population": {
            "exchanges": 200,
            "seed": 20_260_924,
            "checkpoint_every": 20,
            "partitions": 1,
            "pull_strategy": "default",
            "exchange_coef": 0.75,
        },
        "continuous": {
            "D": 6_000,
            "iterations": 4_000,
            "burn_in": 2_000,
            "checkpoint_every": 1_000,
            "replicates": 3,
            "seed": 20_260_925,
        },
    }

    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        exec(compile(cell.source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
        if "parameters" in cell.metadata.get("tags", []):
            namespace["SETTINGS"] = reduced_settings

    results = namespace["results"]
    assert results["summary"]["Experiment"].tolist() == [
        "Unconstrained",
        "Mean = 4",
        "Mean = 4, Var = 1.8475",
    ]
    assert "Final exchange entropy" in results["summary"].columns
    assert "Theoretical MaxEnt entropy" in results["summary"].columns
    assert "MaxEnt entropy" not in results["summary"].columns
    assert (results["summary"]["L1 distance"] < 0.05).all()
    assert all(
        np.array_equal(first, second)
        for first, second in results["replay_pairs"]
    )
    assert results["old_variance"] == pytest.approx(2.242, abs=1e-12)
    assert results["d10000_infeasible"] is True

    assert "continuous" in results
    continuous = results["continuous"]
    assert continuous["mechanism"] == "microscopic signature-preserving atom exchanges"
    assert continuous["entropy_used_in_transition"] is False
    assert continuous["replicates"] == 3
    np.testing.assert_allclose(
        continuous["uniform_reference_moments"],
        (5.0, 25.0 / 3.0),
        rtol=0.0,
        atol=1e-12,
    )
    for case in ("mean", "mean_variance"):
        diagnostic = continuous[case]
        assert diagnostic["proposal_acceptance"] == 1.0
        assert diagnostic["max_constraint_error"] == 0.0
        assert diagnostic["endpoint_cdfs"].shape[0] == 3
        assert np.all(diagnostic["cdf_band_lower"] <= diagnostic["cdf_band_upper"])
        assert diagnostic["final_cdf_sup_error"] > 0.0

    for experiment in namespace["study"].values():
        chain = experiment["chain"]
        assert chain.initial_atoms is not None
        assert chain.final_atoms is not None
        outcomes = experiment["constraints"].outcomes
        derived = np.array(
            [np.count_nonzero(chain.final_atoms == value) for value in outcomes]
        )
        np.testing.assert_array_equal(derived, chain.final_counts)

    finite = results["finite_population"]
    assert finite["N"] == 6
    assert finite["n"] == 3
    np.testing.assert_array_equal(
        finite["prescribed_fip"],
        [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    )
    assert finite["number_of_samples"] == 20
    assert finite["exchange_count"] == 200
    assert finite["initial_support_size"] == 4
    assert finite["exchanged_support_size"] <= 20
    assert finite["theoretical_support_size"] == 20
    assert finite["exchanged_probability_sum_error"] < 1e-12
    assert finite["exchanged_fip_error"] < 1e-12
    assert finite["theoretical_probability_sum_error"] < 1e-12
    assert finite["theoretical_fip_error"] < 1e-10
    assert finite["theoretical_entropy"] > finite["initial_entropy"]
    assert finite["theoretical_entropy"] <= np.log(20) + 1e-12
    assert namespace["finite_exchange_trace"]["exchange"].iloc[-1] == 200

    initial_mass = namespace["design_probability_mass"](
        namespace["finite_gfs_design"]
    )
    exchanged_mass = namespace["design_probability_mass"](
        namespace["finite_exchanged_design"]
    )
    assert len(initial_mass) == 4
    assert len(exchanged_mass) <= 20
    np.testing.assert_allclose(
        namespace["fip_from_probability_mass"](initial_mass, 6),
        finite["prescribed_fip"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        namespace["fip_from_probability_mass"](exchanged_mass, 6),
        finite["prescribed_fip"],
        atol=1e-12,
    )

    # A white, grid-free plotting style applies to every notebook-17 figure.
    assert namespace["plt"].rcParams["axes.grid"] is False
    assert not any(tmp_path.iterdir())
