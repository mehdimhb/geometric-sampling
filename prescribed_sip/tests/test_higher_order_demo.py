"""Focused behavioral checks for the N=12 higher-order notebook prototype."""
import json
from pathlib import Path

import numpy as np


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "GFS_variance_of_variance_demo_N12.ipynb"
)


def notebook_core():
    """Execute only the notebook's settings and reusable core helpers."""
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {"__name__": "__notebook_test__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "core" in cell.get("metadata", {}).get("tags", []):
            break
    return namespace


def test_support_uses_strict_fixed_tolerance_and_fraction_of_all_samples():
    """Catches counting p==1e-12 or sub-tolerance LP noise as support."""
    ns = notebook_core()
    p = np.array([0.5, 0.5 - 2.3e-12, 2e-13, 1e-12, 1.1e-12])
    size, fraction = ns["support_diagnostics"](p, 5)
    assert size == 3
    assert fraction == 0.6


def test_enumerated_moments_match_hand_checked_design():
    """Catches incorrect first/second-order constraint rows or sample indexing."""
    ns = notebook_core()
    samples, membership, pairs, pair_membership, A = ns["enumerate_problem"](4, 2)
    p = np.zeros(len(samples))
    sample_index = {tuple(sample): i for i, sample in enumerate(samples)}
    p[sample_index[(0, 1)]] = 0.25
    p[sample_index[(2, 3)]] = 0.75
    pi, Pi = ns["inclusion_moments"](p, membership, pairs, pair_membership)
    np.testing.assert_array_equal(pi, [0.25, 0.25, 0.75, 0.75])
    np.testing.assert_array_equal(
        Pi,
        [[0.25, 0.25, 0, 0], [0.25, 0.25, 0, 0],
         [0, 0, 0.75, 0.75], [0, 0, 0.75, 0.75]],
    )
    np.testing.assert_allclose(A @ p, np.r_[1.0, pi, [Pi[i, j] for i, j in pairs]])


def test_block_optimizer_records_support_and_preserves_all_constraints():
    """Catches accepted LP steps that omit diagnostics or drift from A p = b."""
    ns = notebook_core()
    samples, membership, pairs, pair_membership, A = ns["enumerate_problem"](6, 3)
    p0 = np.full(len(samples), 1 / len(samples))
    costs = np.linspace(0.0, 1.0, len(samples)) ** 2
    result = ns["block_optimize"](
        p0,
        costs,
        A,
        membership,
        pairs,
        pair_membership,
        max_iter=4,
        stagnation=2,
        block_size=len(samples),
        seed=7,
        validation_tol=1e-10,
        v_values=np.arange(len(samples), dtype=float),
        v_ht=float(np.mean(np.arange(len(samples), dtype=float))),
    )
    trace = result["convergence"]
    required = {
        "iteration", "accepted_iteration", "J", "RelativeVarV", "CV_V",
        "max_FIP_drift", "SIP_RMSE_drift", "max_SIP_drift",
        "support_size", "support_fraction", "entropy", "runtime",
    }
    assert required.issubset(trace.columns)
    assert len(trace) >= 2 and trace.iloc[-1]["J"] < trace.iloc[0]["J"]
    np.testing.assert_allclose(A @ result["probabilities"], A @ p0, atol=1e-10)
    assert trace["max_FIP_drift"].max() < 1e-10
    assert trace["max_SIP_drift"].max() < 1e-10
    expected_support = np.count_nonzero(result["probabilities"] > 1e-12)
    assert trace.iloc[-1]["support_size"] == expected_support
    assert trace.iloc[-1]["support_fraction"] == expected_support / len(samples)
    assert result["min_support_size"] == trace["support_size"].min()
    assert result["max_support_size"] == trace["support_size"].max()
