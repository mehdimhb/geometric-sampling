"""Regression checks for DI matching, with reference clustering held fixed."""

import ast
import importlib
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


REFERENCES = np.array([[4., -1.], [-3., 2.], [3., -2.]])
SAMPLE = np.array([[-1., -4.], [4., 4.], [2., 1.]])
ROOT = Path(__file__).resolve().parents[1]


def optimal_assignments(references, sample):
    # Enumerate independently of the Hungarian solver for these small examples.
    costs = np.sum((references[:, None] - sample[None, :]) ** 2, axis=2)
    totals = {
        assignment: costs[np.arange(len(sample)), assignment].sum()
        for assignment in permutations(range(len(sample)))
    }
    minimum = min(totals.values())
    return minimum, {p for p, cost in totals.items() if cost == minimum}


@pytest.fixture(params=["nmeans", "nmedoids", "stored-density"])
def match_with_fixed_references(request, monkeypatch):
    """Run the real DI pipeline while controlling only reference construction."""
    stored = request.param == "stored-density"
    module = importlib.import_module(
        "graphical_sampling.index.store.density" if stored
        else "graphical_sampling.index.density_disparity"
    )

    def run(references, sample):
        n = len(sample)
        coords = np.vstack([sample, references])
        probabilities = np.full(2 * n, .5)
        population = SimpleNamespace(
            coords=coords, inclusions=probabilities, probs=probabilities,
            N=len(coords), n=n,
        )
        labels = np.tile(np.arange(n), 2)
        # Distinct centroid values ensure the medoid branch uses its own refs.
        centroids = references + 10 if request.param == "nmedoids" else references

        class FixedClustering:
            def __init__(self, **kwargs):
                self.labels = labels
                self.centroids = centroids

            def fit(self, pop, init_centroids):
                np.testing.assert_array_equal(init_centroids, sample)

        monkeypatch.setattr(module, "FIPBalancedNMeans", FixedClustering)
        if stored:
            scorer = module.Density(population, n=n, n_jobs=1)
        else:
            scorer = module.DensityDisparity(
                population, representative=request.param, n_jobs=1,
            )
            if request.param == "nmedoids":
                monkeypatch.setattr(
                    scorer, "_cluster_medoids",
                    lambda labels, centroids: (references, np.arange(n, 2 * n)),
                )

        assignments = []
        translated = []

        def solve(cost):
            rows, cols = linear_sum_assignment(cost)
            assignments.append(tuple(cols))
            return rows, cols

        real_density = scorer._density

        def density(points):
            translated.append(points.copy())
            return real_density(points)

        monkeypatch.setattr(module, "linear_sum_assignment", solve)
        monkeypatch.setattr(scorer, "_density", density)
        scores, densities = scorer.score(np.arange(n)[None, :], return_densities=True)
        assert len(assignments) == 1
        assignment = assignments[0]
        expected = coords + (sample[list(assignment)] - references)[labels]
        np.testing.assert_array_equal(translated[0], expected)
        np.testing.assert_allclose(densities[0][1], real_density(expected))
        assert scores.shape == (1,) and np.isfinite(scores).all()
        assert densities[0][0].shape == densities[0][1].shape == (2 * n,)
        return assignment

    return run


def test_di_matching_minimizes_squared_distances(match_with_fixed_references):
    euclidean = tuple(linear_sum_assignment(cdist(REFERENCES, SAMPLE))[1])
    minimum, optimal = optimal_assignments(REFERENCES, SAMPLE)
    assert euclidean == (1, 0, 2)
    assert minimum == 71 and optimal == {(1, 2, 0)}
    assert euclidean not in optimal
    assignment = match_with_fixed_references(REFERENCES, SAMPLE)
    assert assignment in optimal


def test_fixed_reference_matching_preserves_unique_optimum_under_sample_translation(
    match_with_fixed_references,
):
    original = match_with_fixed_references(REFERENCES, SAMPLE)
    for shift in (np.array([7., -3.]), np.array([-5., 6.])):
        shifted = SAMPLE + shift
        _, optimal = optimal_assignments(REFERENCES, shifted)
        assert optimal == {(1, 2, 0)}
        assignment = match_with_fixed_references(REFERENCES, shifted)
        assert assignment == original and assignment in optimal


def test_fixed_reference_translation_preserves_tied_optima(
    match_with_fixed_references,
):
    references = np.array([[0., 0.], [2., 0.], [1., 3.]])
    sample = np.array([[1., -1.], [1., 1.], [1., 4.]])
    _, original_optimal = optimal_assignments(references, sample)
    assert len(original_optimal) == 2
    for shift in (np.zeros(2), np.array([7., -3.]), np.array([-5., 6.])):
        shifted = sample + shift
        _, optimal = optimal_assignments(references, shifted)
        assert optimal == original_optimal
        assignment = match_with_fixed_references(references, shifted)
        assert assignment in optimal
        # Identical inputs replay consistently; across translations either tied
        # optimum is valid, so do not require a particular solver tie ordering.
        assert match_with_fixed_references(references, shifted) == assignment


@pytest.mark.parametrize("filename,function_name", [
    ("temporary_2D_DI_relative_rotation.py", "calculate_di"),
    ("temporary_2D_OT_DI_four_configurations.py", "evaluate"),
    ("temporary_3D_DI_translation_rotation.py", "evaluate"),
    ("temporary_compare_ipnm_initializations.py", "score_with_reference"),
    ("temporary_measure_DI_OT_robustness.py", "one_score"),
    ("temporary_random2D_DI_two_six_panel_figures.py", "evaluate"),
])
def test_simulation_di_matching_minimizes_squared_distances(filename, function_name):
    # These scripts execute full studies on import. Execute their actual
    # matching assignment alone, without running studies or rewriting results.
    path = ROOT / "simulations" / filename
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    assignment = next(
        node for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "linear_sum_assignment"
            for child in ast.walk(node)
        )
    )
    namespace = dict(
        linear_sum_assignment=linear_sum_assignment, cdist=cdist,
        medoids=REFERENCES, refs=REFERENCES, med=REFERENCES,
        sample=SAMPLE, raw_sample=SAMPLE, raw=SAMPLE,
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), namespace)
    rows, cols = (namespace[name.id] for name in assignment.targets[0].elts)
    assert tuple(rows) == (0, 1, 2)
    _, optimal = optimal_assignments(REFERENCES, SAMPLE)
    assert tuple(cols) in optimal
