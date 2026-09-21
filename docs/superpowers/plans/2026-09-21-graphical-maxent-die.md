# Graphical Maximum Entropy Die Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable exact count-projected microscopic maximum-entropy chain and an executed publication-quality six-sided-die notebook with three experiments.

**Architecture:** `graphical_sampling/maxent/` owns integer moment constraints, count states, reversible replacement kernels, chains, MaxEnt solving, and diagnostics. Notebook 17 contains only experiment configuration, narrative, plotting, and artifact export. Existing survey-sampling `Design` code remains untouched.

**Tech Stack:** Python 3.12, NumPy, SciPy (`root`, `milp`), pandas, matplotlib, pytest, Jupyter/nbconvert.

**Spec:** `docs/superpowers/specs/2026-09-21-graphical-maxent-die-design.md`

## Global Constraints

- Do not modify `graphical_sampling/design.py` or destabilize the existing finite-population GFS API.
- Use count vectors only; transitions must exactly project uniform ordered labeled-atom selection.
- Retain self-loops, use no entropy-greedy acceptance, and keep block-size probabilities state-independent.
- Use block size 1, block size 2, and the fixed `{3:.10, 4:.20, 5:.30, 6:.40}` mixture for Experiments 1–3.
- Use notebook `simulations/jupyters/17_graphical_maxent_die.ipynb` and results directory `simulations/results/graphical_maxent_die/`.
- Do not overwrite unrelated outputs. Do not commit or push; the user explicitly reserved inspection before version-control actions.

## Review Focus

- A source ordered tuple near a count boundary must be sampled with falling-factorial multiplicity, never independent replacement sampling; Task 2 tests a literal hand-derived probability.
- Multiple microscopic paths to the same count transition must preserve multinomial detailed balance; Task 2 enumerates a complete tiny transition matrix.
- `D=10_000` has integral moment totals but is nevertheless infeasible by parity; Task 1 requires the integer feasibility solver to reject it.
- Pattern precomputation must reject an accidental combinatorial explosion before allocating it; Task 2 tests the configured cap.
- MaxEnt moment solving must fail loudly on an unreachable target rather than return an unconverged vector; Task 1 tests an out-of-range mean.

---

### Task 1: Exact constraints, integer feasibility, and theoretical MaxEnt

**Files:**
- Create: `graphical_sampling/maxent/constraints.py`
- Create: `tests/test_maxent_constraints.py`

**Interfaces:**
- Produces `MomentConstraints(outcomes, features, targets)`, `integer_rhs(D)`, `signature(labels)`, `validate_counts(counts)`, `find_feasible_counts(constraints, D, objective)`, and `solve_discrete_maxent(constraints)`.
- Task 2 consumes exact signatures and validated counts; Task 3 consumes theoretical probabilities and residuals.

- [ ] **Step 1: Write failing constraint and solver tests**

Tests must assert literal expected behavior:

```python
mean = MomentConstraints.from_moments([1,2,3,4,5,6], [[1,2,3,4,5,6]], [4.0])
assert mean.integer_rhs(10_000).tolist() == [10_000, 40_000]
assert mean.signature([2, 6]) == (8,)
np.testing.assert_allclose(solve_discrete_maxent(mean), EXPECTED_MEAN_MAXENT, atol=5e-9)

both = MomentConstraints.from_moments(x, [x, x**2], [4.0, 17.8475])
with pytest.raises(ValueError, match="infeasible"):
    find_feasible_counts(both, 10_000, objective=x**3)
counts = find_feasible_counts(both, 20_000, objective=x**3)
np.testing.assert_array_equal(both.constraint_matrix @ counts, [20_000, 80_000, 356_950])

impossible = MomentConstraints.from_moments(x, [x], [7.0])
with pytest.raises(RuntimeError, match="MaxEnt"):
    solve_discrete_maxent(impossible)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_constraints.py`  
Expected: collection fails because `graphical_sampling.maxent` does not exist.

- [ ] **Step 3: Implement the minimal exact constraint API**

Use integer feature matrices, `Fraction(str(target))` for RHS integrality checks, SciPy `milp` with all six variables integral, exact post-rounding validation, and stable softmax/root equations for MaxEnt. Normalization-only constraints use a `(0, q)` feature matrix and return uniform probabilities.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_constraints.py`  
Expected: all Task 1 tests pass.

- [ ] **Step 5: Record a no-commit checkpoint**

Append Task 1 test evidence to the execution ledger. Do not commit.

### Task 2: Count state, exact microscopic projection, and reversible chain

**Files:**
- Create: `graphical_sampling/maxent/discrete.py`
- Create: `tests/test_maxent_discrete.py`

**Interfaces:**
- Consumes `MomentConstraints.signature()` and validation from Task 1.
- Produces `DiscreteCountState`, `ReplacementKernel`, `StepResult`, `ChainResult`, `run_chain`, and `enumerate_count_transitions`.
- Task 4 consumes final counts, checkpoints, proposal counts, and count-changing rates.

- [ ] **Step 1: Write failing projection and reversibility tests**

Tests must cover:

```python
state = DiscreteCountState([2, 1], normalization_constraints([1, 2]))
kernel = ReplacementKernel(state.constraints, {1: 1.0})
transitions = enumerate_count_transitions(state.counts, kernel, 1)
assert transitions[(1, 2)] == pytest.approx(1/3 * 1/2)

# Enumerate all D=3 count states and assert W(n)P(n,m)=W(m)P(m,n).
for n, m in product(states, repeat=2):
    assert multinomial_weight(n) * P[n,m] == pytest.approx(
        multinomial_weight(m) * P[m,n], abs=1e-14
    )

# Source pattern remains in each signature group, so a self-loop is available.
assert kernel.contains_pattern(2, (8,), (2, 6))

with pytest.raises(ValueError, match="pattern cap"):
    ReplacementKernel(constraints, {12: 1.0}, max_patterns=100_000)
```

Also test exact mean and mean/second-moment preservation for 2- and 3–6-atom kernels, fixed-seed replay equality, and that checkpoint iterations include zero and the final iteration.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_discrete.py`  
Expected: import/attribute failure because the discrete API is absent.

- [ ] **Step 3: Implement count-projected proposals**

Precompute all ordered label patterns with `itertools.product(range(q), repeat=k)` and group them by exact integer signature. Sample ordered source labels sequentially without replacement from a six-count temporary vector. Select a target pattern uniformly from the matching group, update counts by two `bincount` vectors, and leave unchanged proposals as explicit self-loops. `run_chain` records immutable count copies only at checkpoints.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_constraints.py tests/test_maxent_discrete.py`  
Expected: all Task 1–2 tests pass.

- [ ] **Step 5: Record a no-commit checkpoint**

Append Task 2 test evidence to the execution ledger. Do not commit.

### Task 3: Diagnostics and public package API

**Files:**
- Create: `graphical_sampling/maxent/diagnostics.py`
- Create: `graphical_sampling/maxent/__init__.py`
- Create: `tests/test_maxent_diagnostics.py`

**Interfaces:**
- Consumes `MomentConstraints` and count/probability arrays.
- Produces `shannon_entropy`, `kl_divergence`, `distribution_diagnostics`, and the documented imports from `graphical_sampling.maxent`.
- Task 4 consumes a dictionary with probability sum, mean, variance, entropy, MaxEnt entropy, L1, L2, KL, and constraint residuals.

- [ ] **Step 1: Write failing diagnostics tests**

Use hand-computed literals:

```python
p = np.array([0.5, 0.5])
assert shannon_entropy(p) == pytest.approx(np.log(2))
assert kl_divergence(p, p) == pytest.approx(0.0)
row = distribution_diagnostics([1, 1], constraints, benchmark=p)
assert row["probability_sum"] == 1.0
assert row["mean"] == 1.5
assert row["variance"] == 0.25
assert row["L1"] == 0.0 and row["L2"] == 0.0 and row["KL"] == 0.0
with pytest.raises(ValueError, match="positive"):
    kl_divergence([1.0, 0.0], [0.0, 1.0])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_diagnostics.py`  
Expected: import failure because diagnostics and exports are absent.

- [ ] **Step 3: Implement validated diagnostics and exports**

Use natural logarithms, include only positive `p_i` terms in entropy/KL, reject a zero benchmark where `p_i>0`, and return numeric residual arrays without clipping or renormalizing invalid inputs.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_constraints.py tests/test_maxent_discrete.py tests/test_maxent_diagnostics.py`  
Expected: all maxent package tests pass.

- [ ] **Step 5: Record a no-commit checkpoint**

Append Task 3 test evidence to the execution ledger. Do not commit.

### Task 4: Publication notebook, experiment artifacts, and execution

**Files:**
- Create: `simulations/jupyters/17_graphical_maxent_die.ipynb`
- Create: `tests/test_maxent_notebook.py`
- Generate: `simulations/results/graphical_maxent_die/*`

**Interfaces:**
- Consumes the public APIs from Tasks 1–3.
- Produces the fully executed notebook, three checkpoint CSVs, summary/configuration CSV/JSON, per-experiment bar and convergence plots, and compact three-panel PNG/PDF.

- [ ] **Step 1: Write the failing notebook contract test**

The test parses and executes real notebook cells with reduced settings. It asserts:

```python
assert settings_tag_count == 1
assert results["summary"]["Experiment"].tolist() == [
    "Unconstrained", "Mean = 4", "Mean = 4, Var = 1.8475"
]
assert (results["summary"]["L1 distance"] < 0.05).all()
assert all(np.array_equal(a, b) for a, b in results["replay_pairs"])
assert results["old_variance"] == pytest.approx(2.242, abs=1e-12)
assert results["d10000_infeasible"] is True
```

Reduced test settings use smaller feasible `D` and enough iterations for behavior without writing production artifacts. The full notebook retains the specification defaults.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_notebook.py`  
Expected: failure because notebook 17 is missing.

- [ ] **Step 3: Create the readable notebook**

Use one tagged parameter cell and separate Markdown/code cells for theory, each experiment, diagnostics, plots, old-result correction, final table, and connectivity discussion. The notebook saves only under `simulations/results/graphical_maxent_die/`. Include exact assertions from the spec and do not hide experiments in a script.

- [ ] **Step 4: Run the reduced notebook test and package suite**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_constraints.py tests/test_maxent_discrete.py tests/test_maxent_diagnostics.py tests/test_maxent_notebook.py`  
Expected: all focused tests pass.

- [ ] **Step 5: Execute the full notebook from a fresh kernel**

Run: `MPLCONFIGDIR=/tmp/gme-mpl .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 simulations/jupyters/17_graphical_maxent_die.ipynb`  
Expected: exit 0, all code cells have execution counts, no error outputs, and all specified result files exist.

- [ ] **Step 6: Record a no-commit checkpoint**

Append Task 4 execution/test evidence and numerical results to the ledger. Do not commit.

### Task 5: Independent validation and final repository regression

**Files:**
- Modify only if a test exposes a defect: files created in Tasks 1–4.
- Do not modify: `graphical_sampling/design.py`.

**Interfaces:**
- Consumes every artifact and public interface.
- Produces final evidence and the user-facing report.

- [ ] **Step 1: Independently recompute saved results**

Load checkpoint/final CSVs without notebook variables. Recompute sums, moments, entropy, L1/L2/KL, acceptance rates, and MaxEnt residuals; verify exact count constraints and deterministic source configuration. Parse the executed notebook and assert zero error outputs.

- [ ] **Step 2: Run the focused tests**

Run: `.venv/bin/python -m pytest -q tests/test_maxent_constraints.py tests/test_maxent_discrete.py tests/test_maxent_diagnostics.py tests/test_maxent_notebook.py`  
Expected: all focused tests pass.

- [ ] **Step 3: Run the existing repository tests**

Run: `.venv/bin/python -m pytest -q tests`  
Expected: all existing and new root tests pass. Any unrelated pre-existing failure is named explicitly with evidence.

- [ ] **Step 4: Verify protected files and working tree**

Confirm `git diff -- graphical_sampling/design.py` contains no new maxent-task change, list every created file, and confirm no commit or push occurred.

- [ ] **Step 5: Record final completion evidence**

Append test totals, independent numerical validation, connectivity evidence/limitations, and all rulings to the ledger. Do not commit.
