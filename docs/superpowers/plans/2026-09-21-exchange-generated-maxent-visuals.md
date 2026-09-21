# Exchange-Generated MaxEnt Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every empirical “MaxEnt” result in notebook 17 arise from unconditional constraint-preserving exchanges, and plot the actual initial and final microscopic segment arrangements for both the die and finite-population demonstrations.

**Architecture:** Extend the existing reversible die chain with an optional explicit atom representation while retaining the count-only path and its exact projected law. Add a focused finite-population exchange runner that copies and seeds an existing `Design`, calls the unchanged `Design.exchange()` method unconditionally, and computes read-only diagnostics. Update notebook 17 to consume these interfaces, retain analytical MaxEnt distributions only as theoretical benchmarks, and save the new grid-free segment figures and traces.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pandas, Matplotlib, nbformat/Jupyter, pytest, existing `graphical_sampling.Design` API.

**Spec:** `docs/superpowers/specs/2026-09-21-graphical-maxent-die-design.md`

## Global Constraints

- Entropy is a diagnostic consequence only; it must never affect proposal, acceptance, state selection, or stopping.
- Every constraint-valid exchange is accepted, including proposals that become self-loops.
- Keep `graphical_sampling/design.py` and the public `Design` API unchanged.
- Die block sizes remain `1`, `2`, and the fixed mixture `{3: 0.10, 4: 0.20, 5: 0.30, 6: 0.40}` for the three experiments.
- Full die iteration counts remain `200_000`, `400_000`, and `1_000_000`; seeds remain `20260921`, `20260922`, and `20260923`.
- The finite-population example remains `N=6`, `n=3`, with FIPs `[0.25, 0.35, 0.45, 0.55, 0.65, 0.75]`.
- Finite defaults are `FINITE_EXCHANGES=10_000`, seed `20260924`, one partition, `pull_strategy="default"`, `exchange_coef=0.75`, and diagnostics every 100 exchanges.
- Use the actual production chain states in all initial-versus-chaotic segment plots; do not run a separate display-only chain.
- Use a fixed white, grid-free plotting style.
- Preserve unrelated dirty and untracked work. Only notebook 17 and its dedicated result directory may be refreshed.
- Do not commit or push; the user explicitly requested an uncommitted implementation.

## Review Focus

- A tracked run with zero iterations must return identical initial/final atoms and the normal iteration-zero checkpoint.
- Atom selection must use distinct positions even when the selected labels repeat, preserving the ordered-without-replacement microscopic law.
- Tracked mode must retain only initial/final atom arrays, not one `D`-length array per checkpoint.
- Finite-design diagnostics must aggregate duplicate samples without mutating or merging the live exchange design.
- A final checkpoint must be recorded when the exchange count is not divisible by the checkpoint interval, and fixed-seed replays must agree exactly.

---

## File structure

- Modify `graphical_sampling/maxent/discrete.py`: explicit atom-step implementation and optional tracked state in `run_chain`/`ChainResult`.
- Create `graphical_sampling/maxent/finite_population.py`: deterministic unconditional `Design.exchange()` runner and non-mutating diagnostics.
- Modify `graphical_sampling/maxent/__init__.py`: export the new supported interfaces.
- Modify `tests/test_maxent_discrete.py`: microscopic tracking, zero-step, distinct-position, preservation, and replay tests.
- Create `tests/test_maxent_finite_population.py`: exchange invariants, non-mutating diagnostics, unconditional iteration count, checkpointing, and replay tests.
- Modify `simulations/jupyters/17_graphical_maxent_die.ipynb`: parameter cell, tracked die runs, real microscopic segment plots, exchanged finite design, corrected narrative, and output writing.
- Modify `tests/test_maxent_notebook.py`: reduced end-to-end contract for the corrected notebook semantics.
- Refresh `simulations/results/graphical_maxent_die/`: executed notebook tables, JSON, PNG, and PDF artifacts.

### Task 1: Track the real microscopic die state

**Files:**
- Modify: `graphical_sampling/maxent/discrete.py`
- Modify: `graphical_sampling/maxent/__init__.py`
- Test: `tests/test_maxent_discrete.py`

**Interfaces:**
- Consumes: `MomentConstraints`, `ReplacementKernel`, and the current `run_chain(...)` API.
- Produces: `ReplacementKernel.step_atoms_inplace(atoms, counts, rng) -> StepResult`.
- Produces: `run_chain(..., track_atoms: bool = False) -> ChainResult`.
- Produces: optional `ChainResult.initial_atoms` and `ChainResult.final_atoms`, both `np.ndarray | None`.

- [ ] **Step 1: Add failing tests for a tracked zero-step run and real atom endpoints**

Append tests that make the required representation explicit:

```python
def test_tracked_zero_step_chain_returns_actual_ordered_atoms():
    constraints = normalization_constraints([1, 2, 3])
    result = run_chain(
        [2, 1, 3],
        constraints,
        ReplacementKernel(constraints, {1: 1.0}),
        iterations=0,
        seed=41,
        checkpoint_every=2,
        track_atoms=True,
    )

    expected = np.array([1, 1, 2, 3, 3, 3])
    np.testing.assert_array_equal(result.initial_atoms, expected)
    np.testing.assert_array_equal(result.final_atoms, expected)
    np.testing.assert_array_equal(result.final_counts, [2, 1, 3])
    np.testing.assert_array_equal(result.checkpoint_iterations, [0])


def test_count_only_chain_does_not_allocate_atom_endpoints():
    constraints = normalization_constraints([1, 2])
    result = run_chain(
        [2, 2],
        constraints,
        ReplacementKernel(constraints, {1: 1.0}),
        iterations=2,
        seed=42,
        checkpoint_every=1,
    )
    assert result.initial_atoms is None
    assert result.final_atoms is None
```

- [ ] **Step 2: Run the two new tests and verify the API is missing**

Run:

```bash
.venv/bin/pytest tests/test_maxent_discrete.py::test_tracked_zero_step_chain_returns_actual_ordered_atoms tests/test_maxent_discrete.py::test_count_only_chain_does_not_allocate_atom_endpoints -q
```

Expected: failure because `run_chain` does not accept `track_atoms` and `ChainResult` has no atom fields.

- [ ] **Step 3: Add atom fields and the microscopic step**

Extend `ChainResult` with defaulted optional fields so existing count-only callers remain valid:

```python
initial_atoms: np.ndarray | None = None
final_atoms: np.ndarray | None = None
```

Implement `ReplacementKernel.step_atoms_inplace` with this exact transition:

```python
def step_atoms_inplace(
    self,
    atoms: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> StepResult:
    block_size = self._choose_block_size(rng)
    positions = rng.choice(len(atoms), size=block_size, replace=False)
    source_values = atoms[positions]
    source = tuple(self._outcome_to_index[int(value)] for value in source_values)
    signature = self._signature_from_indices(source)
    targets = self._groups[block_size][signature]
    target = tuple(int(value) for value in targets[int(rng.integers(len(targets)))])

    source_hist = np.bincount(source, minlength=self.constraints.size)
    target_hist = np.bincount(target, minlength=self.constraints.size)
    counts += target_hist - source_hist
    atoms[positions] = self._outcomes[np.asarray(target, dtype=np.int64)]

    return StepResult(
        block_size=block_size,
        source_labels=tuple(int(value) for value in source_values),
        target_labels=tuple(int(self._outcomes[index]) for index in target),
        changed=not np.array_equal(source_hist, target_hist),
    )
```

In `run_chain`, create `atoms = np.repeat(constraints.outcomes, counts)` only when `track_atoms=True`, call the tracked step inside the existing iteration loop, and verify at every checkpoint:

```python
atom_counts = np.array(
    [np.count_nonzero(atoms == value) for value in constraints.outcomes],
    dtype=np.int64,
)
if not np.array_equal(atom_counts, counts):
    raise RuntimeError("tracked atom multiplicities disagree with macro counts")
```

Copy the initial and final arrays when placing them in `ChainResult`. Reject non-boolean `track_atoms` with `TypeError`. Preserve the existing count-only branch exactly.

- [ ] **Step 4: Run the new endpoint tests**

Run:

```bash
.venv/bin/pytest tests/test_maxent_discrete.py::test_tracked_zero_step_chain_returns_actual_ordered_atoms tests/test_maxent_discrete.py::test_count_only_chain_does_not_allocate_atom_endpoints -q
```

Expected: 2 passed.

- [ ] **Step 5: Add failing tests for distinct-position selection, exact constraints, and replay**

Add:

```python
def test_tracked_chain_preserves_moments_and_atom_multiplicities():
    outcomes = np.arange(1, 7)
    constraints = MomentConstraints.from_moments(outcomes, [outcomes], [4.0])
    kwargs = dict(
        initial_counts=[0, 100, 0, 0, 0, 100],
        constraints=constraints,
        kernel=ReplacementKernel(constraints, {2: 1.0}),
        iterations=300,
        seed=43,
        checkpoint_every=37,
        track_atoms=True,
    )
    first = run_chain(**kwargs)
    second = run_chain(**kwargs)

    np.testing.assert_array_equal(first.final_atoms, second.final_atoms)
    np.testing.assert_array_equal(first.final_counts, second.final_counts)
    derived = np.array(
        [np.count_nonzero(first.final_atoms == value) for value in outcomes]
    )
    np.testing.assert_array_equal(derived, first.final_counts)
    for counts in first.checkpoint_counts:
        constraints.validate_counts(counts)


def test_atom_step_selects_distinct_positions_when_labels_repeat():
    class RecordingRng:
        def __init__(self):
            self.base = np.random.default_rng(44)
            self.positions = None
            self.replace = None

        def random(self):
            return self.base.random()

        def choice(self, values, *, size, replace):
            self.replace = replace
            self.positions = self.base.choice(values, size=size, replace=replace)
            return self.positions

        def integers(self, *args, **kwargs):
            return self.base.integers(*args, **kwargs)

    constraints = normalization_constraints([1, 2])
    kernel = ReplacementKernel(constraints, {2: 1.0})
    atoms = np.array([1, 1], dtype=np.int64)
    counts = np.array([2, 0], dtype=np.int64)
    rng = RecordingRng()
    result = kernel.step_atoms_inplace(atoms, counts, rng)
    assert result.source_labels == (1, 1)
    assert rng.replace is False
    assert len(np.unique(rng.positions)) == 2
    assert atoms.shape == (2,)
    assert counts.sum() == 2
```

The recording RNG asserts `replace=False` and confirms the two selected positions differ even though their labels coincide.

- [ ] **Step 6: Run all discrete-chain tests**

Run:

```bash
.venv/bin/pytest tests/test_maxent_discrete.py -q
```

Expected: all tests pass, including the existing count-projection and detailed-balance tests.

- [ ] **Step 7: Export the tracked interface without changing the root package API**

Keep the existing imports of `ChainResult`, `ReplacementKernel`, and `run_chain` in `graphical_sampling/maxent/__init__.py`. No root `graphical_sampling/__init__.py` or `Design` edits are required. Run:

```bash
.venv/bin/python -c "from graphical_sampling.maxent import run_chain, ChainResult, ReplacementKernel; print('tracked chain API import OK')"
```

Expected output: `tracked chain API import OK`.

### Task 2: Add a deterministic unconditional finite-design exchange runner

**Files:**
- Create: `graphical_sampling/maxent/finite_population.py`
- Modify: `graphical_sampling/maxent/__init__.py`
- Create: `tests/test_maxent_finite_population.py`

**Interfaces:**
- Consumes: an existing `graphical_sampling.Design` and a prescribed FIP vector.
- Produces: `design_probability_mass(design: Design) -> dict[tuple[int, ...], float]`.
- Produces: immutable `FiniteDesignDiagnostic` with `exchange`, `entropy`, `effective_support`, `support_size`, `probability_sum_error`, and `fip_error`.
- Produces: immutable `FiniteExchangeResult` with `final_design` and `trace`.
- Produces: `run_design_exchanges(initial_design, prescribed_fip, *, exchanges, seed, checkpoint_every, partitions=1, pull_strategy="default", exchange_coef=0.75, tolerance=1e-12) -> FiniteExchangeResult`.

- [ ] **Step 1: Write failing tests for non-mutating support aggregation and exchange invariants**

Create the test file with a helper that uses only existing package construction:

```python
import numpy as np
import pytest

from graphical_sampling import Design, Population
from graphical_sampling.maxent.finite_population import (
    design_probability_mass,
    run_design_exchanges,
)


PI = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])


def make_design():
    coords = np.column_stack([np.arange(6, dtype=float), np.zeros(6)])
    population = Population(coords, PI, variable=np.ones(6))
    population._n = 3
    return Design(population)


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
```

- [ ] **Step 2: Run the new file and verify the module is missing**

Run:

```bash
.venv/bin/pytest tests/test_maxent_finite_population.py -q
```

Expected: collection error because `graphical_sampling.maxent.finite_population` does not exist.

- [ ] **Step 3: Implement read-only diagnostics and exact validation**

Create `finite_population.py` with the two frozen dataclasses and a mass aggregator that iterates over `design._heaps` without calling `merge_identical()` or `design.entropy`:

```python
def design_probability_mass(design: Design) -> dict[tuple[int, ...], float]:
    pieces: dict[tuple[int, ...], list[float]] = {}
    for heap in design._heaps:
        for sample in heap:
            key = tuple(sorted(int(unit) for unit in sample.ids))
            pieces.setdefault(key, []).append(float(sample.prob))
    return {sample: fsum(values) for sample, values in pieces.items()}
```

The diagnostic helper must reject wrong sample sizes, non-finite/negative probabilities, probability mass error above `tolerance`, FIP-vector shape mismatch, and FIP error above `tolerance`. Compute entropy only from the aggregated positive masses:

```python
probability = np.asarray(list(mass.values()), dtype=float)
entropy = float(-np.sum(probability * np.log(probability)))
```

- [ ] **Step 4: Implement deterministic unconditional exchanges**

Copy the input design, seed the design RNG and every heap RNG from one `SeedSequence`, diagnose at exchange zero, and call `Design.exchange()` exactly once for every requested exchange:

```python
final_design = initial_design.copy()
children = np.random.SeedSequence(seed).spawn(1 + len(final_design._heaps))
final_design._rng = np.random.default_rng(children[0])
for heap, child in zip(final_design._heaps, children[1:]):
    heap.rng = np.random.default_rng(child)

trace = [diagnose_design(final_design, prescribed_fip, exchange=0, tolerance=tolerance)]
for exchange in range(1, exchanges + 1):
    final_design.exchange(
        partitions=partitions,
        pull_strategy=pull_strategy,
        exchange_coef=exchange_coef,
    )
    if exchange % checkpoint_every == 0 or exchange == exchanges:
        trace.append(
            diagnose_design(
                final_design,
                prescribed_fip,
                exchange=exchange,
                tolerance=tolerance,
            )
        )
```

Validate `exchanges >= 0`, `checkpoint_every > 0`, `partitions >= 1`, `0 < exchange_coef <= 1`, finite FIPs, and exact FIP-vector length before copying the design. There must be no conditional branch involving entropy.

- [ ] **Step 5: Run the invariant tests**

Run:

```bash
.venv/bin/pytest tests/test_maxent_finite_population.py::test_probability_aggregation_is_read_only tests/test_maxent_finite_population.py::test_exchange_runner_preserves_fixed_size_mass_and_fips -q
```

Expected: 2 passed.

- [ ] **Step 6: Add replay, unconditional-loop, and validation tests**

Add tests covering the remaining review risks:

```python
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
    original = Design.exchange

    def counted_exchange(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Design, "exchange", counted_exchange)
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
```

The unconditional-loop test checks the behavioral contract: the runner performs all 11 exchange calls and never selects a state or stopping point based on entropy.

- [ ] **Step 7: Export and run the complete finite test file**

Export `FiniteDesignDiagnostic`, `FiniteExchangeResult`, `design_probability_mass`, and `run_design_exchanges` from `graphical_sampling/maxent/__init__.py` and add them to `__all__`. Then run:

```bash
.venv/bin/pytest tests/test_maxent_finite_population.py -q
```

Expected: all tests pass.

### Task 3: Correct notebook 17 and add the requested segment figures

**Files:**
- Modify: `simulations/jupyters/17_graphical_maxent_die.ipynb`
- Modify: `tests/test_maxent_notebook.py`

**Interfaces:**
- Consumes: `run_chain(..., track_atoms=True)`, `run_design_exchanges`, `design_probability_mass`, and the existing theoretical MaxEnt solvers.
- Produces: notebook namespace objects `study[name]["chain"].initial_atoms`, `study[name]["chain"].final_atoms`, `finite_exchanged_design`, and `finite_exchange_trace`.
- Produces: result dictionary fields for actual exchange counts, initial/exchanged/theoretical entropy, support size, and FIP errors.

- [ ] **Step 1: Update the reduced notebook test to describe the corrected semantics**

Extend `reduced_settings` with:

```python
"finite_population": {
    "exchanges": 200,
    "seed": 20_260_924,
    "checkpoint_every": 20,
    "partitions": 1,
    "pull_strategy": "default",
    "exchange_coef": 0.75,
},
```

Replace assertions involving `finite_maxent_design` with assertions on the real exchange-generated design:

```python
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
assert finite["exchange_count"] == 200
assert finite["initial_support_size"] == 4
assert finite["exchanged_support_size"] <= 20
assert finite["theoretical_support_size"] == 20
assert finite["exchanged_probability_sum_error"] < 1e-12
assert finite["exchanged_fip_error"] < 1e-12
assert finite["theoretical_fip_error"] < 1e-10
assert namespace["finite_exchange_trace"]["exchange"].iloc[-1] == 200

initial_mass = namespace["design_probability_mass"](
    namespace["finite_gfs_design"]
)
exchanged_mass = namespace["design_probability_mass"](
    namespace["finite_exchanged_design"]
)
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
```

Retain the theoretical conditional-Poisson assertions, but rename their keys to `theoretical_*`; delete every assertion that treats the theoretical probabilities as a package `Design`.

- [ ] **Step 2: Run the notebook integration test and confirm the old notebook fails**

Run:

```bash
MPLCONFIGDIR=/tmp/graphical-sampling-mpl .venv/bin/pytest tests/test_maxent_notebook.py -q
```

Expected: failure because the notebook lacks finite exchange settings, tracked atoms, and `finite_exchanged_design`.

- [ ] **Step 3: Extend the single parameter cell**

Add the finite block to `SETTINGS` with the full defaults:

```python
"finite_population": {
    "exchanges": 10_000,
    "seed": 20_260_924,
    "checkpoint_every": 100,
    "partitions": 1,
    "pull_strategy": "default",
    "exchange_coef": 0.75,
},
```

Do not add an entropy threshold, best-checkpoint option, or acceptance switch.

- [ ] **Step 4: Make every die experiment retain its real microscopic endpoints**

In the shared experiment helper, call:

```python
chain = run_chain(
    initial_counts,
    constraints,
    kernel,
    iterations=config["iterations"],
    seed=config["seed"],
    checkpoint_every=config["checkpoint_every"],
    burn_in=config["burn_in"],
    track_atoms=True,
)
```

Immediately assert that the final atom histogram equals `chain.final_counts`. Keep the theoretical distribution computation and L1/KL/entropy diagnostics unchanged, but label them “theoretical MaxEnt benchmark” throughout prose and plot legends.

- [ ] **Step 5: Add the real Initial GAME versus Chaotic GAME die figure**

Add a plotting helper that consumes the stored atom array directly:

```python
def draw_atom_segments(axis, atoms, outcomes, title):
    D = len(atoms)
    lower = np.arange(D, dtype=float) / D
    upper = (np.arange(D, dtype=float) + 0.82) / D
    colors = plt.cm.Set2(np.linspace(0.05, 0.95, len(outcomes)))
    for color, outcome in zip(colors, outcomes):
        positions = np.flatnonzero(atoms == outcome)
        axis.vlines(
            np.full(len(positions), outcome),
            lower[positions],
            upper[positions],
            color=color,
            linewidth=0.45,
            alpha=0.9,
        )
    axis.set(xlim=(0.5, 6.5), ylim=(0, 1), xticks=outcomes, title=title)
    axis.grid(False)
```

Create a `3 x 2` figure. Each row uses the same experiment: `chain.initial_atoms` on the left under **Initial GAME** and `chain.final_atoms` on the right under **Chaotic GAME after {iterations:,} exchanges**. Label the shared axes “Die face” and “Microscopic atom position.” Save as:

- `die_game_segments_initial_vs_chaotic.png`
- `die_game_segments_initial_vs_chaotic.pdf`

- [ ] **Step 6: Replace the analytical finite design with real package exchanges**

Keep enumeration and the conditional-Poisson solve strictly as the theoretical benchmark. Delete construction of `finite_maxent_design` from `_Sample` objects. Instead run:

```python
finite_settings = SETTINGS["finite_population"]
finite_exchange_result = run_design_exchanges(
    finite_gfs_design,
    finite_pi,
    exchanges=finite_settings["exchanges"],
    seed=finite_settings["seed"],
    checkpoint_every=finite_settings["checkpoint_every"],
    partitions=finite_settings["partitions"],
    pull_strategy=finite_settings["pull_strategy"],
    exchange_coef=finite_settings["exchange_coef"],
)
finite_exchanged_design = finite_exchange_result.final_design
finite_exchange_trace = pd.DataFrame(
    [row.__dict__ for row in finite_exchange_result.trace]
)
```

Add a notebook helper with signature `fip_from_probability_mass(mass, N)` and use it to assert both initial and exchanged designs retain `finite_pi`. Build the summary with three distinctly named rows: `Initial Madow/GFS`, `Exchanged GFS/GAME`, and `Theoretical unequal-FIP MaxEnt benchmark`.

- [ ] **Step 7: Replace the finite segment and convergence figures**

Reuse the existing probability-segment renderer with `finite_gfs_design` and `finite_exchanged_design`. Title the panels exactly:

- `Initial Madow/GFS`
- `Exchanged GFS/GAME after 10,000 exchanges` (or the configured count)

Save as:

- `finite_population_segments_madow_vs_exchanged_gfs.png`
- `finite_population_segments_madow_vs_exchanged_gfs.pdf`

Add a two-panel finite diagnostic figure: entropy versus exchange count with a horizontal theoretical benchmark, and support size/effective support versus exchange count. Save as `finite_population_exchange_diagnostics.png` and `.pdf`. Use no background grids.

- [ ] **Step 8: Correct notebook output records and narrative**

Update the Markdown to state:

> Every valid exchange is accepted solely because it preserves the imposed constraints. Entropy is measured afterward. The analytical MaxEnt distribution is a theoretical benchmark, not the mechanism that creates the empirical design.

Write `finite_population_exchange_trace.csv` and update `finite_population_maxent.csv`, `finite_population_maxent_summary.json`, `configuration.json`, and `summary.csv` so their labels distinguish exchanged states from theoretical benchmarks. Do not save a theoretical probability vector as a `Design`. Keep all existing die output tables and figures, adding the new microscopic figure.

- [ ] **Step 9: Run the reduced notebook contract**

Run:

```bash
MPLCONFIGDIR=/tmp/graphical-sampling-mpl .venv/bin/pytest tests/test_maxent_notebook.py -q
```

Expected: pass, create no files in the pytest temporary output directory because `write_outputs=False`, and expose real tracked atom arrays plus a real exchanged package design.

### Task 4: Execute the complete study and verify every artifact

**Files:**
- Refresh: `simulations/jupyters/17_graphical_maxent_die.ipynb`
- Refresh: `simulations/results/graphical_maxent_die/*`
- Verify: `tests/test_maxent_discrete.py`
- Verify: `tests/test_maxent_finite_population.py`
- Verify: `tests/test_maxent_notebook.py`

**Interfaces:**
- Consumes: all completed APIs and the notebook's full default parameter cell.
- Produces: a top-to-bottom executed notebook and reproducible result artifacts from the configured exchange counts.

- [ ] **Step 1: Run focused unit tests before the expensive notebook**

Run:

```bash
MPLCONFIGDIR=/tmp/graphical-sampling-mpl .venv/bin/pytest tests/test_maxent_discrete.py tests/test_maxent_finite_population.py tests/test_maxent_constraints.py tests/test_maxent_diagnostics.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Execute notebook 17 in place with full settings**

Run:

```bash
MPLCONFIGDIR=/tmp/graphical-sampling-mpl .venv/bin/jupyter nbconvert --to notebook --execute simulations/jupyters/17_graphical_maxent_die.ipynb --inplace --ExecutePreprocessor.timeout=1800
```

Expected: exit code zero and no notebook cell with an error output.

- [ ] **Step 3: Verify the full output inventory**

Run:

```bash
test -s simulations/results/graphical_maxent_die/die_game_segments_initial_vs_chaotic.png
test -s simulations/results/graphical_maxent_die/die_game_segments_initial_vs_chaotic.pdf
test -s simulations/results/graphical_maxent_die/finite_population_segments_madow_vs_exchanged_gfs.png
test -s simulations/results/graphical_maxent_die/finite_population_segments_madow_vs_exchanged_gfs.pdf
test -s simulations/results/graphical_maxent_die/finite_population_exchange_diagnostics.png
test -s simulations/results/graphical_maxent_die/finite_population_exchange_diagnostics.pdf
test -s simulations/results/graphical_maxent_die/finite_population_exchange_trace.csv
```

Expected: every command succeeds and every file is nonempty.

- [ ] **Step 4: Inspect machine-readable invariants**

Run a read-only verification script that loads `configuration.json`, `summary.csv`, `finite_population_exchange_trace.csv`, and `finite_population_maxent_summary.json`, then asserts:

```python
assert configuration["finite_population"]["exchanges"] == 10_000
assert configuration["finite_population"]["seed"] == 20_260_924
assert finite_trace["exchange"].iloc[0] == 0
assert finite_trace["exchange"].iloc[-1] == 10_000
assert finite_trace["fip_error"].max() < 1e-12
assert finite_summary["exchange_count"] == 10_000
assert finite_summary["exchanged_fip_error"] < 1e-12
assert finite_summary["theoretical_fip_error"] < 1e-10
assert set(summary["Experiment"]) == {
    "Unconstrained",
    "Mean = 4",
    "Mean = 4, Var = 1.8475",
}
assert (summary["L1 distance"] < 0.05).all()
```

Expected: the script exits successfully and prints the three final die entropies plus the initial, exchanged, and theoretical finite-population entropies.

- [ ] **Step 5: Run the complete repository test suite**

Run:

```bash
MPLCONFIGDIR=/tmp/graphical-sampling-mpl .venv/bin/pytest -q
```

Expected: all tests pass. Existing harmless non-interactive Matplotlib `plt.show()` warnings may remain; no new failures or constraint warnings are acceptable.

- [ ] **Step 6: Perform the final notebook semantics audit**

Search the executed notebook for stale claims and objects:

```bash
rg -n "finite_maxent_design|Exact unequal-FIP MaxEnt.*Design|entropy-greedy|accept.*entropy|best entropy" simulations/jupyters/17_graphical_maxent_die.ipynb
```

Expected: no stale analytical-as-design object or entropy-based acceptance claim. Occurrences of `entropy-greedy` are acceptable only in an explicit sentence saying no such rule is used; otherwise revise the notebook and rerun Steps 2–5.

- [ ] **Step 7: Report results without committing**

Report the completed notebook path, the two new paired-segment figure paths, the configured exchange counts, final constraint/FIP errors, and initial/exchanged/theoretical entropies. Also state explicitly that `Design` was unchanged and no commit or push was made.
