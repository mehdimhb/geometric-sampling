# Constrained GFS Astronomy Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible prototype in which the package's existing fixed-size GFS exchange repairs a seeded Madow design to hard-constraint feasibility and then improves entropy and attainable-pair SIP quality without changing the prescribed FIPs.

**Architecture:** Add one focused `graphical_sampling.constrained` module around the unchanged `Design.exchange`/`Design._switch` machinery: generic constraints and diagnostics, a minimal float64 `Design` adapter, candidate/search accounting, and independent SciPy reference solvers. Start from a seed-derived Madow order, run strict GFS repair, fork the repaired state into entropy and SIP searches, and present the experiment in an executed notebook; LP never supplies a search state.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`linprog`, `minimize`, `logsumexp`), Matplotlib, pandas, nbformat/Jupyter, pytest, existing `graphical_sampling.Design`, `Population`, `Order`, and `design_probability_mass`.

**Spec:** `docs/superpowers/specs/2026-09-29-constrained-gfs-astronomy-design.md`

## Global Constraints

- The production experiment is `Madow -> GFS constraint repair -> feasible GFS navigation -> GFS optimization`.
- LP verifies existence and supplies pair/MaxEnt references only; no LP probability vector may initialize or repair a GFS design.
- Keep `graphical_sampling/design.py`, `Design.exchange`, `Design._switch`, `Order`, and existing heap behavior unchanged.
- A precision adapter may lower fragment removal tolerance and preserve its class/RNG on copy, but must inherit `Design.exchange` and `Design._switch` unchanged.
- Use `N=10`, `n=4`, `pi=[0.55, 0.45, 0.50, 0.40, 0.35, 0.35, 0.40, 0.35, 0.35, 0.30]`, forbidden pairs `(0,1)`, `(2,3)`, `(4,5)`, and capacity two on `{6,7,8}`.
- Use role-tagged streams `SeedSequence([12345, 0])` through `SeedSequence([12345, 3])` for permutation, repair, entropy, and SIP respectively; no phase consumes another phase's RNG.
- The role-0 Madow order is exactly `[4, 8, 1, 3, 7, 9, 6, 0, 2, 5]`. Treat root seed `12345` as the predeclared reproducible demonstration seed: its permutation is generated without consulting the constraints or objectives, while a later paper study must repeat the experiment over many seeds.
- Stage 1 uses `pull_strategy="random"`, `exchange_coef=1.0`, and strict decrease of infeasible mass; Stage 2 uses the same inherited exchange with alpha drawn uniformly from `[0.1, 0.9]` and rejects every infeasible candidate.
- Default budgets are 20,000 proposed exchanges for repair, entropy, and SIP; reduced notebook tests may override them.
- Use float64, default invariant/SIP tolerance `1e-12`, and never repair a failed invariant by clipping, marginal adjustment, or renormalization.
- Label the optimizer result “Numerical constrained MaxEnt benchmark,” never “exact” or “GFS.”
- Add no astronomy-specific or other new dependency.
- Preserve all unrelated dirty and untracked files; every commit stages only the paths named in its task.

## Review Focus

- Invalid FIP vectors (wrong shape, nonfinite/out-of-range entries, or sum not integral) must fail before Madow construction or solver calls; Task 2 adds boundary-validation tests.
- An empty hard-feasible sample space must return a mathematical infeasibility result instead of crashing HiGHS or emitting a design; Task 4 adds this test.
- An initially feasible Madow design must finish repair immediately with a valid iteration-zero trace and no proposals; Task 3 adds this test.
- Stage 2 must reject an infeasible starting design rather than trying to repair it implicitly; Task 5 adds this test.
- A search whose checkpoint interval exceeds its proposal budget must still record its final state, including the zero-proposal case; Tasks 3 and 5 add these tests.

---

## File structure

- Create `graphical_sampling/constrained.py`: constraints, diagnostics, float64 Madow adapter, inherited-exchange proposals, repair/optimization runners, LP references, and immutable result records.
- Create `tests/test_constrained_gfs.py`: focused unit/integration tests for all reusable constrained-GFS behavior.
- Create `simulations/jupyters/constrained_gfs_astronomy_demo.ipynb`: statistician-facing, parameterized, fully executed experiment and figures.
- Create `tests/test_constrained_gfs_notebook.py`: notebook structure and reduced-budget clean-kernel execution contract.
- Do not modify `graphical_sampling/__init__.py`; the supported prototype API is imported from `graphical_sampling.constrained`.

### Task 1: Constraint model and finite-design diagnostics

**Files:**
- Create: `graphical_sampling/constrained.py`
- Create: `tests/test_constrained_gfs.py`

**Interfaces:**
- Produces: immutable `GroupCapacity(units: frozenset[int], capacity: int)`.
- Produces: immutable `ConstraintSystem(population_size: int, sample_size: int, forbidden_pairs: frozenset[frozenset[int]], group_capacities: tuple[GroupCapacity, ...])`.
- Produces: `is_feasible(sample: Iterable[int], constraints: ConstraintSystem) -> bool`.
- Consumes: `Mapping[tuple[int, ...], float]` from existing `design_probability_mass`.
- Produces: `compute_fip(mass, population_size) -> np.ndarray`, `compute_sip(mass, population_size) -> np.ndarray`, `design_entropy(mass) -> float`, and `infeasible_mass(mass, constraints) -> float`.
- Produces: immutable `DesignDiagnostics` with `probability_sum_error`, `max_fip_error`, `support_size`, `violating_support_size`, `infeasible_mass`, `entropy`, `fip`, and `sip`.
- Produces: `validate_probability_mass(mass, prescribed_fip, constraints, *, require_feasible=False, tolerance=1e-12) -> DesignDiagnostics`.

- [ ] **Step 1: Write failing tests for constraint validation and feasibility**

Add tests with the production constraints and explicit malformed cases:

```python
from dataclasses import replace

CONSTRAINTS = ConstraintSystem(
    population_size=10,
    sample_size=4,
    forbidden_pairs=frozenset({frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})}),
    group_capacities=(GroupCapacity(frozenset({6, 7, 8}), 2),),
)

def test_feasibility_checks_pairs_and_group_capacity():
    assert is_feasible((0, 2, 4, 6), CONSTRAINTS)
    assert not is_feasible((0, 1, 4, 6), CONSTRAINTS)
    assert not is_feasible((0, 2, 6, 7, 8), replace(CONSTRAINTS, sample_size=5))

def test_invalid_samples_and_constraints_are_rejected():
    with pytest.raises(ValueError, match="distinct"):
        is_feasible((0, 0, 2, 4), CONSTRAINTS)
    with pytest.raises(ValueError, match="range"):
        is_feasible((0, 2, 4, 10), CONSTRAINTS)
    with pytest.raises(ValueError, match="capacity"):
        GroupCapacity(frozenset({1, 2}), -1)
```

- [ ] **Step 2: Run the feasibility tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline pytest tests/test_constrained_gfs.py -k 'feasibility or invalid_samples' -q
```

Expected: collection/import failure because `graphical_sampling.constrained` does not exist.

- [ ] **Step 3: Implement the immutable constraint records and `is_feasible`**

Normalize forbidden pairs to two-element frozensets in `ConstraintSystem.__post_init__`; validate population/sample sizes, unit ranges, nonempty capacity groups, and capacities. Treat a well-formed sample violating a hard constraint as `False`; raise for wrong size, duplicates, or out-of-range units.

- [ ] **Step 4: Run the feasibility tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Write failing tests for hand-computed diagnostics and validation**

Use a four-unit, size-two design with mass `{(0,2): 0.4, (1,3): 0.6}` and assert:

```python
np.testing.assert_allclose(compute_fip(mass, 4), [0.4, 0.6, 0.4, 0.6])
np.testing.assert_allclose(np.diag(compute_sip(mass, 4)), [0.4, 0.6, 0.4, 0.6])
assert compute_sip(mass, 4)[0, 2] == pytest.approx(0.4)
assert design_entropy(mass) == pytest.approx(-(0.4*np.log(0.4) + 0.6*np.log(0.6)))
assert infeasible_mass(mass, constraints_forbid_0_2) == pytest.approx(0.4)
```

Also assert `validate_probability_mass` rejects negative mass, mass not summing to one, wrong FIPs, and an infeasible support when `require_feasible=True`, but reports the same support when `require_feasible=False`.

- [ ] **Step 6: Implement diagnostics and strict validation**

Use `math.fsum` for aggregated totals, float64 arrays for FIP/SIP, and zero relative tolerance. Do not mutate or normalize the input mapping.

- [ ] **Step 7: Run Task 1 tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline pytest tests/test_constrained_gfs.py -k 'feasibility or diagnostics or probability_mass' -q
```

Expected: all selected tests pass.

Commit only:

```bash
git add graphical_sampling/constrained.py tests/test_constrained_gfs.py
git commit -m "feat: add constrained design diagnostics"
```

### Task 2: Predeclared seeded Madow construction and inherited GFS proposals

**Files:**
- Modify: `graphical_sampling/constrained.py`
- Modify: `tests/test_constrained_gfs.py`

**Interfaces:**
- Consumes: Task 1 constraints and diagnostics; existing `Design`, `Population`, `Order`, `Cluster`, `Zone`, `_Sample`, and `design_probability_mass`.
- Produces: `role_rng(root_seed: int, role: int) -> np.random.Generator`.
- Produces: `seeded_madow_order(population_size: int, root_seed: int = 12345) -> np.ndarray` using role 0.
- Produces: `build_madow_design(prescribed_fip: Sequence[float], coordinates: np.ndarray, order: Sequence[int]) -> Design` backed by private `_Float64Design`.
- Produces: immutable `ExchangeProposal(candidate: Design | None, diagnostics: DesignDiagnostics | None, no_op: bool, constraint_feasible: bool)`.
- Produces: `propose_gfs_exchange(design, prescribed_fip, constraints, *, rng, exchange_coef, tolerance=1e-12) -> ExchangeProposal`.

- [ ] **Step 1: Write failing tests for the exact predeclared Madow order and stream isolation**

```python
def test_role_zero_generates_the_predeclared_madow_order():
    np.testing.assert_array_equal(
        seeded_madow_order(10, 12345),
        [4, 8, 1, 3, 7, 9, 6, 0, 2, 5],
    )

def test_role_streams_are_replayable_and_isolated():
    entropy_before = role_rng(12345, 2).random(8)
    role_rng(12345, 1).random(10_000)
    np.testing.assert_array_equal(role_rng(12345, 2).random(8), entropy_before)
    assert not np.array_equal(role_rng(12345, 2).random(8), role_rng(12345, 3).random(8))
```

- [ ] **Step 2: Write failing boundary tests for FIPs and Madow construction**

(Review Focus: invalid FIPs.) Assert rejection of non-1D, nonfinite, nonpositive, greater-than-one, and non-integer-sum FIPs before constructing `Population`. For the production vector, assert the built design uses `_Float64Design`, `_Float64Design.exchange is Design.exchange`, `_Float64Design._switch is Design._switch`, every sample has size four, the FIP error is below `1e-12`, and initial infeasible mass is `pytest.approx(0.9, abs=1e-12)`.

- [ ] **Step 3: Run the seeded-order and Madow tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline pytest tests/test_constrained_gfs.py -k 'role_ or madow or invalid_fip' -q
```

Expected: failures for missing functions.

- [ ] **Step 4: Implement role RNGs, `_Float64Design`, and Madow construction**

Use `np.random.default_rng(np.random.SeedSequence([root_seed, role]))`. `_Float64Design` overrides only `_push` (retain every strictly positive finite fragment and reject negative mass) and `copy` (preserve subclass); inherit `_build`, `exchange`, and `_switch`. Build a one-cluster/one-zone `Order` in the supplied order, call `_Float64Design.from_order`, and validate against the original FIPs without reading constraints.

- [ ] **Step 5: Run the construction tests and verify GREEN**

Run the Step 3 command. Expected: all selected tests pass.

- [ ] **Step 6: Write failing tests for an inherited exchange and structural no-op**

Monkeypatch `Design.exchange` with a counting wrapper around the original, call `propose_gfs_exchange`, and assert it was called once with `partitions=1` and `pull_strategy="random"`. Assert the input heaps and mass are unchanged.

For the null edge case, use parents `A={1,2,3,4}`, `B={1,2,3,5}` with equal full exchange mass, aggregate the inherited `_switch(..., coef=1.0)` output, and assert it equals the parents. Separately monkeypatch inherited `Design.exchange` to leave a candidate unchanged, then assert `propose_gfs_exchange` returns `no_op is True`, `candidate is None`, and no objective diagnostics.

- [ ] **Step 7: Implement deterministic candidate copying and no-op detection**

Seed the copied design and each copied heap from integers drawn only from the passed phase RNG, call inherited `candidate.exchange`, aggregate before/after support with existing `design_probability_mass`, compare the union of keys at absolute `tolerance`, then validate every non-no-op candidate's core invariants.

- [ ] **Step 8: Run Task 2 tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline pytest tests/test_constrained_gfs.py -k 'role_ or madow or invalid_fip or inherited_exchange or no_op' -q
```

Expected: all selected tests pass.

Commit only:

```bash
git add graphical_sampling/constrained.py tests/test_constrained_gfs.py
git commit -m "feat: propose inherited seeded GFS exchanges"
```

### Task 3: Strict GFS constraint-repair search

**Files:**
- Modify: `graphical_sampling/constrained.py`
- Modify: `tests/test_constrained_gfs.py`

**Interfaces:**
- Consumes: `propose_gfs_exchange`, Task 1 diagnostics, and the role-1 generator.
- Produces: immutable `SearchCounts` with `proposed`, `structurally_valid`, `no_op`, `constraint_feasible_candidates`, `constraint_infeasible_candidates`, `feasibility_rejected`, `accepted`, `non_improving`, and `invariant_failures`.
- Produces: immutable `SearchCheckpoint(proposal, accepted, objective, infeasible_mass, entropy, support_size, max_fip_error, elapsed_seconds)`.
- Produces: immutable `GfsSearchResult(final_design, trace, counts, runtime_seconds, status, reached_feasibility)`.
- Produces: `run_constraint_repair(initial_design, prescribed_fip, constraints, *, proposals: int, seed: int | np.random.SeedSequence, checkpoint_every=100, tolerance=1e-12) -> GfsSearchResult`.

- [ ] **Step 1: Write failing tests for the production repair path and accepted invariants**

Build the role-0 Madow design, call repair with `seed=np.random.SeedSequence([12345, 1])`, `proposals=2_000`, and assert:

```python
assert result.reached_feasibility
assert result.status == "feasible"
assert result.trace[0].infeasible_mass == pytest.approx(0.9, abs=1e-12)
assert result.trace[-1].infeasible_mass <= 1e-12
assert result.counts.accepted > 0
assert result.counts.proposed <= 2_000
assert result.counts.invariant_failures == 0
validate_probability_mass(final_mass, PI, CONSTRAINTS, require_feasible=True)
```

Extract checkpoints corresponding to accepted moves and assert strict `np.diff(V) < -1e-12`; assert at least one accepted checkpoint before the last still has positive `V`, proving infeasible support remains temporarily.

- [ ] **Step 2: Add repair edge-case tests**

(Review Focus: initially feasible and final checkpoint.) For an already feasible input, assert zero proposals/accepted moves, status `"already_feasible"`, and a one-row iteration-zero trace. With `proposals=0` on an infeasible input, assert status `"budget_exhausted"`, unchanged final mass, and a final/initial trace row. With `checkpoint_every > proposals`, assert the last attempted proposal is still represented.

- [ ] **Step 3: Run repair tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline pytest tests/test_constrained_gfs.py -k 'repair or already_feasible or checkpoint' -q
```

Expected: failures for missing repair API.

- [ ] **Step 4: Implement strict repair with maximal inherited exchanges**

Use `exchange_coef=1.0`. Accept only candidates with `new_V < current_V - tolerance`; record every accepted state plus proposal zero, configured proposal checkpoints, and the final state. Stop immediately when no infeasible key above tolerance remains and `V <= tolerance`. Never apply the Stage-2 feasibility gate during repair. Abort on a core invariant failure.

- [ ] **Step 5: Add and pass accounting/replay tests**

Run the production repair twice and assert equal final mass, trace values excluding elapsed time, counts, and status. Assert:

```python
counts.proposed == (
    counts.no_op + counts.invariant_failures + counts.feasibility_rejected
    + counts.accepted + counts.non_improving
)
assert counts.feasibility_rejected == 0
```

Then run all Task 3 tests. Expected: pass.

- [ ] **Step 6: Commit strict repair**

```bash
git add graphical_sampling/constrained.py tests/test_constrained_gfs.py
git commit -m "feat: repair constrained designs with GFS"
```

### Task 4: Independent LP existence, pair attainability, and numerical MaxEnt references

**Files:**
- Modify: `graphical_sampling/constrained.py`
- Modify: `tests/test_constrained_gfs.py`

**Interfaces:**
- Consumes: `ConstraintSystem` and prescribed FIPs; never consumes or returns a search-start `Design`.
- Produces: `enumerate_feasible_samples(constraints) -> tuple[tuple[int, ...], ...]`.
- Produces: `incidence_matrix(samples, population_size) -> np.ndarray`.
- Produces: immutable `FeasibilityResult(success, status, message, feasible_sample_count, positive_support_size, max_equality_error)`.
- Produces: `solve_feasibility_reference(prescribed_fip, samples, incidence, *, tolerance=1e-12) -> FeasibilityResult`.
- Produces: immutable `PairAttainability(maximum_sip: np.ndarray, categories: Mapping[tuple[int, int], str], explicit_forbidden, other_constraint_incompatible, marginal_inattainable, marginal_attainable)`.
- Produces: `analyze_pair_attainability(prescribed_fip, samples, incidence, constraints, *, tolerance=1e-12) -> PairAttainability`.
- Produces: immutable `MaxEntReference(probabilities, entropy, max_equality_error, success, message)`.
- Produces: `solve_numerical_maxent(prescribed_fip, samples, incidence, *, tolerance=1e-10) -> MaxEntReference`.

- [ ] **Step 1: Write failing enumeration and existence tests**

For the production constraints assert 122 lexicographically ordered samples, shape `(10, 122)`, fixed column sum four, successful feasibility, normalization error below `1e-10`, FIP error below `1e-10`, and maximum equality residual below `1e-10`. Do not assert the sparsity or positive-support size of the HiGHS solution.

(Review Focus: empty feasible space.) For `N=3`, `n=2`, with all three pairs explicitly forbidden, assert enumeration returns `()`, feasibility returns `success=False` with status `"infeasible"`, and no design/probability vector is exposed.

For a nonempty but FIP-incompatible system (`N=3`, `n=2`, forbid `(0,1)`, target `[0.9,0.9,0.2]`), assert explicit mathematical infeasibility rather than a candidate.

- [ ] **Step 2: Implement deterministic enumeration and HiGHS existence reference**

Validate reference-solver FIPs independently, allowing the closed interval `[0, 1]` (unlike Madow construction, which requires positive FIPs). Build equality rows `[ones; incidence]`, call `linprog(method="highs")`, and validate any successful solver vector before summarizing it. Keep solver probabilities local to the function.

- [ ] **Step 3: Write failing pair-category tests**

On the production problem, assert the three explicit pairs are category `"explicit_forbidden"`, have maximum SIP zero, and no non-attainable pair appears in `marginal_attainable`.

On unconstrained `N=3`, `n=2`, target `[1.0, 1.0, 0.0]`, assert `(0,1)` is marginal-attainable while `(0,2)` and `(1,2)` are `"marginal_inattainable"`, despite all three pairs occurring in the geometric sample space.

- [ ] **Step 4: Implement and validate per-pair maximization**

Skip solver calls for pairs absent from all feasible samples. For every other pair minimize the negative pair-incidence row with the same equality system, validate the returned vector/residual/objective, then categorize using the declared tolerance. Keep the four category sets disjoint and exhaustive over unordered pairs.

- [ ] **Step 5: Write failing numerical MaxEnt tests**

Assert the production reference is successful, probabilities are nonnegative and sum to one, FIP error is below `1e-9`, entropy is finite and no larger than `log(122)`, and the user-facing label contains “Numerical” and not “Exact”. Task 5 adds the comparison against repaired and entropy-GFS designs once those interfaces exist.

- [ ] **Step 6: Implement the numerical MaxEnt reference**

Solve the finite exponential-family dual with stable `logsumexp`: set `features = incidence[:-1]` and `target = prescribed_fip[:-1]` because the last FIP is implied by fixed sample size, minimize `logsumexp(theta @ features) - theta @ target` with gradient `features @ softmax(theta @ features) - target`, recover positive probabilities, and validate all original equalities. If SciPy does not reach tolerance, return/raise a numerical solver failure distinctly from LP infeasibility.

- [ ] **Step 7: Run Task 4 tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline pytest tests/test_constrained_gfs.py -k 'enumerat or feasibility_reference or empty_feasible or attainability or maxent' -q
```

Expected: all selected tests pass.

Commit only:

```bash
git add graphical_sampling/constrained.py tests/test_constrained_gfs.py
git commit -m "feat: add constrained design reference solvers"
```

### Task 5: Feasible entropy and attainable-pair GFS searches

**Files:**
- Modify: `graphical_sampling/constrained.py`
- Modify: `tests/test_constrained_gfs.py`

**Interfaces:**
- Consumes: a successfully repaired `Design`, `PairAttainability.marginal_attainable`, shared proposal/accounting records, and role-2/role-3 seeds.
- Produces: `attainable_pair_score(mass, pairs, *, offset=1e-12) -> float`.
- Produces: `run_feasible_gfs_search(initial_design, prescribed_fip, constraints, *, objective: Literal["entropy", "sip"], attainable_pairs=(), proposals: int, seed: int | np.random.SeedSequence, checkpoint_every=100, tolerance=1e-12) -> GfsSearchResult`.

- [ ] **Step 1: Write failing tests for score masking and invalid starts**

Build a hand mass where one attainable pair is zero and excluded pairs vary; assert the score equals the manual log sum over only the supplied pair tuple.

(Review Focus: infeasible Stage-2 start.) Call both objectives with the initial Madow design and assert `ValueError` matching `"feasible initial design"` before any proposal is made.

- [ ] **Step 2: Write failing end-to-end Stage-2 tests**

Repair the production Madow design once, then run each objective with `proposals=2_000`, `checkpoint_every=250`, and its role-tagged stream. For both results assert all final support is feasible, FIP error below `1e-10`, zero invariant failures, at least one accepted move, correct accounting, and deterministic replay. For entropy assert final entropy is at least initial entropy, the numerical MaxEnt reference is no smaller within solver tolerance, and accepted checkpoints strictly improve entropy. For SIP assert final attainable-pair score is at least initial score, accepted checkpoints strictly improve it, excluded pairs do not affect acceptance, and constraint-incompatible SIPs remain zero.

- [ ] **Step 3: Run Stage-2 tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline pytest tests/test_constrained_gfs.py -k 'pair_score or feasible_gfs or stage2 or entropy_search or sip_search' -q
```

Expected: failures for missing score/search functions.

- [ ] **Step 4: Implement the shared feasible-search loop**

Validate the initial design with `require_feasible=True`; draw alpha from the phase RNG before each proposal; reject no-ops; count and reject every constraint-infeasible candidate before scoring; accept only strict objective improvement beyond tolerance. Record proposal zero, every accepted state, configured checkpoints, and the final state.

- [ ] **Step 5: Add final-checkpoint and zero-budget tests**

(Review Focus: final checkpoint.) For an already feasible design with `proposals=0`, assert a one-row trace and unchanged final design. For a budget not divisible by `checkpoint_every`, assert the last proposal index is recorded. Confirm consuming role-2 proposals cannot alter a fresh role-3 replay.

- [ ] **Step 6: Run all reusable-module tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline pytest tests/test_constrained_gfs.py -q
```

Expected: all tests pass with no warnings from the new module.

Commit only:

```bash
git add graphical_sampling/constrained.py tests/test_constrained_gfs.py
git commit -m "feat: optimize feasible GFS designs"
```

### Task 6: Statistician-facing executed astronomy notebook

**Files:**
- Create: `simulations/jupyters/constrained_gfs_astronomy_demo.ipynb`
- Create: `tests/test_constrained_gfs_notebook.py`

**Interfaces:**
- Consumes: all public Task 1–5 interfaces; no notebook-local exchange/search/LP implementation.
- Produces: one tagged `parameters` cell with `SETTINGS` containing root seed, three 20,000-proposal budgets, checkpoint interval, tolerances, and `write_outputs`.
- Produces: notebook namespace `results` with keys `settings`, `madow`, `repair`, `entropy`, `sip`, `lp_feasibility`, `pair_attainability`, `maxent_reference`, `comparison`, `figures`, and `runtime_seconds`.

- [ ] **Step 1: Write the failing notebook structure test**

Assert the notebook exists, has exactly one tagged parameter cell, contains the eleven approved section headings in order, contains the phrase “LP verifies existence and provides benchmarks; GFS constructs, repairs, navigates, and optimizes,” and does not define its own `Design.exchange`, repair runner, pair LP, or MaxEnt solver.

- [ ] **Step 2: Write the failing reduced-execution contract**

Execute code cells in one namespace, overriding `SETTINGS` immediately after the parameter cell with repair budget `2_000`, entropy/SIP budgets `500`, checkpoint interval `100`, and `write_outputs=False`. Assert:

```python
np.testing.assert_array_equal(results["madow"]["order"], [4,8,1,3,7,9,6,0,2,5])
assert results["madow"]["infeasible_mass"] == pytest.approx(0.9, abs=1e-12)
assert results["madow"]["violating_support_size"] > 0
assert results["repair"]["reached_feasibility"]
assert results["repair"]["final_infeasible_mass"] <= 1e-12
assert results["repair"]["max_fip_error"] < 1e-10
assert results["repair"]["support_changed"]
assert results["repair"]["sip_change_norm"] > 0.0
assert results["entropy"]["max_fip_error"] < 1e-10
assert results["sip"]["max_fip_error"] < 1e-10
assert results["lp_feasibility"]["success"]
assert results["lp_feasibility"]["feasible_sample_count"] == 122
assert "Numerical constrained MaxEnt benchmark" == results["maxent_reference"]["label"]
assert results["figures"] == {"target_field", "repair_trace", "entropy_trace", "sip_heatmaps"}
```

Also assert proposal-accounting identities, all repaired/Stage-2 support feasibility, marginal-attainable zero counts before/after, and that `write_outputs=False` creates no files.

- [ ] **Step 3: Run notebook tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run --offline pytest tests/test_constrained_gfs_notebook.py -q
```

Expected: failure because the notebook is absent.

- [ ] **Step 4: Create the notebook with concise narrative and parameterized code**

Create the notebook JSON with `apply_patch`. Use manual display coordinates only for Figure 1. Build the role-0 order before constructing constraints, visibly state this ordering, run Madow diagnostics, repair, both Stage-2 searches, then present LP/MaxEnt/pair benchmark tables as references. If repair fails, report the failure and skip Stage 2 without substituting LP state.

Describe `12345` as the predeclared reproducible demonstration seed. State that the permutation itself is generated without constraint/objective input, while acknowledging that the methodological paper should assess robustness over many independently predeclared seeds.

- [ ] **Step 5: Add all required figures and tables**

Produce: target field with dashed explicit pairs; `V(D)` against proposal count with accepted count shown; entropy against accepted entropy moves; three shared-scale SIP heatmaps for Madow/repaired/pair-optimized designs with constraint-incompatible cells marked; comparison and proposal-accounting tables. Keep explanations concise and qualify all connectivity/global-optimum/scaling claims.

- [ ] **Step 6: Run the reduced notebook test and commit the unexecuted source notebook**

Run the Step 3 command. Expected: pass.

Commit only:

```bash
git add simulations/jupyters/constrained_gfs_astronomy_demo.ipynb tests/test_constrained_gfs_notebook.py
git commit -m "docs: add constrained GFS astronomy notebook"
```

### Task 7: Full execution, regression verification, and numerical handoff

**Files:**
- Modify generated outputs in: `simulations/jupyters/constrained_gfs_astronomy_demo.ipynb`
- Modify only if a verified defect is found: files from Tasks 1–6 and their tests

**Interfaces:**
- Consumes: the complete implementation and default notebook settings.
- Produces: an executed notebook with embedded outputs and the final numerical report requested by the spec.

- [ ] **Step 1: Run focused tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run --offline pytest tests/test_constrained_gfs.py tests/test_constrained_gfs_notebook.py -q
```

Expected: all pass.

- [ ] **Step 2: Run the entire repository test suite**

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run --offline pytest -q
```

Expected: all pass. Record every unrelated pre-existing failure by exact test name rather than omitting it.

- [ ] **Step 3: Execute the full default notebook in place**

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run --offline jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  simulations/jupyters/constrained_gfs_astronomy_demo.ipynb
```

Expected: successful clean-kernel execution with 20,000 proposals in each phase and embedded outputs.

- [ ] **Step 4: Re-run notebook and full tests against the executed artifact**

Run the commands from Steps 1 and 2 again. Expected: all pass, and notebook tests ignore embedded execution counts while validating semantic outputs.

- [ ] **Step 5: Inspect and record the scientific results**

Read the executed notebook outputs and record: feasibility sample count; initial support, `V(D_0)`, and violated-sample count/mass; repair proposals/accepted/no-op/non-improving counts and runtime; FIP errors throughout; repaired support/SIP changes; entropy initial/repaired/final/reference and gap; all pair categories; attainable zero count and minimum SIP before/after; Stage-2 counters/runtimes; and limitations/surprises. Confirm no result labels LP as the construction method.

- [ ] **Step 6: Inspect the diff and preserve unrelated work**

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only task files plus the user's pre-existing notebook-17/result changes; do not stage or modify those unrelated paths.

- [ ] **Step 7: Commit the executed notebook only**

```bash
git add simulations/jupyters/constrained_gfs_astronomy_demo.ipynb
git commit -m "docs: execute constrained GFS astronomy study"
```

- [ ] **Step 8: Perform completion review before reporting success**

Use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review` for a fresh whole-branch review. Address verified findings with focused red-green cycles and repeat Steps 1–6. The final response lists changed files, exact commands/results, notebook path, numerical findings, runtime, caveats, and the methodological-paper assessment.
