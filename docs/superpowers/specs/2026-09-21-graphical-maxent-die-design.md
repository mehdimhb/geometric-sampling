# Graphical Maximum Entropy Die Demonstration — Design Specification

**Status:** Revised for user review
**Date:** 2026-09-21
**Repository:** `mehdimhb/graphical-sampling`

## 1. Purpose and scope

Add a model-based Graphical Maximum Entropy (GME/GFS) counterpart to the existing design-based survey-sampling implementation. The deliverable consists of:

- a reusable `graphical_sampling/maxent/` package for finite discrete count states, exact moment constraints, reversible microscopic proposals, MaxEnt benchmarks, and diagnostics;
- an executable publication-quality notebook, `simulations/jupyters/17_graphical_maxent_die.ipynb`;
- reproducible tables, checkpoint data, and figures under `simulations/results/graphical_maxent_die/`;
- explicit initial-versus-chaotic microscopic segment plots for each die experiment;
- a finite-population counterpart constructed by genuine `Design.exchange()` operations from an unequal-FIP Madow/GFS design;
- focused unit and notebook-integration tests.

`graphical_sampling/design.py`, `Design`, `Population`, `Order`, and the existing finite-population GFS API remain unchanged. The new subsystem reuses the conceptual idea of redistributing microscopic segments but does not force a model-based probability problem into the survey-sampling data structures.

The notebook demonstrates three six-sided-die cases: normalization only, fixed mean, and fixed mean plus variance. It presents this as a constructive Wallis/Jaynes realization of maximum entropy, not as a new entropy principle and not as an entropy optimizer. Entropy is always a measured consequence of constraint-preserving mixing. It is never a proposal criterion, acceptance criterion, or stopping rule.

## 2. State space and exact microscopic projection

Let a microscopic state be an assignment

\[
\omega=(\omega_1,\ldots,\omega_D)\in\{1,\ldots,6\}^D.
\]

Its count vector is `n = (n_1,...,n_6)`, with macro probabilities `p_i=n_i/D`. The production notebook retains the explicit labeled-atom assignment as well as `n`, so its probability-segment figures depict the actual simulated chain rather than a reconstructed or independent display run. Count-only operation remains available for callers that do not request microscopic snapshots.

For a chosen block size `k`, a microscopic transition is:

1. Choose `k` distinct atom indices in order, uniformly from the `(D)_k` ordered choices.
2. Read their ordered labels `u=(u_1,...,u_k)`.
3. Compute the constraint signature `T(u)`. It is empty for normalization only, `sum(u)` for a fixed mean, and `(sum(u), sum(u**2))` for fixed mean plus second moment.
4. Choose an ordered replacement tuple `v` uniformly from the precomputed set
   `P_k(T) = {v in {1,...,6}^k : T(v)=T(u)}`.
5. Replace the selected labels by `v` and update `n` by the same multiplicity difference. The current tuple and permutations that leave the count vector unchanged remain eligible, so self-loops are retained.

The count-only implementation samples `u` sequentially without replacement with probabilities determined by the current multiplicities. This is exactly the distribution obtained by choosing ordered labeled atom indices. The tracked implementation performs that microscopic selection directly and updates the atom vector and counts together. At every checkpoint, `bincount(atoms)` must equal the stored counts exactly. Both paths therefore implement the same transition law, and the tracked path supplies genuine atom positions for visualization.

## 3. Reversibility and stationary weighting

Block-size probabilities are fixed in advance and do not depend on the current state. For a particular block size, ordered atom-index tuple, and replacement `u -> v`, the forward proposal probability is

\[
q(\omega,\omega')=
\frac{w_k}{(D)_k}\frac{1}{|P_k(T(u))|}.
\]

Because `T(v)=T(u)`, the reverse move selects the same block size and atom indices and has the same replacement-set cardinality:

\[
q(\omega',\omega)=
\frac{w_k}{(D)_k}\frac{1}{|P_k(T(v))|}
=q(\omega,\omega').
\]

Multiple microscopic proposal paths between two states can be paired one-to-one with reverse paths of identical probability, so the aggregated transition probabilities are also symmetric. No Metropolis or entropy-greedy acceptance rule is used. All constraint-valid proposals are accepted, including self-loops. Consequently the uniform distribution on each connected component of the feasible microscopic state space is stationary; self-loops also remove periodicity concerns.

The transition rule is invariant to permutations of atom identities, so the count vector is a valid Markov lumping of the microscopic chain. A count vector has

\[
W(n)=\frac{D!}{\prod_i n_i!}
\]

compatible labeled microstates. Therefore the induced stationary count-level weight is proportional to `W(n)` within the explored feasible component. Stirling's approximation gives

\[
\log W(n)=D H(p)+O(\log D),
\]

so feasible macrostates concentrate near the distribution maximizing Shannon entropy. This conclusion is conditional on irreducibility over the relevant feasible fiber. The notebook will provide empirical connectivity evidence but will not claim a general connectivity theorem for the multi-moment move set.

## 4. Package architecture

### `graphical_sampling/maxent/constraints.py`

- `MomentConstraints`: immutable outcomes, integer-valued feature rows, and target moments.
- Construct exact block signatures and exact integer right-hand sides for a specified `D`.
- Reject a `D` when `D * target_moment` is not integral within strict tolerance.
- Validate normalization and moment totals directly from integer counts.
- `find_feasible_counts(...)`: use `scipy.optimize.milp` with integer variables and a deterministic nonconstant linear objective to construct a deliberately non-MaxEnt feasible count vector.
- `solve_discrete_maxent(...)`: independently solve the exponential-family moment equations with stable log-sum-exp calculations and verify the returned moments.

### `graphical_sampling/maxent/discrete.py`

- `DiscreteCountState`: validated integer counts, total `D`, probabilities, and exact moment totals.
- `ReplacementKernel`: precompute ordered replacement patterns grouped by exact constraint signature for each permitted block size.
- Sequential multiplicity-weighted sampling of ordered labels without replacement.
- A tracked atom step: select ordered distinct atom positions, choose a same-signature replacement uniformly, and update atoms and counts together.
- `run_chain(...)`: fixed iteration count, burn-in metadata, checkpoint callback/data, proposal count, count-changing move count, deterministic replay, and an optional tracked-atom mode returning the actual initial and final microscopic assignments.
- The tracked mode stores only initial and final atom vectors plus count checkpoints; it does not retain a `D`-length snapshot at every iteration.
- “Acceptance rate” in outputs means the fraction of proposals producing a changed macro count vector. All proposals are valid microscopic transitions; self-loops are retained and counted separately.

### `graphical_sampling/maxent/diagnostics.py`

- Shannon entropy, KL divergence, L1/L2 distance, mean, variance, and exact constraint residuals.
- Checkpoint and final summary helpers with explicit validation rather than silent repair.

### `graphical_sampling/maxent/__init__.py`

Expose the supported public API. The root `graphical_sampling/__init__.py` and `Design` API do not need modification; consumers import from `graphical_sampling.maxent`.

## 5. Experiment definitions

All settings appear in one tagged parameter cell near the notebook beginning. The defaults are chosen to be reproducible and practical on the current repository environment.

### Experiment 1: normalization only

- `D = 10_000`.
- Initial counts exactly represent `[0.40, 0.25, 0.15, 0.10, 0.05, 0.05]`.
- Block size `k=1`; a selected atom is relabeled uniformly to any of the six labels, including its existing label.
- `200_000` iterations, `100_000` burn-in marker, checkpoints every `1_000` iterations.
- Theoretical benchmark is derived by the generic MaxEnt solver with no moment features and must equal the uniform distribution.

### Experiment 2: fixed mean four

- `D = 10_000`.
- Initial counts `[0, 5_000, 0, 0, 0, 5_000]`, giving mean exactly four and intentionally not MaxEnt.
- Block size `k=2`; ordered replacement pairs are uniform within their common label-sum signature.
- `400_000` iterations, `200_000` burn-in marker, checkpoints every `2_000` iterations.
- Theoretical benchmark solves `p_i proportional to exp(lambda*x_i)` with mean four; it is not hard-coded.

### Experiment 3: fixed mean and variance

- Required moments are mean `4` and second moment `17.8475`, equivalent to variance `1.8475`.
- The notebook explicitly shows that `D=10_000` is impossible because the required first-moment total is even while the second-moment total is odd, although `x` and `x^2` have identical parity.
- Use `D = 20_000`, for which both integer totals are feasible. Integer feasibility is then confirmed by `milp`; the initial counts are generated automatically with a deterministic cubic-label objective to obtain a sparse, non-MaxEnt solution. The expected result from the current feasibility probe is `[3079, 0, 1, 7682, 9238, 0]`, but the notebook verifies rather than assumes it.
- Pair moves cannot change counts while preserving both sum and sum of squares. Use a fixed state-independent block-size mixture `{3: 0.10, 4: 0.20, 5: 0.30, 6: 0.40}`. Precomputed exact signatures provide the replacement patterns.
- `1_000_000` iterations, `500_000` burn-in marker, checkpoints every `5_000` iterations.
- The theoretical benchmark solves `p_i proportional to exp(lambda1*x_i + lambda2*x_i**2)` with both moments imposed; it is not hard-coded.

Use seeds `20260921`, `20260922`, and `20260923` respectively. Each experiment also performs a short duplicate replay from the same initial state and seed to assert deterministic reproducibility without doubling the full run.

The notebook parameter cell exposes the iteration counts and seeds. Changing an iteration count changes only how long the same exchange chain is run; it does not introduce an entropy-based stopping decision.

## 6. Notebook narrative and outputs

Create `simulations/jupyters/17_graphical_maxent_die.ipynb` with short, readable cells rather than a monolithic experiment script. It will:

1. Explain the microscopic construction, multinomial multiplicity, Wallis/Jaynes interpretation, feasible initialization, and the `D=10_000` parity obstruction.
2. Import and exercise the reusable package.
3. Display initial, final instantaneous, and theoretical MaxEnt probabilities for each experiment. The theoretical distribution is labeled explicitly as a benchmark and is never substituted for the exchange-generated state. Post-burn-in checkpoint averages may be reported as a mixing diagnostic but will not be substituted for the requested final state.
4. Report sum, mean, variance, entropy, theoretical entropy, L1, L2, KL, residuals, and count-changing acceptance rate.
5. Check the old vector `[0.059, 0.129, 0.178, 0.223, 0.209, 0.202]` and explicitly report mean `4` and variance `2.242`, correcting the old claim.
6. Produce per-experiment grouped bar plots, convergence plots for entropy/L1/moment residuals, and a compact paper-ready three-panel comparison.
7. Produce a three-row, two-column microscopic figure. Each row is one die experiment; the left panel is the real initial atom arrangement and the right panel is the real final chaotic arrangement. Atom position is the vertical coordinate, die label is the horizontal coordinate, and each atom is drawn as a short vertical segment. The initial ordered vector therefore forms contiguous blocks, while exchanges disperse the segments. No background grids are used.
8. End with a concise three-experiment results table and a carefully qualified connectivity discussion.

Save machine-readable diagnostics, a configuration record, summary CSV, and PNG/PDF figures under `simulations/results/graphical_maxent_die/`. The notebook must execute top-to-bottom from a fresh kernel.

## 7. Finite-population exchange counterpart

The notebook ends with a small finite-population example using `N=6`, `n=3`, and prescribed unequal FIPs

`[0.25, 0.35, 0.45, 0.55, 0.65, 0.75]`.

The empirical design is constructed only through the existing graphical-sampling machinery:

1. Build the initial ordered-systematic/Madow `Design` from the prescribed FIPs.
2. Copy it, seed its existing random generator reproducibly, and apply a user-visible `FINITE_EXCHANGES` number of genuine `Design.exchange()` calls.
3. Accept every valid exchange unconditionally. An exchange that cannot change the design is retained as a self-loop. Entropy is not inspected during proposal or acceptance.
4. Aggregate identical samples only for read-only diagnostics and final presentation, without changing the exchange rule.
5. Verify after recorded checkpoints and at the end that total probability is one, every sample has size `n`, and the original FIPs are preserved within floating-point tolerance.

The defaults are `FINITE_EXCHANGES = 10_000`, seed `20260924`, one partition, the package's `pull_strategy="default"`, and `exchange_coef=0.75`. Diagnostics are recorded every 100 exchanges and at the final exchange. These values live together in the notebook parameter cell so the exchange duration is easy to change without altering the transition rule.

The exact conditional-Poisson maximum-entropy solution remains in the notebook only as a theoretical benchmark. It may appear as a reference line, reference bars, or a labeled table row, but it must not be converted into or presented as the exchange-generated final design.

The paired segment figure uses the actual package design states: **Initial Madow/GFS** on the left and **Exchanged GFS/GAME** on the right. It is saved as PNG and PDF without background grids. A diagnostic trace reports exchange count, Shannon entropy, effective support, support size, and FIP error. Any approach toward the theoretical entropy is reported as an observed consequence rather than guaranteed monotonic optimization.

## 8. Validation and tests

Implementation follows test-driven development with focused tests under `tests/`:

- hand-check multiplicity-weighted sampling probabilities;
- enumerate a tiny microscopic system and verify the count transition matrix is the exact projection and satisfies detailed balance with multinomial weights;
- verify signatures preserve each requested integer moment;
- verify self-loops remain possible;
- verify theoretical MaxEnt solutions and moment residuals;
- verify `D=10_000` rejection and `D=20_000` feasibility for Experiment 3;
- verify fixed-seed replay produces identical final counts and diagnostics;
- verify tracked atom counts and stored macro counts agree exactly at every checkpoint;
- verify initial and final plotted atom arrays are the actual chain states;
- verify finite-population exchanges preserve total mass, fixed sample size, and the prescribed unequal FIPs;
- verify the finite exchange loop never evaluates entropy to decide whether to accept a move;
- execute the notebook with reduced test parameters, followed by one full notebook execution.

Notebook assertions require:

- nonnegative integer counts and probabilities summing to one;
- exact integer constraint preservation at every checkpoint;
- theoretical moment residuals below `1e-10`;
- final L1 distance to theoretical MaxEnt below `0.05` for each default full run;
- matching results for the deterministic replay probe;
- exact equality between tracked atom multiplicities and macro counts;
- finite-population FIP error below `1e-12` after the configured exchanges;
- no silent constraint repair.

## 9. Risks and explicit limitations

- Symmetry proves stationarity but not global irreducibility. `k=1` is connected without constraints; the constrained move sets will be assessed empirically through nonzero count-changing rates, movement away from sparse starts, occupation of all reachable faces, and convergence from the specified starts. The final report will not overstate this evidence.
- A final macrostate is random, so exact equality with theoretical MaxEnt is neither expected nor required. Large `D` supplies concentration; convergence traces distinguish equilibration from deterministic optimization.
- Random exchanges do not guarantee that entropy increases at every step. The notebook must not select a best-entropy checkpoint and call it the final chain state; it reports the state reached after the configured number of exchanges.
- Explicit atom tracking increases runtime relative to the projected count-only chain. It is nevertheless the preferred production path because the requested segment plots must show the actual simulated microstate. Only the initial and final atom arrays are retained to control memory use.
- Precomputing ordered patterns through `k=6` is small for six outcomes (`6^6=46,656`) but the implementation will validate requested sizes to avoid accidental combinatorial explosion for larger alphabets.
- Existing dirty/untracked repository work is preserved. Only notebook 17 outputs in its dedicated `simulations/results/graphical_maxent_die/` directory may be refreshed; unrelated simulation outputs remain untouched. No commit or push occurs before user inspection.
