# Fixed N12 n4 proof-of-concept implementation plan

**Goal:** Execute exactly two fixed, feasible reconstruction problems using the original optimizer and precision sensitivity, 20 seeds each, then stop.

**Specification:** `/home/bardia/.codex/attachments/e23a7d73-fff0-4c7b-9bc6-95aee8127a09/pasted-text.txt`.

**Constraints:** Preserve original package/user edits and all historical results. Do not resume the broad study or execute N>12. Maximum 5,000 scored stochastic candidates; early recovery RMSE<1e-10. Print runtime estimate before full execution; stop if estimate >7,200 seconds. Two targets only, same target for every repetition. BLAS/OpenMP one thread per process. Use 70–75% of detected logical CPUs while leaving at least two free.

- [x] Implement and test frozen equal/unequal cases, standard graphical initial support, random valid mass-exchange target walks, trajectory diagnostics, and snapshot selection based on dependence distance/entropy.
- [x] Implement and test original-search instrumentation using D2, strict independent design validation, dense enumerated SIP calculation, candidate checkpoints and profiling. High-precision diagnostic binds `round` to identity in a new function namespace, reusing the original run bytecode without editing package files.
- [x] Benchmark original optimizer on both targets. Save hardware, seeds, provenance, decomposition timings, memory and conservative projected end-to-end runtime; print estimate and enforce the two-hour approval gate.
- [x] Execute 20 independent paired seeds for Original GFS and high-precision sensitivity on each fixed target. Fit deterministic enumerated CP2, check Sinha applicability, reconstruct DSD support, and obtain certified LP oracle.
- [x] Independently validate every returned design and export support/moments/traces. Generate main and supplementary CSV/Markdown/LaTeX tables, six figure families, full numerical report and paper-ready report.
- [x] Request independent scientific/code review, run full relevant tests, verify original tracked edits and artifacts, and stop after returning numerical results.

Files: new `experiments/n12_targets.py`, `methods/n12_original.py`, `n12_proof.py`, `experiments/n12_artifacts.py`, `tests/test_n12_proof.py`. Output exclusively `results/N12_n4/`; final reports in `reports/`.

Artifact contract: `target_equal/target_unequal/{metadata.json,trajectory.csv,p0.json,pstar.json,Pi0.npy,Pistar.npy,pi.npy,x.npy}`. Support JSON has N,n,samples,probabilities. Target metadata includes case,pi,x,H0,Hstar,support0,supportstar,selected_exchanges,initial_RMSE,d_rho,validation and difference summaries. Runs under `raw_runs/*.json` contain case,method,replication,seed,status,final_D2,final_RMSE,final_Dinf,recovery_percent,runtime_seconds,candidate_evaluations,validation,details,trace,design_path. Trace entries contain candidates,D2,RMSE,Dinf,recovery_percent,elapsed_seconds. Each design has a matching SIP `.npy`. Benchmark/provenance JSON and seeds CSV are saved at the root.
