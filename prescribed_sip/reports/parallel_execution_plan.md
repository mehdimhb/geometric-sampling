# Parallel execution plan

Approved scope: four process workers for independent evaluation-budget replications; isolated serial wall-time runs; preserve all completed results and original package code.

1. Freeze the current executable sources and manifest, interrupt the original runner, archive its saved outputs, and record SHA256 checksums of completed run JSON files.
2. Write failing integration tests for serial/parallel numerical agreement, process isolation, serial timing runs, immutable resume, and guarded source-version migration.
3. Extract the existing per-replication execution body without changing target generation, seeds, methods, budgets or file identities. Add a spawn-based process scheduler. Complete any partially executed legacy replication serially, then run evaluation jobs on four workers, wait for every worker to finish, and run wall-time jobs exclusively in the parent process. Complete each study stage before proceeding to the next.
4. Keep the original manifest environment; record authorized execution-only changes in an append-only execution history. Require matching mathematical-method and original-package hashes and a verified snapshot of changed old execution/report/test sources. Refuse ordinary resume after any source change.
5. Record worker count and process ID on new rows; treat legacy rows as one-worker runs in CSV summaries without rewriting their JSON. Keep concurrency strata separate in summaries and paired comparisons. Generate raw/checkpoint CSVs at phase boundaries and plots/reports at stage boundaries.
6. Run the full test suite and a four-worker smoke, verify original results checksums, review the scheduling/resume changes, then restart the full paper configuration. Add a frequently refreshed phase/progress file and log per completed replication so progress is visible.

No new scientific search algorithm or target-generation changes are in scope. Wall-clock safety caps may produce fewer evaluated candidates under parallel load; actual counts, timeouts and worker count remain explicit. Whole-study speedup is limited by the isolated timing phase.

Completed verification: 63 tests passed; four-worker smoke 106 rows; all 1,301 original run checksums unchanged. Reviewer identified the exact-only legacy interruption edge; its regression failed before the fix and passed after it. Full study restarted with four evaluation workers and exclusive timing.
