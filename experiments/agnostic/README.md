# Agnostic imitation experiments: public commands

Paths below are placeholders. Replace `PREP_DIR`, `AUDIT_ROOT`, `RESULTS_ROOT`,
`MANIFEST.json`, and `FRESH_OUTPUT_DIR` with your own locations. No command here
records hosts, job argv, or machine details, and none should be added to
tracked manifests. The design is described in
[`../AGNOSTIC_EXPERIMENT_PLAN.md`](../AGNOSTIC_EXPERIMENT_PLAN.md).

## Classical study

1. Prepare and qualify a frozen expert (stage 1), once per environment:

   ```bash
   python -m imitation.experiments.agnostic.run_classical prepare-expert \
       --env ENV --output-dir PREP_DIR/ENV --seed SEED --qualification-seed QSEED \
       --deadline DEADLINE_UTC
   ```

2. Run the diagnostic alias audit, once per environment. Its labels never
   reach training:

   ```bash
   python -m imitation.experiments.agnostic.run_classical audit \
       --env ENV --preparation-dir PREP_DIR/ENV --seed AUDIT_SEED \
       --output-dir AUDIT_ROOT/ENV --deadline DEADLINE_UTC
   ```

3. Write and freeze the scientific manifest (format and example in the
   docstring of `src/imitation/experiments/agnostic/analyze_classical.py`)
   before any confirmation learning cell runs. It pins the audit hashes from
   step 2 and must not change after step 4 begins.

4. Run one learning cell per environment, representation, and training seed,
   using the frozen budget, batch, checkpoints, and evaluation episodes:

   ```bash
   python -m imitation.experiments.agnostic.run_classical run-cell \
       --env ENV --preparation-dir PREP_DIR/ENV --representation REP \
       --seed N --output-dir RESULTS_ROOT/ENV/REP/seed-N --deadline DEADLINE_UTC
   ```

5. Before analysis, verify outside the analyzer that every queued job exited 0
   with a complete record and that each `result.json` hash matches the queue's
   record. The analyzer cannot see the queue.

6. Analyze into a fresh directory with the frozen manifest from step 3:

   ```bash
   python -m imitation.experiments.agnostic.analyze_classical \
       --manifest MANIFEST.json --results-root RESULTS_ROOT \
       --audit-root AUDIT_ROOT --output-dir FRESH_OUTPUT_DIR
   ```

   Exit codes: 0 complete (summary, report, figures), 2 invalid invocation or
   manifest (nothing written), 3 incomplete and 4 failed (inventory and
   reasons only, no estimates or figures).

   - The production manifest declares two environments by `mild` and `severe`.
     That freezes a Bonferroni family of 4 primary tests, so family intervals
     are at level 0.9875 for a 0.95 confidence level. The analyzer derives the
     family size from the manifest, so smaller test manifests report their own
     size.
   - The manifest's `source_stage2_files` must equal the hashes of the
     `classical.py` and `quantized.py` the analyzer imports; otherwise the
     manifest is invalid (exit 2). `summary.json` records the hashes of every
     module the analysis runs directly.
   - The manifest also pins `source_stage1_sha256` (the imported stage 1
     source fingerprint) and `package_versions` (the producers' `python`,
     `numpy`, `torch`, `stable_baselines3`, `gymnasium` and `imitation`
     versions), frozen before confirmation runs. Every cell and audit must
     match both pins; the analysis itself may run under other versions.
   - Any artifact that is malformed, fails a check, or has producer status
     `failed` gives exit 4. Missing, `running` or `partial` artifacts give
     exit 3. Every artifact stays in the inventory.
   - A complete record must finish by its effective deadline, which must equal
     the requested deadline clamped to the producer's hard cap. A record whose
     timestamps are missing, malformed or out of order fails.

## Toy study

```bash
python -m imitation.experiments.agnostic.analyze_toy \
    --manifest MANIFEST.json --results-root RESULTS_ROOT \
    --output-dir FRESH_OUTPUT_DIR
```
