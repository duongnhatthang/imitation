# Phase 2: Atari Setup and Smoke Test - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify the 7-game Atari suite end-to-end: expert policies load from HuggingFace sb3 org, observation spaces match between learner and expert, random baselines are computed and cached, and a smoke test on 2 Atari games runs all three methods (BC, DAgger, FTRL) to completion. Also set up the CC-server Python environment.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure/setup phase. Requirements are fully specified by REQUIREMENTS.md (ENV-01 through ENV-05, INFRA-05, INFRA-07).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FTRLDAggerTrainer` in `src/imitation/algorithms/ftrl.py` — Phase 1 output, needed for smoke test
- `SimpleDAggerTrainer` in `src/imitation/algorithms/dagger.py` — existing DAgger implementation
- `BC` in `src/imitation/algorithms/bc.py` — existing BC implementation
- Sacred experiment infrastructure in `experiments/`
- Benchmarking utilities in `benchmarking/`
- HuggingFace integration via `huggingface_sb3`

### Established Patterns
- Atari environments use `seals` wrappers with frame stacking, grayscale, 84x84 resize
- Expert policies from HuggingFace `sb3` org (e.g., `sb3/ppo-PongNoFrameskip-v4`)
- Score normalization: `(agent - random) / (expert - random)`

### Integration Points
- Expert policies loaded via `huggingface_sb3.load_from_hub()`
- Atari envs created via `make_atari_env` or `seals` wrappers
- Random baselines either from HumanCompatibleAI datasets or locally collected

</code_context>

<specifics>
## Specific Ideas

No specific requirements — setup follows standard patterns from the imitation library and HuggingFace ecosystem.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
