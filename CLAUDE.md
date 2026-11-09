# CLAUDE.md — FundLens

Standing context for every Claude CLI / Codex session on this repo. Read in full before writing code.

## Read first
`docs/BLUEPRINT.md` is the authoritative build plan: objective, non-negotiable decisions, the
phase chain, and the Checkpoint that ends each phase. **Do not start a phase whose predecessor's
Checkpoint is unmet.** If a request conflicts with the blueprint, say so before proceeding.

## One-line summary
A governed lifecycle for a small open-weight LLM — discovery, benchmarking, governed dataset,
QLoRA fine-tuning, eval gates, registry, promotion, serving, drift, rollback, retirement — applied
to structured extraction from SEC Form D private-fund filings.

## Hard rules
1. Ground truth comes from `primary_doc.xml`. **Never** use an LLM as the primary judge.
2. Splits are temporal **and** issuer-disjoint (CIK-level). Enforced by a failing test, not a comment.
3. All four baselines (B0–B3) run and are reported before the fine-tune. If B2 clears the bar,
   the correct output is a write-up saying the fine-tune was unjustified.
4. Every field returns `{value, char_start, char_end, confidence}`. No bare values.
5. Promotion is decided by `scripts/gate.py`, never by a human override.
6. Uncomputable metric → `None` **with a reason**. Never `0.0`.
7. Cache key = `sha256(model_version ‖ prompt_template_hash ‖ doc_sha ‖ schema_version)`.
8. No number reaches the README that `scripts/report.py` did not generate from `results/*.json`.
9. `data/` is gitignored and rebuilt by script. Never commit raw filings.
10. Set a descriptive `SEC_USER_AGENT`; respect EDGAR's 10 req/s limit.

## Build order
```
ingest → schema+labels → eval harness → baselines → PEFT → gate+registry
       → serving → routing/cache/quantisation → drift+rollback → report
```
The eval harness is built **before** the first model call. This is deliberate.

## Testing discipline
- Every phase producing a standalone unit gets its own test file before the next phase starts.
- Tests are deterministic and make zero model calls. Target 180+.
- `tests/test_dataset.py::test_no_cik_leakage` and `::test_temporal_ordering` must pass before
  any training run.

## Ask before
- Changing any rule above or any decision in BLUEPRINT.md §1.
- Adding a data source beyond EDGAR.
- Adding a top-level dependency outside the stack table in BLUEPRINT.md §2.
- Skipping a Checkpoint.
