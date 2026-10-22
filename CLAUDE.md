# llmlm

@~/.claude/metalm/guidelines/index.md

The one model-call layer for the owner's lm repos (ytlm, beelm, corpuslm, mailerlm, supplylm, authorlm, gamelm, podlm): LiteLLM transport, model profiles, reply cache, price cache, cost estimate, usage ledger and spend caps. A library with no command line; each tool keeps its own verbs and installs this checkout editable. Depends on litellm and jsonschema. Design and migration plan: nagarkar/llmlm#1.

## Commands

- Install: `python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- Test: `.venv/bin/pytest` (offline; no test calls a model or the network)
- Lint: `.venv/bin/ruff check src tests && .venv/bin/lint-imports`
- Consumer gate: `tools/check-consumers.sh` (every migrated tool's suite against this tree; runs on push)
- Docs: `.venv/bin/metalm-gendocs`

## Exceptions to metalm

- None.

## Agent skills

- Issue tracker: GitHub issues on `nagarkar/llmlm`. See `docs/agents/issue-tracker.md`.
- Triage labels: defaults. See `docs/agents/triage-labels.md`.
