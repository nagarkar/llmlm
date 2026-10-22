"""llmlm: the one model-call layer for the lm repos, a library with no command line.

A tool renders its own prompts (`llmlm.prompts` loads the versioned files), owns its reply
schemas and validator, names its config sections and keeps its own `usage`, `doctor` and run
verbs. Under them it calls this package:

- `llmlm.llm`: the LiteLLM transport, the verified `MODEL_PROFILES`, the retry rule, the reply
  cache, strict parsing, the spend caps and the invocation's log; the scripted transport and the
  cache as replay are the test seams.
- `llmlm.prices`: the machine-wide price cache (`~/.llmlm/prices.json`), refreshed weekly from
  litellm's published map, with hand-verified overrides for the models it lacks.
- `llmlm.estimate`: what a run would cost before it spends, from its payloads and the configured
  model's rates.
- `llmlm.typesafe` (extra `llmlm[typesafe]`): the HTTP client for TypeSafe's Jev, its retry
  rule, its input-only cost and its ledger line.
- `llmlm.images` (extra `llmlm[images]`): one rendered image per call, Gemini, OpenAI edits or
  litellm, with a count-only ledger line.
- `llmlm.ledger`: the `usage.jsonl` writer and reader, one line per call, rotated.
- `llmlm.usage`: the model-spend board a tool's `usage` verb and the one dashboard read.

Decisions:
- llmlm is a library only; verbs live in the tools. Enforced by: pyproject.toml (no
  `[project.scripts]`). (owner)
- Every tool depends on the sibling checkout, editable, guarded by the consumer-test gate:
  `consumers.toml` names each tool and its test command; `tools/check-consumers.sh` runs them on
  the scripted tier before a push and in CI. (owner)
- One ledger format, one cap vocabulary (`[llm].max_usd_per_run`, `[llm].max_usd_per_day`),
  one price cache, for every tool. (owner)
- An unknown model's cost is null, never an estimate. (owner)
- The reply-cache key is byte for byte what corpuslm's was. Enforced by:
  tests/test_llm.py::test_the_cache_key_is_byte_for_byte_what_it_always_was. (owner)

Terms:
- **tool**: a repo that calls a model through llmlm (ytlm, beelm, corpuslm, mailerlm, supplylm,
  authorlm, gamelm, podlm).
- **consumer-test gate**: every tool's test suite run against the llmlm working tree before a
  push and before a merge.
"""
