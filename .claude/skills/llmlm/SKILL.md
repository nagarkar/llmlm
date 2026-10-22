---
name: llmlm
description: House rules for llmlm, the shared model-call library the lm repos import (transport, price cache, estimate, ledger, caps). Use when changing how any tool calls a model, prices a run, or records spend, or when a consumer's tests must run against an llmlm change.
---

# llmlm

llmlm is a library. It has no command line; a tool's own `usage`, `doctor` and run verbs call it.

| Ask | Where |
|---|---|
| Call a model with a versioned prompt, cached, parsed strictly, under the spend caps | `llmlm.llm.Client(root, config, validator=..., schema_loader=...)`, then `complete(stage, prompt, variables, tag=...)` |
| What would a run cost at the configured model, before spending | `Client.estimate(prompt, [variables, ...])`; the prompt's front matter sets `expected_output_tokens` |
| A model's price, or why a tool's `doctor` warns about prices | `llmlm.prices.default().rates(model)`; `.problems(models)` |
| Add a model litellm's map lacks or gets wrong | a row in `OVERRIDES` (`src/llmlm/prices.py`) with its source page and verified date |
| Add a model whose request shape was verified live | a row in `MODEL_PROFILES` (`src/llmlm/llm.py`), one exact model string, no wildcard |
| A tool's month of spend for its `usage` verb | `llmlm.usage.tool_spend(name, ledger_folder, settings_file, "YYYY-MM")` |
| Run the consumers' tests against this working tree | `tools/check-consumers.sh [name ...]` (also the pre-push hook) |

- Tests never call a model or the network: the scripted transport (`<dir>/<stage>/<tag>.json`) or a recorded cache directory serves replies. A test that prices anything builds `llmlm.prices.Prices(tmp_path / "prices.json", fetch=...)`; the machine-wide cache under `~/.llmlm/` is never read by a test.
- The reply-cache key (`payload_sha256`) never changes.
- A changed `MODEL_PROFILES` or `OVERRIDES` row names its verification (page and date) in a comment.
- The launcher for checks is `.venv/bin/pytest` in this checkout.

## Secrets

Set by the owner, never by an agent (metalm `operations.md#repo-secrets`). Check with `gh secret list --repo nagarkar/llmlm`.

| Secret | Scope | Read by | Set |
|---|---|---|---|
| `CONSUMERS_TOKEN` | fine-grained PAT, Contents read-only on the eight consumer repos in `consumers.toml`, with an expiry | `.github/workflows/consumers.yml` (`tools/check-consumers.sh --ci`) | `gh secret set CONSUMERS_TOKEN --repo nagarkar/llmlm`, value pasted at the prompt; create it at https://github.com/settings/personal-access-tokens |

## Merging

- Merge only on the owner's word in chat.

## Rulings

- Every mention of a ruling in a message links inline to the document that records it: a markdown link whose target is the absolute path of the rulings file, or the full URL of its Doc tab, on the ruling's id. A ruling named without its link is a claim the owner cannot check. Standard for every repo skill: metalm `guidelines/requirements.md`, "Capturing owner input" (owner).
