# llmlm

The one model-call layer for the lm repos. A library: `llmlm.llm` (transport, profiles, cache, caps), `llmlm.prices` (the weekly-refreshed price cache), `llmlm.estimate` (cost before a run), `llmlm.ledger` and `llmlm.usage` (the spend ledger and board). Beside them, for pictures: `llmlm.images` (one rendered image per call), `llmlm.cast` (one look for a character across many pictures) and `llmlm.picker` (the page for choosing among candidate images, served by each app itself). Start at `docs/generated/index.md`; the design is nagarkar/llmlm#1.
