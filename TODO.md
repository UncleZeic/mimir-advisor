# TODO

## Now
- [ ] Step 3 evaluation set: questions with expected source files, scored with `eval`;
      `eval --min-hits N` exits non-zero when fewer than N questions hit, so it can gate CI

## Done
- [x] Step 2 code review (Claude, agy, Codex) → fixed findings: index safety, local-only
      Ollama without proxies, chunking edge cases (fences, BOM, chunk size)
- [x] Step 2: ingest one vault folder: chunk markdown, embed, store in sqlite-vec,
      `search` command prints top-5 chunks with source filenames. No model call.
- [x] Step 1: qwen3 thinking-output fix: `think: false` on /api/chat, `strip_think` fallback, default qwen3:8b

## Next
- [ ] Sentry: sensitivity classification at ingest (red/amber/green)
- [ ] Local answering: source-grounded answers over retrieved chunks, local model only
- [ ] External gateway: policy-checked, anonymized, logged delegation; off by default

## Someday
- Chat: switch model and thinking mid-session (local Ollama models only, validated via /api/tags; show active model in prompt)
- Reranker in the retrieval path
- Council (multi-model)
- MCP exposure
- Finance domain
- Health domain
- Scheduled agents
- Non-markdown ingest
