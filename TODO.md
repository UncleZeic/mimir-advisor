# TODO

## Now
- [ ] Step 2 code review (Claude, agy, Codex) → fix findings

## Done
- [x] qwen3 thinking-output fix: `think: false` on /api/chat, `strip_think` fallback, default qwen3:8b
- [x] Ingest one vault folder: chunk markdown, embed, store in sqlite-vec,
      `search` command prints top-5 chunks with source filenames. No model call.

## Next
- [ ] Evaluation set: questions with expected source files, scored with `eval`
- [ ] Sentry: sensitivity classification at ingest (red/amber/green)
- [ ] Adviser: source-grounded answering over retrieved chunks
- [ ] Chat: switch model and thinking mid-session (local Ollama models only, validated via /api/tags; show active model in prompt)
- [ ] Reranker in the retrieval path

## Someday
- Council (multi-model)
- MCP exposure
- Finance domain
- Health domain
- Scheduled agents
- Non-markdown ingest
