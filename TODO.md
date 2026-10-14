# TODO

## Now
- [ ] Evaluation set: questions with expected source files, scored with `eval`

## Done
- [x] Ingest one vault folder: chunk markdown, embed, store in sqlite-vec,
      `search` command prints top-5 chunks with source filenames. No model call.

## Next
- [ ] Sentry: sensitivity classification at ingest (red/amber/green)
- [ ] Adviser: source-grounded answering over retrieved chunks
- [ ] Chat: switch model and thinking mid-session (local Ollama models only, validated via /api/tags; show active model in prompt)
- [ ] Chat: switch model and thinking mid-session (local Ollama models only, validated via /api/tags; show active model in prompt)
- [ ] Reranker in the retrieval path

## Someday
- Council (multi-model)
- MCP exposure
- Finance domain
- Health domain
- Scheduled agents
- Non-markdown ingest
