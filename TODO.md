## Now
- [ ] Step 5 local answering: `ask` command, source-grounded, per docs/specs/ask.md## Now

## Done
- [x] Step 3 evaluation set: 10 questions over 40 real notes (question set kept in
      the vault, not the repo): 9/10 found their file in the top 5, 8 at rank 1
- [x] Step 5: `ask "<question>" [-k N]`: top-k chunks as numbered sources to the local
      chat model, answer with `[n]` citations and `Inference:` labels, then the source list.
      No model call without chunks above ASK_MIN_SCORE; NOT_IN_NOTES stops the answer.
      Uncited statements and bad citation numbers are flagged on stderr (see docs/decisions.md).
- [x] Step 2 code review (Claude, agy, Codex) → fixed findings: index safety, local-only
      Ollama without proxies, chunking edge cases (fences, BOM, chunk size)
- [x] Step 2: ingest one vault folder: chunk markdown, embed, store in sqlite-vec,
      `search` command prints top-5 chunks with source filenames. No model call.
- [x] Step 1: qwen3 thinking-output fix: `think: false` on /api/chat, `strip_think` fallback, default qwen3:8b
      (qwen3 workaround removed on 2026-10-28; default is now gemma4:12b)

## Next
- [ ] Sentry: sensitivity classification at ingest (red/amber/green)
- [ ] Local answering follow-ups: tune ASK_MIN_SCORE against the eval set; an `ask` eval
      that checks every cited source is among the expected files
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
