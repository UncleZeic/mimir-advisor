# Decisions

Dated, one entry per decision. Newest first.

## 2026-11-15 Step 5: `ask` answers only from retrieved notes

- **Reuse the search path.** `ask` calls `search_index` exactly like `search`; no second
  retrieval code path, no reranking. Chunks become numbered sources in retrieval order.
- **Threshold before the model.** `ASK_MIN_SCORE` (cosine similarity, default 0.35, range
  0..1, via `.env` and `Settings`) drops weak hits. If no hit survives, the chat model is
  never called and `ask` exits 1 with a plain message. Rationale: a model handed irrelevant
  chunks tends to answer anyway; not asking is the only reliable guard. 0.35 is a starting
  point for bge-m3 and is meant to be tuned against the eval set.
- **Evidence rules in the system prompt, checked after.** The model must cite `[n]` after
  every statement, prefix anything beyond the sources with `Inference:`, and reply with the
  exact token `NOT_IN_NOTES` when the sources do not answer. `answer.check_answer` maps
  citations to sources and lists statements that break the rules. The checks are post hoc:
  they cannot make a local model comply, so violations are printed as warnings on stderr
  rather than silently dropped or silently accepted.
- **Fixed token for "not in notes".** Matching a token is deterministic; matching free
  text about "I don't know" is not. When the token appears anywhere in the reply, `ask`
  prints one line and exits 1 without printing the reply or a source list.
- **One chat function.** `ask_ollama` takes `messages` so `ask` and `chat` share the same
  `/api/chat` request, the same `ensure_local_model` check and the same local-only opener.
  No new network path. Cloud model names are refused in `Settings`; a model Ollama reports
  with `remote_host` is refused before the prompt is built into a request.
- **Sources printed as path and heading, all of them.** Every source sent to the model is
  listed, marked cited or not cited, with its score. The reader sees what the model saw,
  not only what it chose to cite.
- **Out of scope here:** streaming, reranking, Sentry labels, the external gateway.
