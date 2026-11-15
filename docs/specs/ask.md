# ask: source-grounded answering

Status: draft. The evidence rules below are mine to confirm before implementation.

## Command
`ask "<question>" [-k N]` — retrieve top-k chunks, answer from them only, cite sources.

## Evidence rules
- Every factual claim in the answer carries a source number.
- Anything not in the sources is either omitted or labelled "inference", never stated flat.
- If the sources don't answer the question: say so and stop. No general-knowledge answer.
- Paraphrase by default; quote only when the wording matters.
- Conflicting sources: report the conflict, don't pick a winner.
- No chunk above the relevance threshold: don't call the model at all.

## Output shape
Answer, then a Sources block: index, file path, heading.

## Not in scope
Sentry labels, reranking, streaming, multi-turn, any external model.

## Done when
- Five real questions over my own notes give answers I'd act on.
- Each answer's citations point at the section that actually says it.
- A question my notes don't cover returns "not in your notes".

## Open
- Relevance threshold value: pick empirically after the real-notes eval.
- Chunks per answer: default k=5, revisit.
