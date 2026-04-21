# mimir-advisor

mimir-advisor helps you understand your life clearly, using your own data, without compromising your privacy.

It turns fragmented personal information into structured insight, so you can recall accurately, reflect objectively, and decide intentionally. Reasoning is local-first and policy-controlled.

## Vision

mimir-advisor is a private cognition layer for a more examined life.

It is intended to become:

- a trusted inner advisor, not a replacement for thinking
- a single interface to personal truth, grounded in real data
- a system that can reason about your life safely, and only share what is explicitly allowed

Long term, mimir-advisor should integrate personal data sources, provide longitudinal insight across patterns and decisions, and support controlled collaboration with external intelligence through anonymized delegation.

## Core Principles

- **Local-first:** your data stays on your machine by default.
- **Minimum exposure:** only the smallest necessary data is ever processed or shared.
- **Explicit trust boundaries:** no hidden data flows; external access is gated and auditable.
- **User authority:** the system advises, but you remain the decision-maker.
- **Traceable reasoning:** answers should be grounded in identifiable sources.

## System Overview

mimir-advisor is organized around five core layers:

1. **Ingestion:** imports data from local files and external exports.
2. **Normalization and Indexing:** structures data into documents, events, entities, and embeddings.
3. **Policy Layer:** labels and enforces data sensitivity, including private, financial, health, exportable, and non-exportable data.
4. **Reasoning Layer:** retrieves and synthesizes answers using local models.
5. **External Gateway:** optionally delegates to external AI after policy checks, anonymization, and logging.

## Query Flow

```text
User query
-> Retrieval of relevant data
-> Policy filtering
-> Local reasoning
-> Source-grounded response
```

Optional external delegation is disabled by default and only occurs after explicit policy approval.

## Privacy Model

- Raw data is not directly exposed to external models.
- Sensitive data is classified and filtered before use.
- External calls are disabled by default.
- Any external access must be explicitly gated, minimized, and logged.
- The system operates as a closed environment unless opened deliberately.

## Roadmap

### Phase 1: Local Intelligence

- Ingest selected personal data sources.
- Enable accurate personal queries.
- Provide source-grounded answers.
- Share zero data externally.

### Phase 2: Controlled Delegation

- Introduce external AI integration.
- Implement anonymization and pseudonymization.
- Enforce strict policy checks.
- Enable hybrid local and external reasoning.

## Initial Capabilities

mimir-advisor should answer questions such as:

- "What did I work on last week?"
- "When did I last discuss this topic?"
- "What changed across my notes, messages, and files?"

It should also aggregate across sources, provide structured summaries, and preserve privacy guarantees while doing so.

## Local CLI

Check whether the local services are ready:

```bash
python3 src/main.py health
```

Start an interactive chat with the local Ollama model:

```bash
python3 src/main.py chat
```

Send a one-shot prompt:

```bash
python3 src/main.py chat "What can you help me understand?"
```

## Design Intent

mimir-advisor is not a chatbot, a general AI assistant, or a cloud-first product.

It is a private, structured thinking system built on your own data, designed to help you see clearly, remember accurately, and decide better.
