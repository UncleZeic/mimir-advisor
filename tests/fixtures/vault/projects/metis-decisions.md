---
title: Metis architecture decisions
tags: [metis, adr]
created: 2026-09-20
---
# Metis architecture decisions

## Storage
Chose sqlite-vec over ChromaDB for the vector store. A single file needs no
server, no port and no health check, so there is nothing to expose on the
network. Brute-force search is fine up to tens of thousands of chunks.

## Embeddings
Embeddings come from bge-m3 served by the local Ollama. It handles Romanian
and English and avoids pulling PyTorch into the project.

## Chunking
Split by headings first, then pack paragraphs up to roughly 1200 characters.
Each chunk is embedded with its note title and heading path as a prefix.
