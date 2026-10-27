"""Local embeddings through the Ollama HTTP API. Nothing leaves the machine."""

from __future__ import annotations

import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request

from local_http import urlopen
from ollama_models import RemoteModelError, ensure_local_model


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float, batch_size: int = 16) -> None:
        self.base_url = base_url
        self.url = f"{base_url.rstrip('/')}/api/embed"
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_local_model()
        vectors: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))

        return vectors

    def _ensure_local_model(self) -> None:
        try:
            ensure_local_model(self.base_url, self.model, self.timeout, urlopen)
        except RemoteModelError as error:
            raise EmbeddingError(str(error)) from error
        except HTTPError as error:
            raise EmbeddingError(f"Cannot list Ollama models: HTTP {error.code}") from error
        except URLError as error:
            raise EmbeddingError(f"Cannot reach Ollama at {self.base_url}: {error.reason}") from error
        except TimeoutError as error:
            raise EmbeddingError(f"Listing Ollama models timed out after {self.timeout}s") from error
        except ValueError as error:
            raise EmbeddingError("Ollama returned an invalid model list") from error

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        request = Request(
            self.url,
            data=json.dumps({"model": self.model, "input": batch}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")

            if error.code == 404 or "not found" in detail:
                raise EmbeddingError(
                    f"Embedding model '{self.model}' is not available in Ollama. "
                    f"Run: ollama pull {self.model}"
                ) from error

            raise EmbeddingError(f"Ollama embedding failed: HTTP {error.code} {detail}") from error
        except URLError as error:
            raise EmbeddingError(f"Cannot reach Ollama at {self.url}: {error.reason}") from error
        except TimeoutError as error:
            raise EmbeddingError(f"Ollama embedding timed out after {self.timeout}s") from error
        except ValueError as error:
            # Malformed JSON or bytes: must not escape as a bare traceback that
            # skips the caller's cleanup of a half-built index.
            raise EmbeddingError("Ollama returned an invalid embedding response") from error

        if not isinstance(body, dict):
            raise EmbeddingError("Ollama returned an unexpected embedding response")

        embeddings = body.get("embeddings")

        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise EmbeddingError("Ollama returned an unexpected embedding response")

        return embeddings
