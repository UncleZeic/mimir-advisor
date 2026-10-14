"""Test doubles shared across test modules."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from unittest.mock import patch

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """Deterministic bag-of-words vectors: shared words => similar vectors.
    Lets retrieval be tested without Ollama or a real model."""

    def __init__(self, model: str = "fake-embed", dimensions: int = 256) -> None:
        self.model = model
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimensions] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def unreadable_file(target: Path):
    """Make one file raise PermissionError on read, without relying on chmod
    (which root ignores)."""
    read_bytes = Path.read_bytes

    def fake_read_bytes(path: Path) -> bytes:
        if path == target:
            raise PermissionError(13, "Permission denied", str(path))
        return read_bytes(path)

    return patch.object(Path, "read_bytes", fake_read_bytes)
