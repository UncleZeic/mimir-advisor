"""sqlite-vec index: one file per source folder, no server, no port."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import sqlite3

import sqlite_vec


class StoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    path: str
    heading: str
    text: str
    score: float


def index_path_for(source_dir: str, index_dir: str) -> Path:
    """Each source folder gets its own index, so searching the fixtures can
    never return chunks from a real vault that was ingested separately."""
    source = Path(source_dir).expanduser().resolve()
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", source.name).strip("-") or "root"
    return Path(index_dir).expanduser() / f"{name}-{digest}.sqlite"


class IndexStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        is_new = not path.exists()
        self.conn = sqlite3.connect(path)

        if is_new:
            os.chmod(path, 0o600)

        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                heading TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
            """
        )

    def close(self) -> None:
        self.conn.close()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def check_model(self, model: str) -> None:
        indexed_model = self.get_meta("embed_model")

        if indexed_model and indexed_model != model:
            raise StoreError(
                f"This index was built with '{indexed_model}', but EMBED_MODEL is '{model}'. "
                "Run ingest with --rebuild to re-embed everything."
            )

    def _ensure_vector_table(self, model: str, dimensions: int) -> None:
        indexed_dimensions = self.get_meta("dimensions")

        if indexed_dimensions is None:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
                f"embedding float[{dimensions}] distance_metric=cosine)"
            )
            self._set_meta("embed_model", model)
            self._set_meta("dimensions", str(dimensions))
        elif int(indexed_dimensions) != dimensions:
            raise StoreError(
                f"Embedding size changed ({indexed_dimensions} -> {dimensions}). "
                "Run ingest with --rebuild."
            )

    def file_hashes(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT path, hash FROM files"))

    def replace_file(
        self,
        path: str,
        file_hash: str,
        chunks: list[tuple[str, int, str]],
        vectors: list[list[float]],
        model: str,
    ) -> None:
        with self.conn:
            self._delete_chunks(path)

            if vectors:
                self._ensure_vector_table(model, len(vectors[0]))

            for (heading, ordinal, text), vector in zip(chunks, vectors, strict=True):
                cursor = self.conn.execute(
                    "INSERT INTO chunks(path, heading, ordinal, text) VALUES (?, ?, ?, ?)",
                    (path, heading, ordinal, text),
                )
                self.conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, sqlite_vec.serialize_float32(vector)),
                )

            self.conn.execute(
                "INSERT INTO files(path, hash) VALUES (?, ?) "
                "ON CONFLICT(path) DO UPDATE SET hash = excluded.hash",
                (path, file_hash),
            )

    def remove_file(self, path: str) -> None:
        with self.conn:
            self._delete_chunks(path)
            self.conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def _delete_chunks(self, path: str) -> None:
        ids = [row[0] for row in self.conn.execute("SELECT id FROM chunks WHERE path = ?", (path,))]

        if ids and self.get_meta("dimensions") is not None:
            self.conn.executemany("DELETE FROM vec_chunks WHERE rowid = ?", [(i,) for i in ids])

        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))

    def stats(self) -> tuple[int, int]:
        files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return files, chunks

    def search(self, vector: list[float], k: int) -> list[SearchHit]:
        if self.get_meta("dimensions") is None:
            return []

        rows = self.conn.execute(
            """
            WITH knn AS (
                SELECT rowid, distance FROM vec_chunks
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT c.path, c.heading, c.text, knn.distance
            FROM knn JOIN chunks c ON c.id = knn.rowid
            ORDER BY knn.distance
            """,
            (sqlite_vec.serialize_float32(vector), k),
        ).fetchall()

        return [SearchHit(path, heading, text, 1 - distance) for path, heading, text, distance in rows]
