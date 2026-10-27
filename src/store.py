"""sqlite-vec index: one file per source folder, no server, no port."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import sqlite3

import sqlite_vec

# Index files live in this dedicated folder under INDEX_DIR. Only this folder
# is ever chmodded, so pointing INDEX_DIR at an existing directory is safe.
INDEX_SUBDIR = "mimir-advisor-index"
REPO_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")


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
    base = Path(index_dir).expanduser().resolve()

    # Indexes hold note text: inside the repo they can be committed or copied
    # into a Docker image, inside the vault they get synced along with it.
    for label, forbidden in (("the repository", REPO_ROOT), ("DATA_SOURCE_DIR", source)):
        if base.is_relative_to(forbidden):
            raise StoreError(
                f"INDEX_DIR {base} is inside {label} ({forbidden}). "
                "Indexes contain note text; point INDEX_DIR somewhere else, "
                "e.g. ~/.local/share/mimir-advisor/index."
            )

    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", source.name).strip("-") or "root"
    return base / INDEX_SUBDIR / f"{name}-{digest}.sqlite"


def temporary_path(path: Path) -> Path:
    """Where `ingest --rebuild` builds the new index before swapping it in."""
    return path.with_name(f"{path.name}.tmp")


def remove_sidecars(path: Path) -> None:
    """A leftover journal next to a database is replayed into it on open, so
    one from an old or crashed file must never meet a new file of that name."""
    for suffix in SIDECAR_SUFFIXES:
        Path(f"{path}{suffix}").unlink(missing_ok=True)


class IndexStore:
    def __init__(self, path: Path) -> None:
        if path.parent.name != INDEX_SUBDIR:
            raise StoreError(f"Index files must live in a '{INDEX_SUBDIR}' folder, not {path.parent}")

        self.path = path
        path.parent.parent.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(exist_ok=True, mode=0o700)
        # Every open, not only on create: an index dir or file made by an older
        # version, another tool or a loose umask still holds note text.
        os.chmod(path.parent, 0o700)
        # Create the file 0600 ourselves; sqlite3 would create it 0644 & ~umask.
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        os.chmod(path, 0o600)
        self.conn = sqlite3.connect(path)

        try:
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
        except BaseException:
            self.conn.close()
            raise

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

    def check_chunk_max_chars(self, max_chars: int) -> None:
        """Chunks cut at a different size would mix with the old ones, so a
        changed CHUNK_MAX_CHARS needs a rebuild. Recorded on first use."""
        indexed = self.get_meta("chunk_max_chars")

        if indexed is None:
            with self.conn:
                self._set_meta("chunk_max_chars", str(max_chars))
        elif int(indexed) != max_chars:
            raise StoreError(
                f"This index was chunked with CHUNK_MAX_CHARS={indexed}, but it is now {max_chars}. "
                "Run ingest with --rebuild to re-chunk everything."
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
