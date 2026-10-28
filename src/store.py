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
MOUNTS_FILE = "/proc/self/mounts"
# Filesystems whose files live on, or are synced to, another machine. FUSE is
# refused wholesale: sshfs, rclone, s3fs and cloud-drive clients all use it.
REMOTE_FS_TYPES = {
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "ncpfs", "afs", "ceph", "glusterfs", "9p", "davfs", "fuse",
}


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

    # Nor the other way round: an INDEX_DIR that holds the notes or the repo
    # is a shared tree (/, ~, a synced folder), not a place of its own.
    for label, contained in (("the repository", REPO_ROOT), ("DATA_SOURCE_DIR", source)):
        if contained.is_relative_to(base):
            raise StoreError(
                f"{label} ({contained}) is inside INDEX_DIR {base}. "
                "Point INDEX_DIR at a folder of its own, e.g. ~/.local/share/mimir-advisor/index."
            )

    check_local_filesystem(base)
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", source.name).strip("-") or "root"
    return base / INDEX_SUBDIR / f"{name}-{digest}.sqlite"


def check_local_filesystem(path: Path) -> None:
    fs_type = filesystem_type(path)

    if fs_type is not None and (fs_type in REMOTE_FS_TYPES or fs_type.startswith("fuse.")):
        raise StoreError(
            f"INDEX_DIR {path} is on a {fs_type} filesystem, which can live on or sync to "
            "another machine. Indexes contain note text; keep them on a local disk."
        )


def filesystem_type(path: Path) -> str | None:
    """Type of the mount holding ``path`` (which need not exist yet), from
    MOUNTS_FILE; None where that is unavailable, e.g. on macOS."""
    try:
        with open(MOUNTS_FILE, encoding="utf-8", errors="replace") as mounts:
            entries = [line.split() for line in mounts]
    except OSError:
        return None

    best: tuple[int, str] | None = None

    for fields in entries:
        if len(fields) < 3:
            continue

        # Spaces and tabs in mount points are octal-escaped (\040, \011).
        mount_point = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[1]))

        if path.is_relative_to(mount_point) and (best is None or len(mount_point.parts) >= best[0]):
            best = (len(mount_point.parts), fields[2])

    return best[1] if best else None


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

        # A symlinked index folder or file would put note text wherever the
        # link points (the vault, the repo, a synced folder), past the checks
        # index_path_for made on INDEX_DIR, and chmod the link's target.
        for candidate in (path.parent, path):
            if candidate.is_symlink():
                raise StoreError(f"{candidate} is a symlink. Index folders and files must not be links.")

        index_dir = path.parent.parent.resolve()
        resolved = path.resolve()

        if resolved.parent != index_dir / INDEX_SUBDIR:
            raise StoreError(f"Index file {resolved} is outside {index_dir / INDEX_SUBDIR}")

        check_local_filesystem(resolved)

        self.path = path
        path.parent.parent.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(exist_ok=True, mode=0o700)
        # Every open, not only on create: an index dir or file made by an older
        # version, another tool or a loose umask still holds note text.
        # O_NOFOLLOW and fchmod: a link swapped in after the checks above
        # fails to open instead of having its target chmodded.
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

        try:
            os.fchmod(dir_fd, 0o700)
        finally:
            os.close(dir_fd)

        # Create the file 0600 ourselves; sqlite3 would create it 0644 & ~umask.
        file_fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)

        try:
            os.fchmod(file_fd, 0o600)
        finally:
            os.close(file_fd)

        self.conn = sqlite3.connect(path)

        try:
            # Zero freed pages, so a removed note's text and vectors do not
            # linger in the file. Many builds default to this, not all do.
            self.conn.execute("PRAGMA secure_delete = ON")
            # Sorts and statement journals stay in memory instead of spilling
            # note text into /tmp, where secure_delete does not reach.
            self.conn.execute("PRAGMA temp_store = MEMORY")
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

    def _check_dimensions(self, vectors: list[list[float]]) -> None:
        """Every vector must match the index, or the first one if the index
        has none yet: sqlite-vec would fail on it mid-transaction."""
        indexed = self.get_meta("dimensions")
        expected = int(indexed) if indexed is not None else len(vectors[0])

        for vector in vectors:
            if len(vector) != expected:
                raise StoreError(
                    f"Embedding size changed ({expected} -> {len(vector)}). "
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
        if vectors:
            self._check_dimensions(vectors)

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
        indexed_dimensions = self.get_meta("dimensions")

        if indexed_dimensions is None:
            return []

        if len(vector) != int(indexed_dimensions):
            raise StoreError(
                f"The query embedding has {len(vector)} dimensions, but this index has "
                f"{indexed_dimensions}. Run ingest with --rebuild."
            )

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
