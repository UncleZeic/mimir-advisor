"""Walk a notes folder, chunk changed markdown files, embed and index them."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Callable, Iterator

from chunking import HEADING_SEPARATOR, chunk_markdown, note_title, split_frontmatter
from embeddings import Embedder
from store import IndexStore


@dataclass
class IngestReport:
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    empty: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)


def iter_markdown_files(root: Path) -> Iterator[Path]:
    """Every .md file under root, skipping hidden folders (.obsidian, .trash, .git).
    Symlinks are skipped, so a link can never pull a file from outside root
    into the index."""
    real_root = root.resolve()

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)

        if any(part.startswith(".") for part in relative.parts):
            continue

        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".md":
            continue

        try:
            inside = path.resolve().is_relative_to(real_root)
        except (OSError, RuntimeError):
            inside = False

        if inside:
            yield path


def embedding_text(title: str, heading: str, text: str) -> str:
    """Prefix each chunk with where it lives, so a section called 'Decision'
    is still findable by the note's subject."""
    context = title if not heading else f"{title}{HEADING_SEPARATOR}{heading}"
    return f"{context}\n\n{text}"


def ingest_folder(
    store: IndexStore,
    embedder: Embedder,
    root: Path,
    max_chars: int,
    progress: Callable[[str], None] = lambda message: None,
) -> IngestReport:
    store.check_model(embedder.model)
    store.check_chunk_max_chars(max_chars)
    report = IngestReport()
    known = store.file_hashes()
    seen: set[str] = set()

    for path in iter_markdown_files(root):
        relative = path.relative_to(root).as_posix()
        # Seen even when unreadable: a transient error must not drop the note
        # from the index; it is retried on the next ingest.
        seen.add(relative)

        try:
            raw = path.read_bytes()
        except OSError as error:
            report.unreadable.append(relative)
            progress(f"  skipped {relative}: {error.strerror or error}")
            continue

        file_hash = hashlib.sha256(raw).hexdigest()

        if known.get(relative) == file_hash:
            report.unchanged += 1
            continue

        text = raw.decode("utf-8", errors="replace")
        frontmatter, _ = split_frontmatter(text.replace("\r\n", "\n"))
        title = note_title(frontmatter, path.stem)
        chunks = chunk_markdown(text, max_chars=max_chars)

        if not chunks:
            report.empty.append(relative)
            store.replace_file(relative, file_hash, [], [], embedder.model)
            continue

        vectors = embedder.embed([embedding_text(title, c.heading, c.text) for c in chunks])
        store.replace_file(
            relative,
            file_hash,
            [(c.heading, c.ordinal, c.text) for c in chunks],
            vectors,
            embedder.model,
        )
        report.indexed += 1
        report.chunks += len(chunks)
        progress(f"  indexed {relative} ({len(chunks)} chunks)")

    for relative in sorted(set(known) - seen):
        store.remove_file(relative)
        report.removed += 1
        progress(f"  removed {relative}")

    return report
