"""Split markdown notes into retrievable chunks.

Strategy: split by headings first (a section is the natural unit of meaning in
a note), then pack paragraphs up to ``max_chars``. A single paragraph longer
than ``max_chars`` is split on word boundaries with a small overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
TITLE_RE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)
HEADING_SEPARATOR = " › "


@dataclass(frozen=True)
class Chunk:
    heading: str
    ordinal: int
    text: str


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter, body). Frontmatter is '' when the note has none."""
    if not text.startswith("---\n"):
        return "", text

    end = text.find("\n---", 4)

    if end == -1:
        return "", text

    after = text.find("\n", end + 4)
    body = "" if after == -1 else text[after + 1 :]
    return text[4:end], body


def note_title(frontmatter: str, fallback: str) -> str:
    match = TITLE_RE.search(frontmatter)
    return match.group(1).strip() if match else fallback


def chunk_markdown(text: str, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    _, body = split_frontmatter(text)

    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        section = "\n".join(lines).strip()
        lines.clear()

        if not section:
            return

        heading = HEADING_SEPARATOR.join(title for _, title in stack)

        for piece in _split_section(section, max_chars, overlap):
            chunks.append(Chunk(heading=heading, ordinal=len(chunks), text=piece))

    for line in body.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            lines.append(line)
            continue

        match = None if in_fence else HEADING_RE.match(line)

        if match is None:
            lines.append(line)
            continue

        flush()
        level = len(match.group(1))

        while stack and stack[-1][0] >= level:
            stack.pop()

        stack.append((level, match.group(2).strip()))

    flush()
    return chunks


def _split_section(section: str, max_chars: int, overlap: int) -> list[str]:
    if len(section) <= max_chars:
        return [section]

    pieces: list[str] = []
    current = ""

    for paragraph in re.split(r"\n\s*\n", section):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            pieces.extend(_hard_split(paragraph, max_chars, overlap))
            continue

        if current and len(current) + 2 + len(paragraph) > max_chars:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        pieces.append(current)

    return pieces


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            space = text.rfind(" ", start + max_chars // 2, end)

            if space != -1:
                end = space

        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end >= len(text):
            break

        next_start = text.find(" ", max(end - overlap, start + 1), end)
        start = next_start + 1 if next_start != -1 else end

    return pieces
