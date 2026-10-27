"""Split markdown notes into retrievable chunks.

Strategy: split by headings first (a section is the natural unit of meaning in
a note), then pack paragraphs up to ``max_chars``. A single paragraph longer
than ``max_chars`` is split on word boundaries with a small overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

# Linear-time patterns only: notes can hold pasted text with very long lines,
# and a nested quantifier like (.+?)\s*#*\s*$ backtracks cubically on them.
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*)$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
YAML_KEY_RE = re.compile(r"^[\w.-]+\s*:(\s|$)")
HEADING_SEPARATOR = " › "


@dataclass(frozen=True)
class Chunk:
    heading: str
    ordinal: int
    text: str


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter, body). Frontmatter is '' when the note has none.

    A note opening with '---' is only frontmatter when a closing '---' line
    follows and everything between looks like YAML keys; otherwise the
    opening line is a horizontal rule and the whole note is body text.
    """
    text = text.lstrip("\ufeff")

    if not text.startswith("---\n"):
        return "", text

    lines = text.split("\n")

    for index in range(1, len(lines)):
        if lines[index].rstrip() != "---":
            continue

        block = lines[1:index]

        if not _looks_like_yaml(block):
            return "", text

        return "\n".join(block), "\n".join(lines[index + 1 :])

    return "", text


def _looks_like_yaml(block: list[str]) -> bool:
    """Empty, or a key first and then only keys, list items, indented or blank lines."""
    filled = [line for line in block if line.strip()]

    if not filled:
        return True

    return YAML_KEY_RE.match(filled[0]) is not None and all(
        line[0] in " \t-" or YAML_KEY_RE.match(line) for line in filled[1:]
    )


def note_title(frontmatter: str, fallback: str) -> str:
    for line in frontmatter.split("\n"):
        if not line.startswith("title:"):
            continue

        value = line[len("title:") :].strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()

        if value:
            return value

    return fallback


def heading_text(raw: str) -> str:
    """Heading text without an optional closing run of '#' (CommonMark: the
    run must follow a space or be the whole text, so 'C#' keeps its '#')."""
    text = raw.rstrip(" \t")
    without = text.rstrip("#")

    if without != text and (not without or without[-1] in " \t"):
        text = without.rstrip(" \t")

    return text or raw.strip()


def chunk_markdown(text: str, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    if max_chars <= overlap:
        # A hard split would never advance past its own overlap.
        raise ValueError(f"max_chars ({max_chars}) must exceed overlap ({overlap})")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    _, body = split_frontmatter(text)

    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    # The opening fence, e.g. "````"; '' outside a fence. Only the same
    # character at least as long closes it, so "```" inside "````" is content.
    fence = ""

    def flush() -> None:
        section = "\n".join(lines).strip()
        lines.clear()

        if not section:
            return

        heading = HEADING_SEPARATOR.join(title for _, title in stack)

        for piece in _split_section(section, max_chars, overlap):
            chunks.append(Chunk(heading=heading, ordinal=len(chunks), text=piece))

    for line in body.split("\n"):
        marker = FENCE_RE.match(line)

        if marker:
            run = marker.group(1)

            if not fence:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = ""

            lines.append(line)
            continue

        match = None if fence else HEADING_RE.match(line)

        if match is None:
            lines.append(line)
            continue

        flush()
        level = len(match.group(1))

        while stack and stack[-1][0] >= level:
            stack.pop()

        stack.append((level, heading_text(match.group(2))))

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
