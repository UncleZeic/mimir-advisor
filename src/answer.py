"""Source-grounded answers: prompt construction and evidence checks for `ask`.

Pure functions, no I/O. The model is told the evidence rules; the checks here
make it visible when an answer breaks them, because a local model cannot be
trusted to follow instructions every time.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from store import SearchHit

# The model replies with exactly this token when the sources do not answer the
# question. A fixed token is checked, not free text about "not knowing".
NOT_IN_NOTES = "NOT_IN_NOTES"
INFERENCE_LABEL = "Inference:"
CITATION_RE = re.compile(r"\[(\d+)\]")
# A statement ends at a sentence stop followed by whitespace, or at a line end.
STATEMENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

SYSTEM_PROMPT = (
    "You are mimir-advisor, a local-first private advisor. You answer only from the numbered "
    "sources the user provides, which are excerpts from their own notes.\n"
    "Rules:\n"
    "1. Every statement in your answer must end with the number of the source it comes from, "
    "written as [1], [2] and so on. A statement drawn from several sources cites each of them.\n"
    f"2. Anything you add that is not stated in the sources (a conclusion, a guess, a general "
    f"remark) must be on its own line starting with '{INFERENCE_LABEL}'.\n"
    f"3. If the sources do not contain the answer, reply with exactly {NOT_IN_NOTES} and nothing "
    "else. Never answer from general knowledge.\n"
    "4. Do not invent source numbers. Do not mention sources that were not provided.\n"
    "Be concise."
)


@dataclass(frozen=True)
class Source:
    number: int
    hit: SearchHit

    @property
    def location(self) -> str:
        return f"{self.hit.path} › {self.hit.heading}" if self.hit.heading else self.hit.path


@dataclass(frozen=True)
class AnswerReport:
    not_in_notes: bool
    cited: tuple[int, ...]
    out_of_range: tuple[int, ...]
    uncited: tuple[str, ...]


def select_sources(hits: list[SearchHit], min_score: float) -> list[Source]:
    """Hits at or above ``min_score``, numbered from 1 in retrieval order."""
    kept = [hit for hit in hits if hit.score >= min_score]
    return [Source(number, hit) for number, hit in enumerate(kept, start=1)]


def format_sources(sources: list[Source]) -> str:
    blocks = []

    for source in sources:
        blocks.append(f"[{source.number}] {source.location}\n{source.hit.text.strip()}")

    return "\n\n".join(blocks)


def build_messages(question: str, sources: list[Source]) -> list[dict[str, str]]:
    if not sources:
        raise ValueError("build_messages needs at least one source; the model is not asked without evidence")

    user = (
        f"Sources:\n\n{format_sources(sources)}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Answer from the sources above, citing [n] after every statement. "
        f"Reply {NOT_IN_NOTES} if they do not contain the answer."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def check_answer(text: str, source_count: int) -> AnswerReport:
    """Map citations to sources and list statements that break the rules."""
    stripped = text.strip()

    if NOT_IN_NOTES in stripped:
        return AnswerReport(not_in_notes=True, cited=(), out_of_range=(), uncited=())

    numbers = [int(n) for n in CITATION_RE.findall(stripped)]
    cited = tuple(sorted({n for n in numbers if 1 <= n <= source_count}))
    out_of_range = tuple(sorted({n for n in numbers if not 1 <= n <= source_count}))
    uncited = []

    for statement in STATEMENT_SPLIT_RE.split(stripped):
        statement = statement.strip()

        if not statement or statement.startswith(INFERENCE_LABEL):
            continue

        if not CITATION_RE.search(statement):
            uncited.append(statement)

    return AnswerReport(not_in_notes=False, cited=cited, out_of_range=out_of_range, uncited=tuple(uncited))
