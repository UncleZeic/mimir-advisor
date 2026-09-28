import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import chunk_markdown, note_title, split_frontmatter

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vault"


class FrontmatterTests(unittest.TestCase):
    def test_frontmatter_is_removed_and_title_extracted(self) -> None:
        text = (FIXTURES / "projects" / "metis-decisions.md").read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(text)

        self.assertIn("tags:", frontmatter)
        self.assertNotIn("tags:", body)
        self.assertEqual(note_title(frontmatter, "fallback"), "Metis architecture decisions")

    def test_note_without_frontmatter_uses_fallback_title(self) -> None:
        frontmatter, body = split_frontmatter("# Just a note\n\nBody")

        self.assertEqual(frontmatter, "")
        self.assertEqual(note_title(frontmatter, "just-a-note"), "just-a-note")
        self.assertTrue(body.startswith("# Just a note"))


class ChunkMarkdownTests(unittest.TestCase):
    def test_sections_carry_their_heading_path(self) -> None:
        text = (FIXTURES / "projects" / "metis-decisions.md").read_text(encoding="utf-8")
        chunks = chunk_markdown(text)
        headings = [chunk.heading for chunk in chunks]

        self.assertEqual(
            headings,
            [
                "Metis architecture decisions › Storage",
                "Metis architecture decisions › Embeddings",
                "Metis architecture decisions › Chunking",
            ],
        )
        self.assertIn("sqlite-vec", chunks[0].text)
        self.assertNotIn("tags:", " ".join(chunk.text for chunk in chunks))

    def test_heading_levels_pop_correctly(self) -> None:
        text = "# A\n\n## B\n\nb text\n\n### C\n\nc text\n\n## D\n\nd text\n"
        headings = [chunk.heading for chunk in chunk_markdown(text)]

        self.assertEqual(headings, ["A › B", "A › B › C", "A › D"])

    def test_comment_inside_code_fence_is_not_a_heading(self) -> None:
        text = (FIXTURES / "projects" / "home-lab.md").read_text(encoding="utf-8")
        commands = [chunk for chunk in chunk_markdown(text) if chunk.heading.endswith("Commands")]

        self.assertEqual(len(commands), 1)
        self.assertIn("# check pool health", commands[0].text)
        self.assertIn("zpool status tank", commands[0].text)

    def test_long_section_splits_under_the_limit(self) -> None:
        text = (FIXTURES / "projects" / "home-lab.md").read_text(encoding="utf-8")
        chunks = chunk_markdown(text, max_chars=1200)
        zfs = [chunk for chunk in chunks if chunk.heading.endswith("ZFS pool")]

        self.assertGreater(len(zfs), 1)
        self.assertTrue(all(len(chunk.text) <= 1200 for chunk in chunks))
        self.assertEqual([chunk.ordinal for chunk in chunks], list(range(len(chunks))))

    def test_single_huge_paragraph_is_hard_split_with_overlap(self) -> None:
        words = [f"word{i}" for i in range(600)]
        chunks = chunk_markdown(" ".join(words), max_chars=500, overlap=100)
        joined = " ".join(chunk.text for chunk in chunks)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 500 for chunk in chunks))
        self.assertTrue(all(word in joined.split() for word in words))
        self.assertIn(chunks[1].text.split()[0], chunks[0].text.split())

    def test_romanian_diacritics_survive(self) -> None:
        text = (FIXTURES / "jurnal" / "2026-09-14-revizie-bicicleta.md").read_text(encoding="utf-8")
        chunks = chunk_markdown(text)

        self.assertEqual(chunks[0].heading, "Revizie bicicletă › Lanțul")
        self.assertIn("scârțâie", chunks[1].text)

    def test_empty_and_heading_only_notes_produce_no_chunks(self) -> None:
        self.assertEqual(chunk_markdown(""), [])
        self.assertEqual(chunk_markdown("# Title\n\n## Empty section\n"), [])
        self.assertEqual(chunk_markdown("---\ntitle: x\n---\n"), [])

    def test_windows_line_endings(self) -> None:
        chunks = chunk_markdown("# T\r\n\r\nline one\r\nline two\r\n")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "line one\nline two")


if __name__ == "__main__":
    unittest.main()
