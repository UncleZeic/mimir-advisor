import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import chunk_markdown, heading_text, note_title, split_frontmatter

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

    def test_empty_frontmatter_keeps_the_body(self) -> None:
        frontmatter, body = split_frontmatter("---\n---\n# Title\n\nBody text\n")

        self.assertEqual(frontmatter, "")
        self.assertEqual(body, "# Title\n\nBody text\n")
        self.assertEqual([chunk.text for chunk in chunk_markdown("---\n---\nBody text\n")], ["Body text"])

    def test_leading_rule_without_closing_line_is_body(self) -> None:
        text = "---\nFirst paragraph after a rule.\n\nSecond paragraph.\n"
        frontmatter, body = split_frontmatter(text)

        self.assertEqual(frontmatter, "")
        self.assertEqual(body, text)

    def test_leading_rule_plus_later_rule_loses_nothing(self) -> None:
        text = "---\nOpening thoughts on the week.\n\n---\n\nAfter the second rule.\n"
        frontmatter, body = split_frontmatter(text)
        joined = " ".join(chunk.text for chunk in chunk_markdown(text))

        self.assertEqual(frontmatter, "")
        self.assertEqual(body, text)
        self.assertIn("Opening thoughts on the week.", joined)
        self.assertIn("After the second rule.", joined)

    def test_closing_line_must_be_exactly_three_dashes(self) -> None:
        frontmatter, body = split_frontmatter("---\ntitle: x\n----\nstill yaml?\n---\nBody\n")

        self.assertEqual(frontmatter, "")
        self.assertTrue(body.startswith("---\ntitle: x"))

    def test_byte_order_mark_before_frontmatter_is_ignored(self) -> None:
        frontmatter, body = split_frontmatter("\ufeff---\ntitle: Boat log\n---\nBody\n")

        self.assertEqual(note_title(frontmatter, "fallback"), "Boat log")
        self.assertEqual(body, "Body\n")

    def test_title_quotes_and_empty_values(self) -> None:
        self.assertEqual(note_title('title: "Quoted"', "f"), "Quoted")
        self.assertEqual(note_title("title: 'Single'", "f"), "Single")
        self.assertEqual(note_title('title: Say "hi"', "f"), 'Say "hi"')
        self.assertEqual(note_title("title:", "f"), "f")
        self.assertEqual(note_title('title: ""\ntitle: Second', "f"), "Second")
        self.assertEqual(note_title("subtitle: no\n  title: nested", "f"), "f")

    def test_frontmatter_with_list_values(self) -> None:
        frontmatter, body = split_frontmatter("---\ntitle: x\ntags:\n  - a\n- b\n---\nBody\n")

        self.assertEqual(frontmatter, "title: x\ntags:\n  - a\n- b")
        self.assertEqual(body, "Body\n")


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

    def test_max_chars_not_above_overlap_raises_instead_of_looping(self) -> None:
        for max_chars in (0, 150):
            with self.subTest(max_chars=max_chars):
                with self.assertRaises(ValueError):
                    chunk_markdown("word " * 100, max_chars=max_chars, overlap=150)

    def test_shorter_fence_inside_a_longer_one_does_not_close_it(self) -> None:
        text = "# Doc\n\n````md\n```\n# not a heading\n```\n````\n\n## After\n\nafter text\n"
        chunks = chunk_markdown(text)

        self.assertEqual([chunk.heading for chunk in chunks], ["Doc", "Doc › After"])
        self.assertIn("# not a heading", chunks[0].text)

    def test_other_fence_character_does_not_close_the_fence(self) -> None:
        text = "# Doc\n\n```\n~~~\n# still code\n```\n\n## After\n\nafter text\n"
        chunks = chunk_markdown(text)

        self.assertEqual([chunk.heading for chunk in chunks], ["Doc", "Doc › After"])
        self.assertIn("# still code", chunks[0].text)

    def test_longer_closing_fence_closes(self) -> None:
        text = "# Doc\n\n~~~\ncode\n~~~~~\n\n## After\n\nafter text\n"

        self.assertEqual([chunk.heading for chunk in chunk_markdown(text)], ["Doc", "Doc › After"])

    def test_unclosed_fence_does_not_leak_into_the_next_note(self) -> None:
        unclosed = chunk_markdown("# Broken\n\n```\n# inside the fence\n")
        next_note = chunk_markdown("# Next\n\nbody\n\n## Section\n\nmore\n")

        self.assertEqual([chunk.heading for chunk in unclosed], ["Broken"])
        self.assertIn("# inside the fence", unclosed[0].text)
        self.assertEqual([chunk.heading for chunk in next_note], ["Next", "Next › Section"])

    def test_byte_order_mark_does_not_hide_the_first_heading(self) -> None:
        chunks = chunk_markdown("\ufeff# Title\n\nbody\n")

        self.assertEqual([(chunk.heading, chunk.text) for chunk in chunks], [("Title", "body")])

    def test_closing_hashes_are_stripped_only_after_a_space(self) -> None:
        self.assertEqual(heading_text("Title ##"), "Title")
        self.assertEqual(heading_text("Title ##   "), "Title")
        self.assertEqual(heading_text("C#"), "C#")
        self.assertEqual(heading_text("F# and C#"), "F# and C#")
        self.assertEqual(heading_text("##"), "##")
        self.assertEqual([c.heading for c in chunk_markdown("# Notes on C#\n\nbody\n")], ["Notes on C#"])

    def test_tabs_separate_heading_markers(self) -> None:
        self.assertEqual([c.heading for c in chunk_markdown("##\tTabbed\n\nbody\n")], ["Tabbed"])
        self.assertEqual([c.heading for c in chunk_markdown("#hashtag\n\nbody\n")], [""])

    def test_very_long_lines_are_parsed_in_linear_time(self) -> None:
        # The old heading pattern took ~10s on 2,000 spaces and grew cubically,
        # the old title pattern quadratically. Run in a subprocess so a
        # regression fails on the timeout instead of hanging the suite.
        script = (
            "import sys; sys.path.insert(0, sys.argv[1])\n"
            "from chunking import chunk_markdown, note_title, split_frontmatter\n"
            "pad = ' ' * 200_000\n"
            "text = f'---\\ntitle: a{pad}b\\n---\\n# a{pad}b\\n\\nbody\\n'\n"
            "assert chunk_markdown(text)[0].heading == f'a{pad}b'\n"
            "assert note_title(split_frontmatter(text)[0], 'f') == f'a{pad}b'\n"
        )
        src = str(Path(__file__).resolve().parents[1] / "src")

        try:
            result = subprocess.run([sys.executable, "-c", script, src], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            self.fail("parsing a 200,000-character line took over 10s")

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_windows_line_endings(self) -> None:
        chunks = chunk_markdown("# T\r\n\r\nline one\r\nline two\r\n")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "line one\nline two")


if __name__ == "__main__":
    unittest.main()
