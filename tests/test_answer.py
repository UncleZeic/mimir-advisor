import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import answer
from store import SearchHit

HITS = [
    SearchHit("projects/metis.md", "Storage", "We chose sqlite-vec because it needs no server.\n", 0.81),
    SearchHit("jurnal/2026-01.md", "Notes", "Tried Chroma, it wanted a daemon.", 0.62),
    SearchHit("reading/list.md", "", "Unrelated book notes.", 0.12),
]


class SelectSourcesTests(unittest.TestCase):
    def test_hits_below_the_threshold_are_dropped_and_the_rest_renumbered(self) -> None:
        sources = answer.select_sources(HITS, 0.35)

        self.assertEqual([s.number for s in sources], [1, 2])
        self.assertEqual([s.hit.path for s in sources], ["projects/metis.md", "jurnal/2026-01.md"])

    def test_threshold_is_inclusive(self) -> None:
        self.assertEqual(len(answer.select_sources(HITS, 0.62)), 2)
        self.assertEqual(len(answer.select_sources(HITS, 0.63)), 1)

    def test_no_hits_or_all_below_gives_no_sources(self) -> None:
        self.assertEqual(answer.select_sources([], 0.0), [])
        self.assertEqual(answer.select_sources(HITS, 0.9), [])

    def test_location_omits_a_missing_heading(self) -> None:
        sources = answer.select_sources(HITS, 0.0)

        self.assertEqual(sources[0].location, "projects/metis.md › Storage")
        self.assertEqual(sources[2].location, "reading/list.md")


class BuildMessagesTests(unittest.TestCase):
    def test_sources_are_numbered_with_path_heading_and_text(self) -> None:
        messages = answer.build_messages("Why sqlite-vec?", answer.select_sources(HITS, 0.35))

        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        user = messages[1]["content"]
        self.assertIn("[1] projects/metis.md › Storage\nWe chose sqlite-vec because it needs no server.\n\n", user)
        self.assertIn("[2] jurnal/2026-01.md › Notes\nTried Chroma, it wanted a daemon.", user)
        self.assertNotIn("Unrelated", user)
        self.assertTrue(user.index("Sources:") < user.index("Question: Why sqlite-vec?"))

    def test_system_prompt_states_the_evidence_rules(self) -> None:
        system = answer.build_messages("q", answer.select_sources(HITS, 0.0))[0]["content"]

        self.assertIn("[1], [2]", system)
        self.assertIn(answer.INFERENCE_LABEL, system)
        self.assertIn(answer.NOT_IN_NOTES, system)
        self.assertIn("Never answer from general knowledge", system)

    def test_no_sources_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            answer.build_messages("q", [])


class CheckAnswerTests(unittest.TestCase):
    def test_citations_are_mapped_and_deduplicated(self) -> None:
        report = answer.check_answer("No server needed [1]. Chroma needed a daemon [2][1].", 2)

        self.assertFalse(report.not_in_notes)
        self.assertEqual(report.cited, (1, 2))
        self.assertEqual(report.out_of_range, ())
        self.assertEqual(report.uncited, ())

    def test_numbers_outside_the_sources_are_reported(self) -> None:
        report = answer.check_answer("Fact [3]. Other [0].", 2)

        self.assertEqual(report.cited, ())
        self.assertEqual(report.out_of_range, (0, 3))

    def test_statements_without_a_citation_are_listed(self) -> None:
        report = answer.check_answer("Cited claim [1]. Bare claim. Another bare one!\nLast line [2]", 2)

        self.assertEqual(report.uncited, ("Bare claim.", "Another bare one!"))

    def test_inference_lines_need_no_citation(self) -> None:
        report = answer.check_answer("Cited [1].\nInference: this probably saved setup time.", 1)

        self.assertEqual(report.uncited, ())
        self.assertEqual(report.cited, (1,))

    def test_not_in_notes_short_circuits(self) -> None:
        for text in ("NOT_IN_NOTES", "  NOT_IN_NOTES\n", "The answer is NOT_IN_NOTES [1]."):
            with self.subTest(text=text):
                report = answer.check_answer(text, 1)
                self.assertTrue(report.not_in_notes)
                self.assertEqual(report.cited, ())

    def test_empty_reply_has_nothing_to_flag(self) -> None:
        report = answer.check_answer("", 1)

        self.assertFalse(report.not_in_notes)
        self.assertEqual(report.uncited, ())


if __name__ == "__main__":
    unittest.main()
