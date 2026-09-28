import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import HashingEmbedder
from ingest import ingest_folder, iter_markdown_files
from store import IndexStore, StoreError, index_path_for

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vault"


class IndexTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.vault = self.tmp / "vault"
        shutil.copytree(FIXTURES, self.vault)
        self.store = IndexStore(self.tmp / "index" / "test.sqlite")
        self.embedder = HashingEmbedder()

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.tmp)

    def ingest(self):
        return ingest_folder(self.store, self.embedder, self.vault, max_chars=1200)

    def search(self, query: str, k: int = 5):
        return self.store.search(self.embedder.vector(query), k)


class DiscoveryTests(unittest.TestCase):
    def test_only_visible_markdown_files_are_found(self) -> None:
        found = [path.relative_to(FIXTURES).as_posix() for path in iter_markdown_files(FIXTURES)]

        self.assertEqual(
            found,
            [
                "empty.md",
                "jurnal/2026-09-14-revizie-bicicleta.md",
                "meetings/2026-09-10-kestrel-retro.md",
                "projects/home-lab.md",
                "projects/metis-decisions.md",
                "reading/deep-work.md",
            ],
        )

    def test_each_source_folder_gets_its_own_index(self) -> None:
        fixtures = index_path_for("./tests/fixtures/vault", "/idx")
        real = index_path_for("~/vault/tech", "/idx")

        self.assertNotEqual(fixtures, real)
        self.assertEqual(fixtures, index_path_for(str(FIXTURES), "/idx"))
        self.assertEqual(fixtures.parent, Path("/idx"))


class IngestTests(IndexTestCase):
    def test_ingest_indexes_fixture_vault(self) -> None:
        report = self.ingest()
        files, chunks = self.store.stats()

        self.assertEqual(report.indexed, 5)
        self.assertEqual(report.empty, ["empty.md"])
        self.assertEqual(files, 6)
        self.assertEqual(chunks, report.chunks)
        self.assertGreater(chunks, 10)

    def test_search_returns_the_right_note(self) -> None:
        self.ingest()

        self.assertEqual(self.search("sqlite-vec instead of ChromaDB")[0].path, "projects/metis-decisions.md")
        self.assertEqual(self.search("lanțul bicicletei uzura")[0].path, "jurnal/2026-09-14-revizie-bicicleta.md")
        self.assertEqual(self.search("Kestrel retro estimări")[0].path, "meetings/2026-09-10-kestrel-retro.md")

    def test_hits_carry_heading_text_and_score(self) -> None:
        self.ingest()
        hit = self.search("restic backups external drive rotated")[0]

        self.assertEqual(hit.path, "projects/home-lab.md")
        self.assertEqual(hit.heading, "Home lab build log › Backups")
        self.assertIn("restic", hit.text)
        self.assertGreater(hit.score, 0)
        self.assertLessEqual(hit.score, 1.0001)

    def test_reingest_skips_unchanged_files(self) -> None:
        self.ingest()
        calls_before = len(self.embedder.calls)
        report = self.ingest()

        self.assertEqual(report.indexed, 0)
        self.assertEqual(report.unchanged, 6)
        self.assertEqual(len(self.embedder.calls), calls_before)

    def test_changed_file_is_reindexed_without_duplicates(self) -> None:
        self.ingest()
        _, chunks_before = self.store.stats()
        note = self.vault / "reading" / "deep-work.md"
        note.write_text("# Notes on Deep Work\n\nOnly one paragraph now about pomodoro timers.\n", encoding="utf-8")
        report = self.ingest()
        _, chunks_after = self.store.stats()

        self.assertEqual(report.indexed, 1)
        self.assertEqual(chunks_after, chunks_before - 1)
        self.assertEqual(self.search("pomodoro timers")[0].path, "reading/deep-work.md")

    def test_deleted_file_is_removed_from_index(self) -> None:
        self.ingest()
        (self.vault / "reading" / "deep-work.md").unlink()
        report = self.ingest()
        paths = {hit.path for hit in self.search("focused work shutdown ritual", k=50)}

        self.assertEqual(report.removed, 1)
        self.assertNotIn("reading/deep-work.md", paths)

    def test_changing_embedding_model_requires_rebuild(self) -> None:
        self.ingest()

        with self.assertRaises(StoreError):
            ingest_folder(self.store, HashingEmbedder(model="other-model"), self.vault, 1200)

    def test_index_file_is_private_to_the_user(self) -> None:
        mode = stat.S_IMODE((self.tmp / "index" / "test.sqlite").stat().st_mode)

        self.assertEqual(mode, 0o600)

    def test_search_on_empty_index_returns_nothing(self) -> None:
        self.assertEqual(self.search("anything"), [])


if __name__ == "__main__":
    unittest.main()
