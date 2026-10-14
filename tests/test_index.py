import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import HashingEmbedder, unreadable_file
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


class SymlinkTests(IndexTestCase):
    def test_symlinked_note_pointing_outside_the_vault_is_skipped(self) -> None:
        outside = self.tmp / "outside.md"
        outside.write_text("# Secret\n\nnot part of this vault\n", encoding="utf-8")
        (self.vault / "leak.md").symlink_to(outside)

        found = [path.name for path in iter_markdown_files(self.vault)]

        self.assertNotIn("leak.md", found)
        self.assertIn("deep-work.md", found)

    def test_symlinked_folder_pointing_outside_the_vault_is_skipped(self) -> None:
        outside = self.tmp / "elsewhere"
        outside.mkdir()
        (outside / "private.md").write_text("# Private\n\nsecret\n", encoding="utf-8")
        (self.vault / "linked").symlink_to(outside, target_is_directory=True)

        found = [path.name for path in iter_markdown_files(self.vault)]

        self.assertNotIn("private.md", found)


class PermissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_existing_loose_index_dir_and_file_are_tightened_on_open(self) -> None:
        index_dir = self.tmp / "index"
        index_dir.mkdir()
        db = index_dir / "test.sqlite"
        sqlite3.connect(db).close()
        os.chmod(index_dir, 0o775)
        os.chmod(db, 0o644)

        IndexStore(db).close()

        self.assertEqual(stat.S_IMODE(index_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(db.stat().st_mode), 0o600)

    def test_new_index_dir_is_private(self) -> None:
        IndexStore(self.tmp / "index" / "test.sqlite").close()

        self.assertEqual(stat.S_IMODE((self.tmp / "index").stat().st_mode), 0o700)


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

    def test_changed_embedding_dimension_raises(self) -> None:
        self.ingest()
        (self.vault / "reading" / "deep-work.md").write_text("# Changed\n\nnew text\n", encoding="utf-8")

        with self.assertRaises(StoreError) as caught:
            ingest_folder(self.store, HashingEmbedder(dimensions=128), self.vault, 1200)

        self.assertIn("256 -> 128", str(caught.exception))

    def test_chunk_size_is_recorded_and_a_change_requires_rebuild(self) -> None:
        self.ingest()
        self.assertEqual(self.store.get_meta("chunk_max_chars"), "1200")

        with self.assertRaises(StoreError) as caught:
            ingest_folder(self.store, self.embedder, self.vault, max_chars=800)

        self.assertIn("--rebuild", str(caught.exception))
        self.assertEqual(self.store.get_meta("chunk_max_chars"), "1200")

    def test_unreadable_note_is_reported_and_skipped(self) -> None:
        unreadable = self.vault / "reading" / "deep-work.md"
        messages: list[str] = []

        with unreadable_file(unreadable):
            report = ingest_folder(self.store, self.embedder, self.vault, 1200, messages.append)

        self.assertEqual(report.unreadable, ["reading/deep-work.md"])
        self.assertEqual(report.indexed, 4)
        self.assertIn("  skipped reading/deep-work.md: Permission denied", messages)
        self.assertNotIn("reading/deep-work.md", self.store.file_hashes())

    def test_unreadable_note_keeps_its_previous_index_entry(self) -> None:
        self.ingest()
        unreadable = self.vault / "reading" / "deep-work.md"

        with unreadable_file(unreadable):
            report = self.ingest()

        self.assertEqual(report.removed, 0)
        self.assertIn("reading/deep-work.md", self.store.file_hashes())

    def test_index_file_is_private_to_the_user(self) -> None:
        mode = stat.S_IMODE((self.tmp / "index" / "test.sqlite").stat().st_mode)

        self.assertEqual(mode, 0o600)

    def test_search_on_empty_index_returns_nothing(self) -> None:
        self.assertEqual(self.search("anything"), [])


if __name__ == "__main__":
    unittest.main()
