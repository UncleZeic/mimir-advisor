import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import HashingEmbedder, unreadable_file
from ingest import embedding_text, ingest_folder, iter_markdown_files
import store
from store import (
    INDEX_SUBDIR,
    REPO_ROOT,
    IndexStore,
    StoreError,
    index_path_for,
    remove_sidecars,
    temporary_path,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vault"


class IndexTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.vault = self.tmp / "vault"
        shutil.copytree(FIXTURES, self.vault)
        self.db = self.tmp / "index" / INDEX_SUBDIR / "test.sqlite"
        self.store = IndexStore(self.db)
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
        self.assertEqual(fixtures.parent, Path("/idx") / INDEX_SUBDIR)


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
        index_dir = self.tmp / "index" / INDEX_SUBDIR
        index_dir.mkdir(parents=True)
        db = index_dir / "test.sqlite"
        sqlite3.connect(db).close()
        os.chmod(index_dir, 0o775)
        os.chmod(db, 0o644)

        IndexStore(db).close()

        self.assertEqual(stat.S_IMODE(index_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(db.stat().st_mode), 0o600)

    def test_new_index_dir_is_private(self) -> None:
        IndexStore(self.tmp / "index" / INDEX_SUBDIR / "test.sqlite").close()

        self.assertEqual(stat.S_IMODE((self.tmp / "index" / INDEX_SUBDIR).stat().st_mode), 0o700)

    def test_index_dir_itself_is_never_chmodded(self) -> None:
        index_dir = self.tmp / "shared"
        index_dir.mkdir()
        os.chmod(index_dir, 0o755)

        IndexStore(index_dir / INDEX_SUBDIR / "test.sqlite").close()

        self.assertEqual(stat.S_IMODE(index_dir.stat().st_mode), 0o755)

    def test_index_file_outside_the_dedicated_folder_is_refused(self) -> None:
        index_dir = self.tmp / "shared"
        index_dir.mkdir()
        os.chmod(index_dir, 0o755)

        with self.assertRaisesRegex(StoreError, INDEX_SUBDIR):
            IndexStore(index_dir / "test.sqlite")

        self.assertEqual(stat.S_IMODE(index_dir.stat().st_mode), 0o755)
        self.assertFalse((index_dir / "test.sqlite").exists())

    def test_db_file_is_private_before_sqlite_opens_it(self) -> None:
        db = self.tmp / "index" / INDEX_SUBDIR / "test.sqlite"
        modes: list[int] = []
        connect = sqlite3.connect

        def recording_connect(path, *args, **kwargs):
            modes.append(stat.S_IMODE(os.stat(path).st_mode))
            return connect(path, *args, **kwargs)

        old_umask = os.umask(0o022)

        try:
            with patch.object(store.sqlite3, "connect", recording_connect):
                IndexStore(db).close()
        finally:
            os.umask(old_umask)

        self.assertEqual(modes, [0o600])

    def test_secure_delete_is_on_even_where_sqlite_defaults_it_off(self) -> None:
        connect = sqlite3.connect

        def connect_with_secure_delete_off(path, *args, **kwargs):
            conn = connect(path, *args, **kwargs)
            conn.execute("PRAGMA secure_delete = OFF")
            return conn

        with patch.object(store.sqlite3, "connect", connect_with_secure_delete_off):
            index = IndexStore(self.tmp / "index" / INDEX_SUBDIR / "test.sqlite")

        try:
            self.assertEqual(index.conn.execute("PRAGMA secure_delete").fetchone(), (1,))
        finally:
            index.close()

    def test_removed_note_text_and_vectors_do_not_linger_in_the_file(self) -> None:
        vault = self.tmp / "vault"
        shutil.copytree(FIXTURES, vault)
        body = "\n\n".join(f"CANARY-{i} private paragraph, word{i} " * 8 for i in range(20))
        (vault / "secret.md").write_text("# Secret\n\n" + body, encoding="utf-8")
        db = self.tmp / "index" / INDEX_SUBDIR / "test.sqlite"
        embedder = HashingEmbedder()
        index = IndexStore(db)

        try:
            ingest_folder(index, embedder, vault, 1200)
            ids = [row[0] for row in index.conn.execute("SELECT id FROM chunks WHERE path = 'secret.md'")]
            vectors = [
                bytes(row[0])
                for row in index.conn.execute(
                    f"SELECT embedding FROM vec_chunks WHERE rowid IN ({','.join('?' * len(ids))})", ids
                )
            ]
            (vault / "secret.md").unlink()
            ingest_folder(index, embedder, vault, 1200)
        finally:
            index.close()

        raw = db.read_bytes()
        self.assertGreater(len(vectors), 1)
        self.assertNotIn(b"CANARY-", raw)
        self.assertFalse(any(vector in raw for vector in vectors))

    def test_temporary_data_stays_in_memory(self) -> None:
        index = IndexStore(self.tmp / "index" / INDEX_SUBDIR / "test.sqlite")

        try:
            self.assertEqual(index.conn.execute("PRAGMA temp_store").fetchone(), (2,))  # 2 = MEMORY
        finally:
            index.close()

    def test_remove_sidecars_leaves_the_database(self) -> None:
        db = self.tmp / "x.sqlite"
        db.write_bytes(b"db")

        for suffix in ("-journal", "-wal", "-shm"):
            Path(f"{db}{suffix}").write_bytes(b"stale")

        remove_sidecars(db)
        remove_sidecars(db)  # nothing left to remove is fine

        self.assertEqual(sorted(path.name for path in self.tmp.iterdir()), ["x.sqlite"])


class IndexSymlinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.index_dir = self.tmp / "index" / INDEX_SUBDIR
        self.db = self.index_dir / "test.sqlite"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_symlinked_index_subfolder_is_refused(self) -> None:
        target = self.tmp / "vault"
        target.mkdir()
        os.chmod(target, 0o755)
        self.index_dir.parent.mkdir()
        self.index_dir.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(StoreError, "is a symlink"):
            IndexStore(self.db)

        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)
        self.assertEqual(list(target.iterdir()), [])

    def test_dangling_index_subfolder_symlink_is_refused(self) -> None:
        target = self.tmp / "not-yet"
        self.index_dir.parent.mkdir()
        self.index_dir.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(StoreError, "is a symlink"):
            IndexStore(self.db)

        self.assertFalse(target.exists())

    def test_symlinked_index_file_is_refused(self) -> None:
        for db in (self.db, temporary_path(self.db)):
            with self.subTest(name=db.name):
                target = self.tmp / f"elsewhere-{db.name}"
                target.write_bytes(b"not an index")
                os.chmod(target, 0o644)
                self.index_dir.mkdir(parents=True, exist_ok=True)
                db.symlink_to(target)

                with self.assertRaisesRegex(StoreError, "is a symlink"):
                    IndexStore(db)

                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)
                self.assertEqual(target.read_bytes(), b"not an index")

    def test_dangling_index_file_symlink_is_refused(self) -> None:
        target = self.tmp / "created-through-link.sqlite"
        self.index_dir.mkdir(parents=True)
        self.db.symlink_to(target)

        with self.assertRaisesRegex(StoreError, "is a symlink"):
            IndexStore(self.db)

        self.assertFalse(target.exists())

    def test_file_swapped_for_a_symlink_after_the_checks_is_not_followed(self) -> None:
        target = self.tmp / "elsewhere.sqlite"
        target.write_bytes(b"")
        os.chmod(target, 0o644)
        self.index_dir.mkdir(parents=True)

        def swap_in_link(path: Path, label: str) -> None:
            self.db.symlink_to(target)

        with patch.object(store, "check_local_filesystem", swap_in_link):
            with self.assertRaises(OSError):
                IndexStore(self.db)

        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_folder_swapped_for_a_symlink_after_the_checks_is_not_followed(self) -> None:
        target = self.tmp / "vault"
        target.mkdir()
        os.chmod(target, 0o755)
        self.index_dir.mkdir(parents=True)

        def swap_in_link(path: Path, label: str) -> None:
            self.index_dir.rmdir()
            self.index_dir.symlink_to(target, target_is_directory=True)

        with patch.object(store, "check_local_filesystem", swap_in_link):
            with self.assertRaises(OSError):
                IndexStore(self.db)

        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)
        self.assertEqual(list(target.iterdir()), [])

    def test_symlinked_index_dir_above_the_subfolder_is_allowed(self) -> None:
        # INDEX_DIR itself may be a link (e.g. ~/.local/share on another disk);
        # index_path_for resolves and checks it.
        real = self.tmp / "real-index"
        real.mkdir()
        (self.tmp / "linked-index").symlink_to(real, target_is_directory=True)

        IndexStore(self.tmp / "linked-index" / INDEX_SUBDIR / "test.sqlite").close()

        self.assertTrue((real / INDEX_SUBDIR / "test.sqlite").is_file())


class IndexLocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_index_dir_inside_the_repo_is_refused(self) -> None:
        for index_dir in (REPO_ROOT, REPO_ROOT / "data" / "index"):
            with self.subTest(index_dir=index_dir):
                with self.assertRaisesRegex(StoreError, "inside the repository"):
                    index_path_for(str(self.tmp / "notes"), str(index_dir))

    def test_relative_index_dir_resolving_into_the_repo_is_refused(self) -> None:
        with self.assertRaisesRegex(StoreError, "inside the repository"):
            index_path_for(str(self.tmp / "notes"), str(REPO_ROOT / "tests" / ".." / "data"))

    def test_index_dir_inside_the_source_folder_is_refused(self) -> None:
        notes = self.tmp / "notes"

        for index_dir in (notes, notes / ".index"):
            with self.subTest(index_dir=index_dir):
                with self.assertRaisesRegex(StoreError, "inside DATA_SOURCE_DIR"):
                    index_path_for(str(notes), str(index_dir))

    def test_index_dir_holding_the_source_folder_is_refused(self) -> None:
        for index_dir in (self.tmp, self.tmp / "notes" / ".."):
            with self.subTest(index_dir=index_dir):
                with self.assertRaisesRegex(StoreError, r"DATA_SOURCE_DIR \(.*\) is inside INDEX_DIR"):
                    index_path_for(str(self.tmp / "notes" / "work"), str(index_dir))

    def test_index_dir_holding_the_repo_is_refused(self) -> None:
        for index_dir in (REPO_ROOT.parent, Path(REPO_ROOT.anchor)):
            with self.subTest(index_dir=index_dir):
                with self.assertRaisesRegex(StoreError, r"the repository \(.*\) is inside INDEX_DIR"):
                    index_path_for(str(self.tmp / "notes"), str(index_dir))

    def test_index_dir_next_to_the_source_folder_is_allowed(self) -> None:
        path = index_path_for(str(self.tmp / "notes"), str(self.tmp / "notes-index"))

        self.assertEqual(path.parent, self.tmp.resolve() / "notes-index" / INDEX_SUBDIR)


class RemoteFilesystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.mounts = self.tmp / "mounts"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def with_mounts(self, *entries: tuple[str, str]):
        lines = [f"dev {point} {fs_type} rw 0 0" for point, fs_type in entries]
        self.mounts.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return patch.object(store, "MOUNTS_FILE", str(self.mounts))

    def test_index_dir_on_a_network_or_fuse_filesystem_is_refused(self) -> None:
        for fs_type in ("nfs", "nfs4", "cifs", "smb3", "fuse.sshfs", "fuse.rclone", "fuse", "9p"):
            with self.subTest(fs_type=fs_type), self.with_mounts(("/", "ext4"), ("/mnt/share", fs_type)):
                with self.assertRaisesRegex(StoreError, f"on a {re.escape(fs_type)} filesystem"):
                    index_path_for("/notes", "/mnt/share/mimir/index")

    def test_index_dir_on_a_local_filesystem_is_allowed(self) -> None:
        for fs_type in ("ext4", "zfs", "btrfs", "xfs", "tmpfs", "overlay", "fuseblk"):
            with self.subTest(fs_type=fs_type), self.with_mounts(("/", "nfs"), ("/home", fs_type)):
                self.assertEqual(
                    index_path_for("/notes", "/home/me/.local/share/mimir-advisor/index").parent.name,
                    INDEX_SUBDIR,
                )

    def test_the_most_specific_mount_wins(self) -> None:
        with self.with_mounts(("/", "ext4"), ("/home", "nfs4"), ("/home/me/local", "ext4")):
            self.assertEqual(store.filesystem_type(Path("/home/me/local/x")), "ext4")
            self.assertEqual(store.filesystem_type(Path("/home/me/other")), "nfs4")
            self.assertEqual(store.filesystem_type(Path("/homeless")), "ext4")

    def test_later_mount_on_the_same_point_wins(self) -> None:
        with self.with_mounts(("/", "ext4"), ("/data", "ext4"), ("/data", "fuse.rclone")):
            self.assertEqual(store.filesystem_type(Path("/data/index")), "fuse.rclone")

    def test_escaped_spaces_in_mount_points(self) -> None:
        with self.with_mounts(("/", "ext4"), ("/media/My\\040Share", "cifs")):
            self.assertEqual(store.filesystem_type(Path("/media/My Share/index")), "cifs")

    def test_index_store_checks_the_resolved_location(self) -> None:
        # The link looks local; where it resolves to is a network share.
        share = self.tmp / "share"
        share.mkdir()
        (self.tmp / "local").symlink_to(share, target_is_directory=True)

        with self.with_mounts(("/", "ext4"), (str(share.resolve()), "nfs4")):
            with self.assertRaisesRegex(StoreError, "^Index file .* is on a nfs4 filesystem"):
                IndexStore(self.tmp / "local" / INDEX_SUBDIR / "test.sqlite")

        self.assertFalse((share / INDEX_SUBDIR).exists())

    def test_unreadable_mount_table_skips_the_check(self) -> None:
        with patch.object(store, "MOUNTS_FILE", str(self.tmp / "missing")):
            self.assertIsNone(store.filesystem_type(Path("/anywhere")))
            index_path_for("/notes", "/anywhere/index")


class EmbeddingTextTests(unittest.TestCase):
    def test_title_is_prefixed_to_a_heading_that_lacks_it(self) -> None:
        self.assertEqual(embedding_text("Metis", "Storage", "x"), "Metis › Storage\n\nx")

    def test_title_is_not_repeated_when_the_heading_path_starts_with_it(self) -> None:
        self.assertEqual(embedding_text("Metis", "Metis › Storage", "x"), "Metis › Storage\n\nx")
        self.assertEqual(embedding_text("Metis", "Metis", "x"), "Metis\n\nx")

    def test_heading_that_only_shares_a_prefix_keeps_the_title(self) -> None:
        self.assertEqual(embedding_text("Metis", "Metis notes", "x"), "Metis › Metis notes\n\nx")

    def test_note_without_headings_uses_the_title(self) -> None:
        self.assertEqual(embedding_text("Metis", "", "x"), "Metis\n\nx")


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

    def test_vectors_of_mixed_size_are_refused_before_anything_is_written(self) -> None:
        chunks = [("A", 0, "first"), ("A", 1, "second")]

        with self.assertRaisesRegex(StoreError, "4 -> 3"):
            self.store.replace_file("a.md", "h", chunks, [[1.0] * 4, [1.0] * 3], "fake-embed")

        self.assertEqual(self.store.stats(), (0, 0))
        self.assertIsNone(self.store.get_meta("dimensions"))

    def test_vector_not_matching_the_index_keeps_the_old_chunks(self) -> None:
        self.ingest()
        before = self.store.stats()
        chunks = [("A", 0, "first"), ("A", 1, "second")]

        with self.assertRaisesRegex(StoreError, "256 -> 255"):
            self.store.replace_file(
                "reading/deep-work.md", "h", chunks, [[1.0] * 256, [1.0] * 255], "fake-embed"
            )

        self.assertEqual(self.store.stats(), before)
        self.assertEqual(self.search("focused work shutdown ritual")[0].path, "reading/deep-work.md")

    def test_search_with_a_vector_of_the_wrong_size_raises(self) -> None:
        self.ingest()

        for size in (128, 257):
            with self.subTest(size=size):
                with self.assertRaisesRegex(StoreError, f"{size} dimensions, but this index has 256"):
                    self.store.search([1.0] * size, 5)

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
        mode = stat.S_IMODE(self.db.stat().st_mode)

        self.assertEqual(mode, 0o600)

    def test_embedded_text_does_not_repeat_the_title(self) -> None:
        self.ingest()
        texts = [text for call in self.embedder.calls for text in call]
        storage = next(text for text in texts if "sqlite-vec" in text)

        self.assertTrue(storage.startswith("Metis architecture decisions › Storage\n\n"), storage)
        self.assertFalse(any(text.startswith("Metis architecture decisions › Metis") for text in texts))

    def test_note_with_byte_order_mark_keeps_frontmatter_and_headings(self) -> None:
        note = self.vault / "bom.md"
        note.write_bytes(b"\xef\xbb\xbf---\ntitle: Boat log\n---\n# Engine\n\nOil changed at 200 hours.\n")
        self.ingest()
        texts = [text for call in self.embedder.calls for text in call]
        hit = self.search("oil changed engine hours")[0]

        self.assertEqual(hit.path, "bom.md")
        self.assertEqual(hit.heading, "Engine")
        self.assertEqual(hit.text, "Oil changed at 200 hours.")
        self.assertIn("Boat log › Engine\n\nOil changed at 200 hours.", texts)
        self.assertFalse(any("\ufeff" in text or "title:" in text for text in texts))

    def test_search_on_empty_index_returns_nothing(self) -> None:
        self.assertEqual(self.search("anything"), [])


if __name__ == "__main__":
    unittest.main()
