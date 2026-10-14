"""Runs the real CLI and real HTTP code against a fake Ollama on localhost."""

import contextlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main
from helpers import HashingEmbedder, unreadable_file
from store import index_path_for

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EMBEDDER = HashingEmbedder(model="bge-m3")


class FakeOllama(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send({"models": [{"name": "qwen3:8b"}, {"name": "bge-m3:latest"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"embeddings": [EMBEDDER.vector(text) for text in body["input"]]})

    def _send(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return None


class CliEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.env = {
            "OLLAMA_BASE_URL": f"http://127.0.0.1:{self.server.server_address[1]}",
            "DATA_SOURCE_DIR": str(FIXTURES / "vault"),
            "INDEX_DIR": self.tmp,
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def run_cli(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()

        with (
            patch.dict(os.environ, self.env, clear=True),
            patch.object(main, "load_dotenv"),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            code = main.main(list(args))

        return code, output.getvalue()

    def test_health_ingest_search_eval(self) -> None:
        code, out = self.run_cli("health")
        self.assertEqual(code, 0, out)
        self.assertIn("[OK] Embedding model", out)

        code, out = self.run_cli("ingest")
        self.assertEqual(code, 0, out)
        self.assertIn("Done: 5 indexed", out)
        self.assertIn("1 empty", out)

        code, out = self.run_cli("ingest")
        self.assertIn("0 indexed (0 chunks), 6 unchanged", out)

        code, out = self.run_cli("search", "sqlite-vec", "instead", "of", "Chroma")
        self.assertEqual(code, 0, out)
        self.assertIn("1. [", out)
        self.assertIn("projects/metis-decisions.md › Metis architecture decisions › Storage", out)

        code, out = self.run_cli("eval", str(FIXTURES / "eval.txt"))
        self.assertEqual(code, 0, out)
        self.assertIn("questions found their file in the top 5", out)

    def test_search_before_ingest_says_what_to_do(self) -> None:
        code, out = self.run_cli("search", "anything")

        self.assertEqual(code, 1)
        self.assertIn("Run ingest first", out)

    def test_missing_source_folder_fails_cleanly(self) -> None:
        self.env["DATA_SOURCE_DIR"] = str(Path(self.tmp) / "does-not-exist")
        code, out = self.run_cli("ingest")

        self.assertEqual(code, 1)
        self.assertIn("Source folder not found", out)

    def index_path(self) -> Path:
        return index_path_for(self.env["DATA_SOURCE_DIR"], self.env["INDEX_DIR"])

    def test_rebuild_deletes_and_recreates_the_index(self) -> None:
        self.run_cli("ingest")

        with sqlite3.connect(self.index_path()) as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES ('marker', 'old index')")
            conn.execute("INSERT INTO files(path, hash) VALUES ('stale.md', 'x')")

        code, out = self.run_cli("ingest", "--rebuild")

        self.assertEqual(code, 0, out)
        self.assertIn("Done: 5 indexed", out)
        self.assertIn("0 unchanged, 0 removed", out)
        self.assertIn("Index now holds 6 files", out)

        with sqlite3.connect(self.index_path()) as conn:
            self.assertIsNone(conn.execute("SELECT value FROM meta WHERE key = 'marker'").fetchone())

    def test_changed_chunk_size_requires_rebuild(self) -> None:
        self.run_cli("ingest")
        self.env["CHUNK_MAX_CHARS"] = "800"

        code, out = self.run_cli("ingest")
        self.assertEqual(code, 1)
        self.assertIn("Ingest stopped", out)
        self.assertIn("CHUNK_MAX_CHARS=1200", out)
        self.assertIn("--rebuild", out)

        code, out = self.run_cli("ingest", "--rebuild")
        self.assertEqual(code, 0, out)
        self.assertIn("Done: 5 indexed", out)

    def test_corrupt_index_fails_cleanly(self) -> None:
        self.index_path().write_bytes(b"this is not a sqlite database" * 100)

        code, out = self.run_cli("ingest")
        self.assertEqual(code, 1)
        self.assertIn("Ingest stopped", out)
        self.assertIn(str(self.index_path()), out)
        self.assertNotIn("Traceback", out)

        code, out = self.run_cli("search", "anything")
        self.assertEqual(code, 1)
        self.assertIn("Search failed", out)
        self.assertIn(str(self.index_path()), out)

        code, out = self.run_cli("eval", str(FIXTURES / "eval.txt"))
        self.assertEqual(code, 1)
        self.assertIn("Search failed", out)
        self.assertIn(str(self.index_path()), out)

    def test_unwritable_index_location_fails_cleanly(self) -> None:
        blocker = Path(self.tmp) / "not-a-dir"
        blocker.write_text("", encoding="utf-8")
        self.env["INDEX_DIR"] = str(blocker)

        code, out = self.run_cli("ingest")

        self.assertEqual(code, 1)
        self.assertIn("Ingest stopped", out)
        self.assertIn(str(blocker), out)

    def test_unreadable_note_is_reported_and_ingest_continues(self) -> None:
        note = FIXTURES / "vault" / "reading" / "deep-work.md"

        with unreadable_file(note):
            code, out = self.run_cli("ingest")

        self.assertEqual(code, 0, out)
        self.assertIn("skipped reading/deep-work.md: Permission denied", out)
        self.assertIn("Done: 4 indexed", out)
        self.assertIn("Skipped 1 unreadable: reading/deep-work.md", out)


if __name__ == "__main__":
    unittest.main()
