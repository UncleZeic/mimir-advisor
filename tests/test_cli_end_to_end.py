"""Runs the real CLI and real HTTP code against a fake Ollama on localhost."""

import contextlib
import io
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_http
import main
from embeddings import EmbeddingError
from helpers import HashingEmbedder, unreadable_file, with_proxy_environment
from store import INDEX_SUBDIR, REPO_ROOT, index_path_for

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EMBEDDER = HashingEmbedder(model="bge-m3")


LOCAL_TAGS = [{"name": "gemma4:12b"}, {"name": "bge-m3:latest"}]
REMOTE = {"remote_model": "gpt-oss:120b", "remote_host": "https://ollama.com:443"}


class FakeOllama(BaseHTTPRequestHandler):
    """``tags`` is what /api/tags lists; ``posts`` records every POST path, so
    tests can prove no prompt or chunk was sent; ``embed_raw`` replaces the
    /api/embed reply with raw bytes."""
    tags: list[dict] = LOCAL_TAGS
    posts: list[str] = []
    embed_raw: bytes | None = None
    chat_bodies: list[dict] = []

    def do_GET(self) -> None:
        self._send({"models": FakeOllama.tags})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.posts.append(self.path)

        if self.path == "/api/embed" and FakeOllama.embed_raw is not None:
            self._send_raw(FakeOllama.embed_raw)
        elif self.path == "/api/chat":
            FakeOllama.chat_bodies.append(body)
            self._send({"message": {"content": "Local answer.\n"}})
        else:
            self._send({"embeddings": [EMBEDDER.vector(text) for text in body["input"]]})

    def _send(self, payload: dict) -> None:
        self._send_raw(json.dumps(payload).encode("utf-8"))

    def _send_raw(self, data: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return None


class RecordingProxy(BaseHTTPRequestHandler):
    """Stands in for a configured HTTP proxy; nothing should ever reach it."""
    requests: list[str] = []

    def do_GET(self) -> None:
        self._refuse()

    def do_POST(self) -> None:
        self._refuse()

    def _refuse(self) -> None:
        RecordingProxy.requests.append(self.path)
        self.send_response(502)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        return None


class RedirectingOllama(BaseHTTPRequestHandler):
    """Something on the Ollama port that bounces every request elsewhere."""
    target = ""

    def do_GET(self) -> None:
        self._redirect(302)

    def do_POST(self) -> None:
        self._redirect(303)

    def _redirect(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Location", f"{RedirectingOllama.target}{self.path}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        return None


class FailingEmbedder:
    model = "bge-m3"

    def __init__(self, error: BaseException = EmbeddingError("Ollama went away")) -> None:
        self.error = error

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise self.error


class CliEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.proxy = ThreadingHTTPServer(("127.0.0.1", 0), RecordingProxy)
        cls.redirector = ThreadingHTTPServer(("127.0.0.1", 0), RedirectingOllama)
        RedirectingOllama.target = f"http://127.0.0.1:{cls.proxy.server_address[1]}"

        for server in (cls.server, cls.proxy, cls.redirector):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.server, cls.proxy, cls.redirector):
            server.shutdown()
            server.server_close()

    def setUp(self) -> None:
        FakeOllama.tags = LOCAL_TAGS
        FakeOllama.posts = []
        FakeOllama.embed_raw = None
        FakeOllama.chat_bodies = []
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

    def test_failed_rebuild_keeps_the_old_index(self) -> None:
        self.run_cli("ingest")

        with sqlite3.connect(self.index_path()) as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES ('marker', 'old index')")

        with patch.object(main, "make_embedder", return_value=FailingEmbedder()):
            code, out = self.run_cli("ingest", "--rebuild")

        self.assertEqual(code, 1)
        self.assertIn("Ingest stopped: Ollama went away", out)
        self.assertEqual(self.index_files(), [self.index_path().name])

        with sqlite3.connect(self.index_path()) as conn:
            marker = conn.execute("SELECT value FROM meta WHERE key = 'marker'").fetchone()
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

        self.assertEqual(marker, ("old index",))
        self.assertEqual(files, 6)

        code, out = self.run_cli("search", "sqlite-vec", "instead", "of", "Chroma")
        self.assertEqual(code, 0, out)
        self.assertIn("projects/metis-decisions.md", out)

    def test_rebuild_clears_stale_temp_and_journal_files(self) -> None:
        self.run_cli("ingest")
        index = self.index_path()
        temp = index.with_name(index.name + ".tmp")
        temp.write_bytes(b"half-built index from a crashed rebuild" * 100)

        for path in (index, temp):
            for suffix in ("-journal", "-wal", "-shm"):
                Path(f"{path}{suffix}").write_bytes(b"stale")

        code, out = self.run_cli("ingest", "--rebuild")

        self.assertEqual(code, 0, out)
        self.assertIn("Index now holds 6 files", out)
        self.assertEqual(self.index_files(), [index.name])
        self.assertEqual(stat.S_IMODE(index.stat().st_mode), 0o600)

        code, out = self.run_cli("ingest")
        self.assertIn("6 unchanged", out)

    def test_index_lives_in_a_private_subfolder_of_index_dir(self) -> None:
        os.chmod(self.tmp, 0o755)
        code, out = self.run_cli("ingest")

        self.assertEqual(code, 0, out)
        self.assertEqual(self.index_path().parent, Path(self.tmp).resolve() / INDEX_SUBDIR)
        self.assertEqual(stat.S_IMODE(self.index_path().parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.index_path().stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(Path(self.tmp).stat().st_mode), 0o755)

    def test_index_dir_inside_the_repo_is_refused(self) -> None:
        inside = REPO_ROOT / "tests" / "index-that-must-not-exist"
        self.env["INDEX_DIR"] = str(inside)

        for command in (["ingest"], ["search", "anything"]):
            code, out = self.run_cli(*command)

            self.assertEqual(code, 1)
            self.assertIn("inside the repository", out)

        self.assertFalse(inside.exists())

    def test_index_dir_inside_the_notes_is_refused(self) -> None:
        notes = Path(self.tmp) / "notes"
        shutil.copytree(FIXTURES / "vault", notes)
        self.env["DATA_SOURCE_DIR"] = str(notes)
        self.env["INDEX_DIR"] = str(notes / ".index")

        code, out = self.run_cli("ingest")

        self.assertEqual(code, 1)
        self.assertIn("inside DATA_SOURCE_DIR", out)
        self.assertFalse((notes / ".index").exists())

    def test_configured_proxy_is_never_used(self) -> None:
        proxy = f"http://127.0.0.1:{self.proxy.server_address[1]}"
        self.env.update({"http_proxy": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "HTTPS_PROXY": proxy})
        RecordingProxy.requests.clear()

        with with_proxy_environment(proxy):
            for command in (["health"], ["ingest"], ["search", "backups"], ["chat", "hello"]):
                code, out = self.run_cli(*command)
                self.assertEqual(code, 0, out)

        self.assertEqual(RecordingProxy.requests, [])

    def test_redirects_are_never_followed(self) -> None:
        self.env["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{self.redirector.server_address[1]}"
        RecordingProxy.requests.clear()

        code, out = self.run_cli("health")
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] Ollama: HTTP 302", out)

        code, out = self.run_cli("ingest")
        self.assertEqual(code, 1)
        self.assertIn("Ingest stopped: Cannot list Ollama models: HTTP 302", out)

        code, out = self.run_cli("chat", "hello")
        self.assertEqual(code, 1)

        self.assertEqual(RecordingProxy.requests, [])

    def test_cloud_embedding_model_never_receives_a_chunk(self) -> None:
        FakeOllama.tags = [{"name": "gemma4:12b"}, {"name": "bge-m3:latest", **REMOTE}]

        code, out = self.run_cli("health")
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] Embedding model: refused: 'bge-m3' is served from https://ollama.com:443", out)

        code, out = self.run_cli("ingest")
        self.assertEqual(code, 1)
        self.assertIn("Ingest stopped: 'bge-m3' is served from https://ollama.com:443", out)

        code, out = self.run_cli("search", "anything")
        self.assertEqual(code, 1)

        self.assertEqual(FakeOllama.posts, [])

    def test_cloud_chat_model_never_receives_a_prompt(self) -> None:
        FakeOllama.tags = [{"name": "gemma4:12b", **REMOTE}, {"name": "bge-m3:latest"}]

        code, out = self.run_cli("chat", "my private question")

        self.assertEqual(code, 1)
        self.assertIn("Chat request failed: 'gemma4:12b' is served from https://ollama.com:443", out)
        self.assertEqual(FakeOllama.posts, [])

    def test_model_turned_remote_between_runs_is_caught(self) -> None:
        code, out = self.run_cli("ingest")
        self.assertEqual(code, 0, out)
        FakeOllama.posts.clear()
        FakeOllama.tags = [{"name": "gemma4:12b"}, {"name": "bge-m3:latest", **REMOTE}]

        code, out = self.run_cli("search", "sqlite-vec")

        self.assertEqual(code, 1)
        self.assertIn("served from https://ollama.com:443", out)
        self.assertEqual(FakeOllama.posts, [])

    def test_cloud_model_name_is_refused_before_any_request(self) -> None:
        self.env["EMBED_MODEL"] = "embeddinggemma:300m-cloud"

        code, out = self.run_cli("ingest")

        self.assertEqual(code, 2)
        self.assertIn("Invalid settings: EMBED_MODEL=embeddinggemma:300m-cloud is an Ollama cloud model", out)
        self.assertEqual(FakeOllama.posts, [])

    def test_malformed_embedding_reply_during_rebuild_is_clean(self) -> None:
        self.run_cli("ingest")
        FakeOllama.embed_raw = b"<html>502 Bad Gateway</html>"

        code, out = self.run_cli("ingest", "--rebuild")

        self.assertEqual(code, 1)
        self.assertIn("Ingest stopped: Ollama returned an invalid embedding response", out)
        self.assertNotIn("Traceback", out)
        self.assertEqual(self.index_files(), [self.index_path().name])

    def test_interrupted_rebuild_leaves_no_copy_of_the_notes(self) -> None:
        self.run_cli("ingest")

        for error in (KeyboardInterrupt(), ValueError("unexpected")):
            with self.subTest(error=type(error).__name__):
                with patch.object(main, "make_embedder", return_value=FailingEmbedder(error)):
                    with self.assertRaises(type(error)):
                        self.run_cli("ingest", "--rebuild")

                self.assertEqual(self.index_files(), [self.index_path().name])

        code, out = self.run_cli("ingest")
        self.assertIn("6 unchanged", out)

    def test_chat_sends_the_default_model_without_think(self) -> None:
        code, out = self.run_cli("chat", "hello")

        self.assertEqual(code, 0, out)
        self.assertEqual(out, "Local answer.\n")
        [body] = FakeOllama.chat_bodies
        self.assertEqual(body["model"], "gemma4:12b")
        self.assertNotIn("think", body)

    def test_remote_ollama_url_is_refused_before_any_request(self) -> None:
        # .invalid never resolves, and the opener is blocked as well, so even a
        # regressed guard cannot make this test reach another host.
        self.env["OLLAMA_BASE_URL"] = "http://ollama.invalid:11434"

        with patch.object(local_http.OPENER, "open", side_effect=AssertionError("network call")) as opener_open:
            code, out = self.run_cli("ingest")

        opener_open.assert_not_called()

        self.assertEqual(code, 2)
        self.assertIn("Invalid settings: OLLAMA_BASE_URL must point at a local Ollama", out)
        self.assertFalse((Path(self.tmp) / INDEX_SUBDIR).exists())

    def index_files(self) -> list[str]:
        return sorted(path.name for path in self.index_path().parent.iterdir())

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
        self.index_path().parent.mkdir(parents=True)
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
