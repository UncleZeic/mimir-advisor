"""Runs the real CLI and real HTTP code against a fake Ollama on localhost."""

import contextlib
import io
import json
import os
import shutil
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
from helpers import HashingEmbedder

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EMBEDDER = HashingEmbedder(model="bge-m3")


class FakeOllama(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send({"models": [{"name": "llama3:latest"}, {"name": "bge-m3:latest"}]})

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


if __name__ == "__main__":
    unittest.main()
