import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import embeddings
from embeddings import EmbeddingError, OllamaEmbedder


class Response:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def fake_embed(request, timeout):
    texts = json.loads(request.data.decode("utf-8"))["input"]
    return Response({"embeddings": [[float(len(text)), 1.0] for text in texts]})


class OllamaEmbedderTests(unittest.TestCase):
    def test_embeds_in_batches_and_keeps_order(self) -> None:
        embedder = OllamaEmbedder("http://localhost:11434/", "bge-m3", timeout=5, batch_size=2)

        with patch.object(embeddings, "urlopen", side_effect=fake_embed) as urlopen:
            vectors = embedder.embed(["a", "bb", "ccc"])

        self.assertEqual(vectors, [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        self.assertEqual(urlopen.call_count, 2)
        request = urlopen.call_args_list[0].args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/embed")
        self.assertEqual(json.loads(request.data)["model"], "bge-m3")

    def test_missing_model_tells_you_how_to_pull_it(self) -> None:
        error = HTTPError("u", 404, "Not Found", {}, io.BytesIO(b'{"error":"model \\"bge-m3\\" not found"}'))

        with patch.object(embeddings, "urlopen", side_effect=error):
            with self.assertRaisesRegex(EmbeddingError, "ollama pull bge-m3"):
                OllamaEmbedder("http://localhost:11434", "bge-m3", 5).embed(["x"])

    def test_unreachable_ollama_is_a_clear_error(self) -> None:
        with patch.object(embeddings, "urlopen", side_effect=URLError("connection refused")):
            with self.assertRaisesRegex(EmbeddingError, "Cannot reach Ollama"):
                OllamaEmbedder("http://localhost:11434", "bge-m3", 5).embed(["x"])

    def test_wrong_number_of_vectors_is_rejected(self) -> None:
        with patch.object(embeddings, "urlopen", return_value=Response({"embeddings": [[1.0]]})):
            with self.assertRaises(EmbeddingError):
                OllamaEmbedder("http://localhost:11434", "bge-m3", 5).embed(["x", "y"])


if __name__ == "__main__":
    unittest.main()
