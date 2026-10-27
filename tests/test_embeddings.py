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

# Port 9 (discard): if a patch were ever missed, the call fails instead of
# reaching a real Ollama on 11434.
BASE_URL = "http://127.0.0.1:9"
LOCAL_TAGS = {"models": [{"name": "bge-m3:latest"}]}


class Response:
    def __init__(self, body: object = None, raw: bytes | None = None) -> None:
        self.body = raw if raw is not None else json.dumps(body).encode("utf-8")

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def fake_ollama(tags: dict = LOCAL_TAGS):
    """/api/tags answers with ``tags``; /api/embed with one vector per text."""
    def respond(request, timeout):
        if request.full_url.endswith("/api/tags"):
            return Response(tags)

        texts = json.loads(request.data.decode("utf-8"))["input"]
        return Response({"embeddings": [[float(len(text)), 1.0] for text in texts]})

    return respond


def tags_then(*responses):
    return [Response(LOCAL_TAGS), *responses]


class OllamaEmbedderTests(unittest.TestCase):
    def test_embeds_in_batches_and_keeps_order(self) -> None:
        embedder = OllamaEmbedder(f"{BASE_URL}/", "bge-m3", timeout=5, batch_size=2)

        with patch.object(embeddings, "urlopen", side_effect=fake_ollama()) as urlopen:
            vectors = embedder.embed(["a", "bb", "ccc"])

        self.assertEqual(vectors, [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        urls = [call.args[0].full_url for call in urlopen.call_args_list]
        self.assertEqual(urls, [f"{BASE_URL}/api/tags", f"{BASE_URL}/api/embed", f"{BASE_URL}/api/embed"])
        self.assertEqual(json.loads(urlopen.call_args_list[1].args[0].data)["model"], "bge-m3")

    def test_missing_model_tells_you_how_to_pull_it(self) -> None:
        error = HTTPError("u", 404, "Not Found", {}, io.BytesIO(b'{"error":"model \\"bge-m3\\" not found"}'))

        with patch.object(embeddings, "urlopen", side_effect=tags_then(error)):
            with self.assertRaisesRegex(EmbeddingError, "ollama pull bge-m3"):
                OllamaEmbedder(BASE_URL, "bge-m3", 5).embed(["x"])

    def test_unreachable_ollama_is_a_clear_error(self) -> None:
        with patch.object(embeddings, "urlopen", side_effect=URLError("connection refused")):
            with self.assertRaisesRegex(EmbeddingError, "Cannot reach Ollama"):
                OllamaEmbedder(BASE_URL, "bge-m3", 5).embed(["x"])

    def test_wrong_number_of_vectors_is_rejected(self) -> None:
        with patch.object(embeddings, "urlopen", side_effect=tags_then(Response({"embeddings": [[1.0]]}))):
            with self.assertRaises(EmbeddingError):
                OllamaEmbedder(BASE_URL, "bge-m3", 5).embed(["x", "y"])

    def test_malformed_json_is_an_embedding_error(self) -> None:
        for raw in (b"not json", b"\xff\xfe", b"[1, 2]"):
            with self.subTest(raw=raw):
                with patch.object(embeddings, "urlopen", side_effect=tags_then(Response(raw=raw))):
                    with self.assertRaisesRegex(EmbeddingError, "invalid|unexpected"):
                        OllamaEmbedder(BASE_URL, "bge-m3", 5).embed(["x"])

    def test_malformed_model_list_is_an_embedding_error(self) -> None:
        with patch.object(embeddings, "urlopen", return_value=Response(raw=b"<html>")):
            with self.assertRaisesRegex(EmbeddingError, "invalid model list"):
                OllamaEmbedder(BASE_URL, "bge-m3", 5).embed(["x"])


class CloudModelTests(unittest.TestCase):
    def test_cloud_model_name_is_refused_before_any_request(self) -> None:
        with patch.object(embeddings, "urlopen") as urlopen:
            with self.assertRaisesRegex(EmbeddingError, "cloud model"):
                OllamaEmbedder(BASE_URL, "embeddinggemma:300m-cloud", 5).embed(["private note"])

        urlopen.assert_not_called()

    def test_model_listed_with_a_remote_host_is_refused_before_embedding(self) -> None:
        # A cloud model copied under a local-looking name keeps remote_host.
        tags = {"models": [{"name": "bge-m3:latest", "remote_model": "x", "remote_host": "https://ollama.com:443"}]}

        with patch.object(embeddings, "urlopen", side_effect=fake_ollama(tags)) as urlopen:
            with self.assertRaisesRegex(EmbeddingError, "served from https://ollama.com:443"):
                OllamaEmbedder(BASE_URL, "bge-m3", 5).embed(["private note"])

        urls = [call.args[0].full_url for call in urlopen.call_args_list]
        self.assertEqual(urls, [f"{BASE_URL}/api/tags"])

    def test_model_is_checked_again_on_every_embed_call(self) -> None:
        tags = {"models": [{"name": "bge-m3:latest"}]}
        embedder = OllamaEmbedder(BASE_URL, "bge-m3", 5)

        with patch.object(embeddings, "urlopen", side_effect=fake_ollama(tags)):
            embedder.embed(["first"])
            tags["models"][0]["remote_host"] = "https://ollama.com:443"

            with self.assertRaises(EmbeddingError):
                embedder.embed(["second"])


if __name__ == "__main__":
    unittest.main()
