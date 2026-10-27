import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ollama_models import (
    RemoteModelError,
    ensure_local_model,
    fetch_tags,
    find_entry,
    is_cloud_name,
    is_remote_entry,
    remote_reason,
)

BASE_URL = "http://127.0.0.1:9"


class Response:
    def __init__(self, body: object) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def serving(tags: dict):
    calls: list[str] = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        return Response(tags)

    urlopen.calls = calls
    return urlopen


class CloudNameTests(unittest.TestCase):
    def test_cloud_tags_are_recognised(self) -> None:
        for name in ("gpt-oss:120b-cloud", "qwen3-coder:480b-cloud", "deepseek-v3.1:671b-cloud",
                     "kimi-k2:cloud", "GPT-OSS:20B-CLOUD", "something-cloud", " x:y-cloud "):
            with self.subTest(name=name):
                self.assertTrue(is_cloud_name(name))

    def test_local_names_are_not_cloud(self) -> None:
        for name in ("qwen3:8b", "bge-m3", "bge-m3:latest", "cloudy:7b", "soundcloud-ish:1b", "wordcloud"):
            with self.subTest(name=name):
                self.assertFalse(is_cloud_name(name))


class RemoteEntryTests(unittest.TestCase):
    def test_remote_fields_mark_an_entry_remote_whatever_its_name(self) -> None:
        self.assertTrue(is_remote_entry({"name": "my-local-looking:latest", "remote_host": "https://ollama.com:443"}))
        self.assertTrue(is_remote_entry({"name": "renamed", "remote_model": "gpt-oss:120b"}))
        self.assertTrue(is_remote_entry({"name": "gpt-oss:120b-cloud"}))
        self.assertFalse(is_remote_entry({"name": "qwen3:8b", "remote_host": ""}))

    def test_find_entry_matches_exact_names_and_latest(self) -> None:
        entries = [{"name": "llama3:latest"}, {"name": "qwen2.5:7b"}]

        self.assertEqual(find_entry("llama3", entries), {"name": "llama3:latest"})
        self.assertEqual(find_entry("qwen2.5:7b", entries), {"name": "qwen2.5:7b"})
        self.assertIsNone(find_entry("qwen2.5", entries))

    def test_remote_reason(self) -> None:
        self.assertIsNone(remote_reason("qwen3:8b", {"name": "qwen3:8b"}))
        self.assertIsNone(remote_reason("qwen3:8b", None))
        self.assertIn("ollama.com", remote_reason("gpt-oss:120b-cloud", None))
        self.assertIn("https://h:443", remote_reason("x", {"name": "x:latest", "remote_host": "https://h:443"}))


class EnsureLocalModelTests(unittest.TestCase):
    def test_local_model_passes_after_listing_tags(self) -> None:
        urlopen = serving({"models": [{"name": "qwen3:8b"}]})

        ensure_local_model(BASE_URL, "qwen3:8b", 2, urlopen)

        self.assertEqual(urlopen.calls, [f"{BASE_URL}/api/tags"])

    def test_cloud_name_is_refused_without_any_request(self) -> None:
        urlopen = serving({"models": []})

        with self.assertRaisesRegex(RemoteModelError, "cloud model"):
            ensure_local_model(BASE_URL, "gpt-oss:120b-cloud", 2, urlopen)

        self.assertEqual(urlopen.calls, [])

    def test_renamed_cloud_model_is_refused(self) -> None:
        urlopen = serving({"models": [{"name": "assistant:latest", "remote_host": "https://ollama.com:443"}]})

        with self.assertRaisesRegex(RemoteModelError, "never sent off this machine"):
            ensure_local_model(BASE_URL, "assistant", 2, urlopen)

    def test_missing_model_is_left_to_ollama_to_report(self) -> None:
        ensure_local_model(BASE_URL, "qwen3:8b", 2, serving({"models": []}))

    def test_odd_tag_bodies_do_not_crash(self) -> None:
        for body in ([], {"models": None}, {"models": ["x", 3, {"name": "qwen3:8b"}]}):
            with self.subTest(body=body):
                with patch("ollama_models.json.loads", return_value=body):
                    entries = fetch_tags(BASE_URL, 2, serving({}))

                self.assertTrue(all(isinstance(entry, dict) for entry in entries))


if __name__ == "__main__":
    unittest.main()
