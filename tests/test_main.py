import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main


class UrlopenResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "UrlopenResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class SettingsTests(unittest.TestCase):
    def test_load_settings_uses_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(main, "load_dotenv"):
            settings = main.load_settings()

        self.assertEqual(settings.project_name, "mimir-advisor")
        self.assertEqual(settings.persona, "Adjutant_Mimir")
        self.assertEqual(settings.model_name, "llama3")
        self.assertEqual(settings.ollama_base_url, "http://localhost:11434")
        self.assertEqual(settings.vector_db_url, "http://localhost:8000")
        self.assertEqual(settings.data_source_dir, "./data/raw")
        self.assertEqual(settings.health_timeout_seconds, 2)

    def test_load_settings_uses_environment_values(self) -> None:
        env = {
            "PROJECT_NAME": "custom-advisor",
            "PERSONA": "Guide",
            "MODEL_NAME": "mistral",
            "OLLAMA_BASE_URL": "http://ollama:11434",
            "VECTOR_DB_URL": "http://db:8000",
            "DATA_SOURCE_DIR": "/data/raw",
            "HEALTH_TIMEOUT_SECONDS": "5",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(main, "load_dotenv"):
            settings = main.load_settings()

        self.assertEqual(settings.project_name, "custom-advisor")
        self.assertEqual(settings.persona, "Guide")
        self.assertEqual(settings.model_name, "mistral")
        self.assertEqual(settings.ollama_base_url, "http://ollama:11434")
        self.assertEqual(settings.vector_db_url, "http://db:8000")
        self.assertEqual(settings.data_source_dir, "/data/raw")
        self.assertEqual(settings.health_timeout_seconds, 5)


class HealthCheckTests(unittest.TestCase):
    def test_check_http_endpoint_returns_ok_for_successful_response(self) -> None:
        with patch.object(main, "urlopen", return_value=UrlopenResponse(200)):
            check = main.check_http_endpoint(
                "Ollama",
                ["http://localhost:11434/api/tags"],
                timeout=1,
            )

        self.assertTrue(check.ok)
        self.assertEqual(check.name, "Ollama")
        self.assertEqual(check.detail, "reachable")

    def test_check_http_endpoint_returns_failure_for_connection_error(self) -> None:
        with patch.object(main, "urlopen", side_effect=URLError("connection refused")):
            check = main.check_http_endpoint(
                "ChromaDB",
                ["http://localhost:8000/api/v2/heartbeat"],
                timeout=1,
            )

        self.assertFalse(check.ok)
        self.assertEqual(check.name, "ChromaDB")
        self.assertEqual(check.detail, "connection refused")

    def test_check_http_endpoint_tries_fallback_urls(self) -> None:
        with patch.object(
            main,
            "urlopen",
            side_effect=[URLError("not found"), UrlopenResponse(200)],
        ):
            check = main.check_http_endpoint(
                "ChromaDB",
                [
                    "http://localhost:8000/api/v2/heartbeat",
                    "http://localhost:8000/api/v1/heartbeat",
                ],
                timeout=1,
            )

        self.assertTrue(check.ok)
        self.assertEqual(check.url, "http://localhost:8000/api/v1/heartbeat")


class MainTests(unittest.TestCase):
    def test_main_returns_zero_when_services_are_ready(self) -> None:
        checks = [
            main.HealthCheck("Ollama", "http://ollama:11434/api/tags", True, "reachable"),
            main.HealthCheck("ChromaDB", "http://db:8000/api/v2/heartbeat", True, "reachable"),
        ]

        with (
            patch.object(main, "load_settings", return_value=self._settings()),
            patch.object(main, "run_health_checks", return_value=checks),
            patch.object(main, "print_startup_status"),
            patch("builtins.print"),
        ):
            exit_code = main.main()

        self.assertEqual(exit_code, 0)

    def test_main_returns_one_when_a_service_is_unavailable(self) -> None:
        checks = [
            main.HealthCheck("Ollama", "http://ollama:11434/api/tags", False, "connection refused"),
            main.HealthCheck("ChromaDB", "http://db:8000/api/v2/heartbeat", True, "reachable"),
        ]

        with (
            patch.object(main, "load_settings", return_value=self._settings()),
            patch.object(main, "run_health_checks", return_value=checks),
            patch.object(main, "print_startup_status"),
            patch("builtins.print"),
        ):
            exit_code = main.main()

        self.assertEqual(exit_code, 1)

    def _settings(self) -> main.Settings:
        return main.Settings(
            project_name="mimir-advisor",
            persona="Adjutant_Mimir",
            model_name="llama3",
            ollama_base_url="http://ollama:11434",
            vector_db_url="http://db:8000",
            data_source_dir="./data/raw",
            health_timeout_seconds=2,
        )


if __name__ == "__main__":
    unittest.main()
