import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main


class UrlopenResponse:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self.body = body

    def __enter__(self) -> "UrlopenResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


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
            "OLLAMA_CHAT_TIMEOUT_SECONDS": "30",
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
        self.assertEqual(settings.chat_timeout_seconds, 30)


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
            exit_code = main.main([])

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
            exit_code = main.main([])

        self.assertEqual(exit_code, 1)

    def test_main_routes_chat_command_to_chat_runner(self) -> None:
        with (
            patch.object(main, "load_settings", return_value=self._settings()),
            patch.object(main, "run_chat_command", return_value=0) as run_chat_command,
        ):
            exit_code = main.main(["chat", "hello", "there"])

        self.assertEqual(exit_code, 0)
        run_chat_command.assert_called_once_with(self._settings(), "hello there")

    def _settings(self) -> main.Settings:
        return main.Settings(
            project_name="mimir-advisor",
            persona="Adjutant_Mimir",
            model_name="llama3",
            ollama_base_url="http://ollama:11434",
            vector_db_url="http://db:8000",
            data_source_dir="./data/raw",
            health_timeout_seconds=2,
            chat_timeout_seconds=60,
        )


class ChatTests(unittest.TestCase):
    def test_ask_ollama_sends_prompt_and_returns_message_content(self) -> None:
        response = UrlopenResponse(
            200,
            b'{"message": {"content": "No source data is available yet."}}',
        )

        with patch.object(main, "urlopen", return_value=response) as urlopen:
            answer = main.ask_ollama(self._settings(), "What do you know?")

        request = urlopen.call_args.args[0]
        payload = json_from_request(request)

        self.assertEqual(answer, "No source data is available yet.")
        self.assertEqual(request.full_url, "http://ollama:11434/api/chat")
        self.assertEqual(payload["model"], "llama3")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "What do you know?"})

    def test_run_chat_command_returns_one_when_ollama_is_unavailable(self) -> None:
        with (
            patch.object(
                main,
                "check_http_endpoint",
                return_value=main.HealthCheck("Ollama", "http://ollama:11434/api/tags", False, "down"),
            ),
            patch("builtins.print"),
        ):
            exit_code = main.run_chat_command(self._settings(), "hello")

        self.assertEqual(exit_code, 1)

    def test_run_chat_command_sends_one_shot_prompt(self) -> None:
        with (
            patch.object(
                main,
                "check_http_endpoint",
                return_value=main.HealthCheck("Ollama", "http://ollama:11434/api/tags", True, "reachable"),
            ),
            patch.object(main, "ask_ollama", return_value="hello back") as ask_ollama,
            patch("builtins.print") as print_output,
        ):
            exit_code = main.run_chat_command(self._settings(), "hello")

        self.assertEqual(exit_code, 0)
        ask_ollama.assert_called_once_with(self._settings(), "hello")
        print_output.assert_called_once_with("hello back")

    def test_run_chat_command_handles_one_shot_request_failure(self) -> None:
        with (
            patch.object(
                main,
                "check_http_endpoint",
                return_value=main.HealthCheck("Ollama", "http://ollama:11434/api/tags", True, "reachable"),
            ),
            patch.object(main, "ask_ollama", side_effect=URLError("connection lost")),
            patch("builtins.print"),
        ):
            exit_code = main.run_chat_command(self._settings(), "hello")

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
            chat_timeout_seconds=60,
        )


def json_from_request(request: object) -> dict[str, object]:
    return main.json.loads(request.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
