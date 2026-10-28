import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import embeddings
import local_http
import main
from helpers import with_proxy_environment

ROOT = Path(__file__).resolve().parents[1]
TAGS_URL = "http://localhost:11434/api/tags"


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
        self.assertEqual(settings.model_name, "gemma4:12b")
        self.assertEqual(settings.embed_model, "bge-m3")
        self.assertEqual(settings.ollama_base_url, "http://localhost:11434")
        self.assertEqual(settings.data_source_dir, "./tests/fixtures/vault")
        self.assertEqual(settings.index_dir, "~/.local/share/mimir-advisor/index")
        self.assertEqual(settings.chunk_max_chars, 1200)
        self.assertEqual(settings.health_timeout_seconds, 2)

    def test_load_settings_uses_environment_values(self) -> None:
        env = {
            "PROJECT_NAME": "custom-advisor",
            "PERSONA": "Guide",
            "MODEL_NAME": "mistral",
            "EMBED_MODEL": "nomic-embed-text",
            "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
            "DATA_SOURCE_DIR": "/data/raw",
            "INDEX_DIR": "/data/index",
            "CHUNK_MAX_CHARS": "800",
            "HEALTH_TIMEOUT_SECONDS": "5",
            "OLLAMA_CHAT_TIMEOUT_SECONDS": "30",
            "OLLAMA_EMBED_TIMEOUT_SECONDS": "90",
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(main, "load_dotenv"),
            patch.object(main, "running_in_container", return_value=True),
            patch.object(main, "hosts_file_address", return_value="172.17.0.1"),
        ):
            settings = main.load_settings()

        self.assertEqual(settings.project_name, "custom-advisor")
        self.assertEqual(settings.persona, "Guide")
        self.assertEqual(settings.model_name, "mistral")
        self.assertEqual(settings.embed_model, "nomic-embed-text")
        self.assertEqual(settings.ollama_base_url, "http://host.docker.internal:11434")
        self.assertEqual(settings.data_source_dir, "/data/raw")
        self.assertEqual(settings.index_dir, "/data/index")
        self.assertEqual(settings.chunk_max_chars, 800)
        self.assertEqual(settings.health_timeout_seconds, 5)
        self.assertEqual(settings.chat_timeout_seconds, 30)
        self.assertEqual(settings.embed_timeout_seconds, 90)

    def test_index_dir_follows_xdg_data_home(self) -> None:
        with patch.dict(os.environ, {"XDG_DATA_HOME": "/xdg"}, clear=True), patch.object(main, "load_dotenv"):
            settings = main.load_settings()

        self.assertEqual(settings.index_dir, "/xdg/mimir-advisor/index")


    def test_empty_index_dir_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"INDEX_DIR": ""}, clear=True), patch.object(main, "load_dotenv"):
            settings = main.load_settings()

        self.assertEqual(settings.index_dir, "~/.local/share/mimir-advisor/index")

    def test_local_ollama_hosts_are_accepted(self) -> None:
        for url in (
            "http://localhost:11434",
            "http://127.0.0.1:11434/",
            "http://[::1]:11434",
            "https://LOCALHOST:11434",
        ):
            with self.subTest(url=url):
                self.assertEqual(load_with({"OLLAMA_BASE_URL": url}).ollama_base_url, url)

    def test_host_docker_internal_is_accepted_only_inside_a_container(self) -> None:
        env = {"OLLAMA_BASE_URL": "http://host.docker.internal:11434"}

        with with_hosts_file("172.17.0.1 host.docker.internal\n"):
            with patch.object(main, "running_in_container", return_value=True):
                self.assertEqual(load_with(env).ollama_base_url, env["OLLAMA_BASE_URL"])

            with patch.object(main, "running_in_container", return_value=False):
                with self.assertRaisesRegex(main.SettingsError, "only resolves safely inside Docker"):
                    load_with(env)

    def test_host_docker_internal_must_come_from_the_hosts_file(self) -> None:
        env = {"OLLAMA_BASE_URL": "http://host.docker.internal:11434"}
        refused = (
            "127.0.0.1 localhost\n",                      # plain docker run: not mapped, DNS would answer
            "# 172.17.0.1 host.docker.internal\n",        # commented out
            "8.8.8.8 host.docker.internal\n",             # mapped, but to a public address
            "not-an-ip host.docker.internal\n",
        )

        with patch.object(main, "running_in_container", return_value=True):
            for hosts in refused:
                with self.subTest(hosts=hosts), with_hosts_file(hosts):
                    with self.assertRaisesRegex(main.SettingsError, "--add-host=host.docker.internal:host-gateway"):
                        load_with(env)

            for hosts in (
                "172.17.0.1\thost.docker.internal\n",
                "192.168.65.254 gateway HOST.DOCKER.INTERNAL # Docker Desktop\n",
                "fe80::1 host.docker.internal\n",
            ):
                with self.subTest(hosts=hosts), with_hosts_file(hosts):
                    self.assertEqual(load_with(env).ollama_base_url, env["OLLAMA_BASE_URL"])

    def test_missing_hosts_file_refuses_host_docker_internal(self) -> None:
        with (
            patch.object(main, "running_in_container", return_value=True),
            patch.object(main, "HOSTS_FILE", "/nonexistent/hosts"),
        ):
            with self.assertRaises(main.SettingsError):
                load_with({"OLLAMA_BASE_URL": "http://host.docker.internal:11434"})

    def test_cloud_models_are_refused_in_settings(self) -> None:
        for variable, model in (("MODEL_NAME", "gpt-oss:120b-cloud"), ("EMBED_MODEL", "embeddinggemma:cloud")):
            with self.subTest(variable=variable):
                with self.assertRaisesRegex(main.SettingsError, f"{variable}={model} is an Ollama cloud model"):
                    load_with({variable: model})

    def test_only_the_repo_env_file_is_loaded(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(main, "load_dotenv") as load_dotenv:
            main.load_settings()

        load_dotenv.assert_called_once_with(ROOT / ".env")
        self.assertEqual(main.DOTENV_PATH, ROOT / ".env")

    def test_fallback_env_loader_reads_only_the_given_file(self) -> None:
        source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")

        self.assertIn("def load_dotenv(env_path) -> None:", source)
        self.assertNotIn('env_path = ".env"', source)

    def test_running_in_container_checks_docker_and_podman_markers(self) -> None:
        for present, expected in (({"/.dockerenv"}, True), ({"/run/.containerenv"}, True), (set(), False)):
            with self.subTest(present=present):
                with patch.object(main.os.path, "exists", side_effect=lambda path: path in present):
                    self.assertIs(main.running_in_container(), expected)

    def test_non_local_ollama_hosts_are_rejected(self) -> None:
        for url in (
            "http://ollama:11434",
            "http://192.168.1.10:11434",
            "http://localhost.ollama.invalid:11434",
            "http://user@ollama.invalid:11434",
            "ftp://localhost:11434",
            "localhost:11434",
            "",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(main.SettingsError, "OLLAMA_BASE_URL must point at a local Ollama"):
                    load_with({"OLLAMA_BASE_URL": url})

    def test_chunk_max_chars_zero_raises_instead_of_looping(self) -> None:
        with self.assertRaisesRegex(main.SettingsError, "CHUNK_MAX_CHARS must be at least 300"):
            load_with({"CHUNK_MAX_CHARS": "0"})

    def test_chunk_max_chars_minimum(self) -> None:
        with self.assertRaises(main.SettingsError):
            load_with({"CHUNK_MAX_CHARS": "299"})

        self.assertEqual(load_with({"CHUNK_MAX_CHARS": "300"}).chunk_max_chars, 300)

    def test_malformed_numbers_name_the_variable(self) -> None:
        for variable, value in (
            ("CHUNK_MAX_CHARS", "large"),
            ("CHUNK_MAX_CHARS", "1200.5"),
            ("CHUNK_MAX_CHARS", ""),
            ("HEALTH_TIMEOUT_SECONDS", "2s"),
            ("OLLAMA_CHAT_TIMEOUT_SECONDS", "one minute"),
            ("OLLAMA_EMBED_TIMEOUT_SECONDS", ""),
        ):
            with self.subTest(variable=variable, value=value):
                with self.assertRaisesRegex(main.SettingsError, f"^{variable} must be a (whole )?number, got {value!r}"):
                    load_with({variable: value})

    def test_timeouts_must_be_finite_and_positive(self) -> None:
        for variable in ("HEALTH_TIMEOUT_SECONDS", "OLLAMA_CHAT_TIMEOUT_SECONDS", "OLLAMA_EMBED_TIMEOUT_SECONDS"):
            for value in ("0", "-1", "-0.5", "nan", "inf", "-inf"):
                with self.subTest(variable=variable, value=value):
                    with self.assertRaisesRegex(main.SettingsError, f"^{variable} must be a finite number of seconds"):
                        load_with({variable: value})

            with self.subTest(variable=variable, value="0.25"):
                load_with({variable: "0.25"})

    def test_main_reports_malformed_and_out_of_range_numbers_without_a_traceback(self) -> None:
        for variable, value in (
            ("CHUNK_MAX_CHARS", "large"),
            ("CHUNK_MAX_CHARS", "10"),
            ("HEALTH_TIMEOUT_SECONDS", "fast"),
            ("OLLAMA_EMBED_TIMEOUT_SECONDS", "nan"),
            ("OLLAMA_CHAT_TIMEOUT_SECONDS", "0"),
        ):
            stderr = io.StringIO()

            with (
                self.subTest(variable=variable, value=value),
                patch.dict(os.environ, {variable: value}, clear=True),
                patch.object(main, "load_dotenv"),
                patch.object(local_http.OPENER, "open", side_effect=AssertionError("network call")) as urlopen,
                patch("sys.stderr", stderr),
            ):
                exit_code = main.main(["health"])

                self.assertEqual(exit_code, 2)
                self.assertIn(f"Invalid settings: {variable}", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                urlopen.assert_not_called()

    def test_main_reports_invalid_settings_without_a_traceback(self) -> None:
        stderr = io.StringIO()

        with (
            patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://ollama.invalid:11434"}, clear=True),
            patch.object(main, "load_dotenv"),
            patch.object(local_http.OPENER, "open", side_effect=AssertionError("network call")) as urlopen,
            patch("sys.stderr", stderr),
        ):
            exit_code = main.main(["health"])

        self.assertEqual(exit_code, 2)
        self.assertIn("Invalid settings: OLLAMA_BASE_URL", stderr.getvalue())
        urlopen.assert_not_called()

    def test_env_template_lists_every_setting(self) -> None:
        template = (ROOT / ".env.template").read_text(encoding="utf-8")
        listed = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", template, re.MULTILINE))
        source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
        read = set(re.findall(r'os\.getenv\("([A-Z0-9_]+)"', source)) - {"XDG_DATA_HOME"}

        self.assertEqual(read - listed, set())
        self.assertIn("MODEL_NAME=gemma4:12b\n", template)


class LocalHttpTests(unittest.TestCase):
    def test_opener_ignores_proxy_environment(self) -> None:
        # The opener is built at import, so build it with proxies configured.
        with with_proxy_environment("http://127.0.0.1:9"):
            self.assertFalse(any(isinstance(h, ProxyHandler) for h in local_http.OPENER.handlers))

    def test_opener_has_only_plain_http_handlers(self) -> None:
        names = sorted(type(handler).__name__ for handler in local_http.OPENER.handlers)

        self.assertEqual(
            names,
            ["HTTPDefaultErrorHandler", "HTTPErrorProcessor", "HTTPHandler", "HTTPSHandler", "UnknownHandler"],
        )

    def test_opener_refuses_file_and_ftp_urls(self) -> None:
        for url in ("file:///etc/hostname", "ftp://localhost/x"):
            with self.subTest(url=url):
                with self.assertRaisesRegex(URLError, "unknown url type"):
                    local_http.urlopen(url, timeout=1)

    def test_every_ollama_call_goes_through_the_one_opener(self) -> None:
        # Not assertIs: other tests reload local_http, which makes a new
        # function object that still reads the same module's OPENER.
        for module in (main, embeddings):
            self.assertEqual((module.urlopen.__module__, module.urlopen.__name__), ("local_http", "urlopen"))

        for module in ("main.py", "embeddings.py"):
            source = (ROOT / "src" / module).read_text(encoding="utf-8")
            self.assertNotRegex(source, r"from urllib\.request import[^\n]*\burlopen\b")

    def test_urlopen_delegates_to_the_opener(self) -> None:
        with patch.object(local_http.OPENER, "open", return_value="response") as opener_open:
            self.assertEqual(local_http.urlopen(TAGS_URL, timeout=3), "response")

        opener_open.assert_called_once_with(TAGS_URL, timeout=3)


class HealthCheckTests(unittest.TestCase):
    def test_check_http_endpoint_returns_ok_for_successful_response(self) -> None:
        with patch.object(main, "urlopen", return_value=UrlopenResponse(200)) as urlopen:
            check = main.check_http_endpoint("Ollama", TAGS_URL, timeout=1)

        urlopen.assert_called_once_with(TAGS_URL, timeout=1)
        self.assertEqual(check, main.HealthCheck("Ollama", TAGS_URL, True, "reachable"))

    def test_check_http_endpoint_returns_failure_for_connection_error(self) -> None:
        with patch.object(main, "urlopen", side_effect=URLError("connection refused")):
            check = main.check_http_endpoint("Ollama", TAGS_URL, timeout=1)

        self.assertEqual(check, main.HealthCheck("Ollama", TAGS_URL, False, "connection refused"))

    def test_check_http_endpoint_reports_http_errors(self) -> None:
        error = HTTPError(TAGS_URL, 500, "Server Error", {}, io.BytesIO(b""))

        with patch.object(main, "urlopen", side_effect=error):
            check = main.check_http_endpoint("Ollama", TAGS_URL, timeout=1)

        self.assertFalse(check.ok)
        self.assertEqual(check.detail, "HTTP 500")

    def test_check_http_endpoint_reports_non_2xx_status(self) -> None:
        with patch.object(main, "urlopen", return_value=UrlopenResponse(304)):
            check = main.check_http_endpoint("Ollama", TAGS_URL, timeout=1)

        self.assertFalse(check.ok)
        self.assertEqual(check.detail, "HTTP 304")

    def test_check_http_endpoint_reports_timeouts(self) -> None:
        with patch.object(main, "urlopen", side_effect=TimeoutError()):
            check = main.check_http_endpoint("Ollama", TAGS_URL, timeout=1)

        self.assertFalse(check.ok)
        self.assertEqual(check.detail, "request timed out")

    def test_health_checks_report_installed_and_missing_models(self) -> None:
        tags = b'{"models": [{"name": "gemma4:12b"}, {"name": "other:7b"}]}'

        with patch.object(main, "urlopen", side_effect=[UrlopenResponse(200), UrlopenResponse(200, tags)]):
            checks = main.run_health_checks(make_settings())

        self.assertEqual([(c.name, c.ok) for c in checks], [
            ("Ollama", True),
            ("Chat model", True),
            ("Embedding model", False),
        ])
        self.assertEqual(checks[1].detail, "installed, runs locally")
        self.assertIn("ollama pull bge-m3", checks[2].detail)

    def test_health_check_refuses_an_installed_model_that_runs_remotely(self) -> None:
        tags = (
            b'{"models": [{"name": "gemma4:12b", "remote_model": "gpt-oss:120b",'
            b' "remote_host": "https://ollama.com:443"}, {"name": "bge-m3:latest"}]}'
        )

        with patch.object(main, "urlopen", side_effect=[UrlopenResponse(200), UrlopenResponse(200, tags)]):
            checks = main.run_health_checks(make_settings())

        self.assertEqual([(c.name, c.ok) for c in checks], [
            ("Ollama", True),
            ("Chat model", False),
            ("Embedding model", True),
        ])
        self.assertIn("refused: 'gemma4:12b' is served from https://ollama.com:443", checks[1].detail)

    def test_model_checks_fail_when_ollama_is_down(self) -> None:
        with patch.object(main, "urlopen", side_effect=URLError("connection refused")):
            checks = main.run_health_checks(make_settings())

        self.assertFalse(any(check.ok for check in checks))

    def test_malformed_tag_list_reports_ollama_not_reachable(self) -> None:
        with patch.object(main, "urlopen", side_effect=[UrlopenResponse(200), UrlopenResponse(200, b"<html>")]):
            checks = main.run_health_checks(make_settings())

        self.assertEqual([c.detail for c in checks[1:]], ["Ollama not reachable", "Ollama not reachable"])


class MainTests(unittest.TestCase):
    def test_main_returns_zero_when_services_are_ready(self) -> None:
        checks = [
            main.HealthCheck("Ollama", "http://ollama:11434/api/tags", True, "reachable"),
            main.HealthCheck("Embedding model", "bge-m3", True, "installed"),
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
            main.HealthCheck("Embedding model", "bge-m3", True, "installed"),
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

    def test_main_routes_ingest_search_and_eval(self) -> None:
        with (
            patch.object(main, "load_settings", return_value=self._settings()),
            patch.object(main, "run_ingest_command", return_value=0) as ingest,
            patch.object(main, "run_search_command", return_value=0) as search,
            patch.object(main, "run_eval_command", return_value=0) as evaluate,
        ):
            main.main(["ingest", "--rebuild"])
            main.main(["search", "why", "sqlite-vec", "-k", "3", "--full"])
            main.main(["eval", "questions.txt"])

        ingest.assert_called_once_with(self._settings(), True)
        search.assert_called_once_with(self._settings(), "why sqlite-vec", 3, True)
        evaluate.assert_called_once_with(self._settings(), "questions.txt", 5)

    def test_k_below_one_is_rejected_by_the_parser(self) -> None:
        for command in (["search", "why"], ["eval", "questions.txt"]):
            for k in ("0", "-1"):
                stderr = io.StringIO()

                with (
                    self.subTest(command=command[0], k=k),
                    patch.object(main, "load_settings") as load_settings,
                    patch("sys.stderr", stderr),
                ):
                    with self.assertRaises(SystemExit) as caught:
                        main.main([*command, "-k", k])

                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn("-k", stderr.getvalue())
                    load_settings.assert_not_called()

    def test_k_of_one_is_accepted(self) -> None:
        with (
            patch.object(main, "load_settings", return_value=self._settings()),
            patch.object(main, "run_search_command", return_value=0) as search,
            patch.object(main, "run_eval_command", return_value=0) as evaluate,
        ):
            main.main(["search", "why", "-k", "1"])
            main.main(["eval", "questions.txt", "-k", "1"])

        search.assert_called_once_with(self._settings(), "why", 1, False)
        evaluate.assert_called_once_with(self._settings(), "questions.txt", 1)

    def _settings(self) -> main.Settings:
        return make_settings()


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
        self.assertEqual(payload["model"], "gemma4:12b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "What do you know?"})

    def test_ask_ollama_checks_the_model_before_sending_the_prompt(self) -> None:
        tags = UrlopenResponse(200, b'{"models": [{"name": "gemma4:12b"}]}')
        reply = UrlopenResponse(200, b'{"message": {"content": "ok"}}')

        with patch.object(main, "urlopen", side_effect=[tags, reply]) as urlopen:
            main.ask_ollama(self._settings(), "hi")

        urls = [call.args[0].full_url for call in urlopen.call_args_list]
        self.assertEqual(urls, ["http://ollama:11434/api/tags", "http://ollama:11434/api/chat"])

    def test_ask_ollama_never_sends_a_prompt_to_a_remote_model(self) -> None:
        tags = UrlopenResponse(200, b'{"models": [{"name": "gemma4:12b", "remote_host": "https://ollama.com:443"}]}')

        with patch.object(main, "urlopen", side_effect=[tags]) as urlopen:
            with self.assertRaises(main.RemoteModelError):
                main.ask_ollama(self._settings(), "my private question")

        self.assertEqual(urlopen.call_count, 1)

    def test_run_chat_command_reports_a_refused_model(self) -> None:
        with (
            patch.object(main, "check_http_endpoint", return_value=main.HealthCheck("Ollama", "u", True, "reachable")),
            patch.object(main, "ask_ollama", side_effect=main.RemoteModelError("'x' is served from elsewhere")),
            patch("sys.stderr", io.StringIO()) as stderr,
        ):
            exit_code = main.run_chat_command(self._settings(), "hello")

        self.assertEqual(exit_code, 1)
        self.assertIn("Chat request failed: 'x' is served from elsewhere", stderr.getvalue())

    def test_run_chat_command_reports_malformed_replies(self) -> None:
        with (
            patch.object(main, "check_http_endpoint", return_value=main.HealthCheck("Ollama", "u", True, "reachable")),
            patch.object(main, "urlopen", return_value=UrlopenResponse(200, b"not json")),
            patch("sys.stderr", io.StringIO()) as stderr,
        ):
            exit_code = main.run_chat_command(self._settings(), "hello")

        self.assertEqual(exit_code, 1)
        self.assertIn("Chat request failed", stderr.getvalue())

    def test_ask_ollama_sends_no_think_field(self) -> None:
        response = UrlopenResponse(200, b'{"message": {"content": "ok"}}')

        with patch.object(main, "urlopen", return_value=response) as urlopen:
            main.ask_ollama(self._settings(), "hi")

        payload = json_from_request(urlopen.call_args.args[0])

        self.assertNotIn("think", payload)
        self.assertNotIn("<|think|>", payload["messages"][0]["content"])

    def test_ask_ollama_returns_the_answer_unchanged(self) -> None:
        response = UrlopenResponse(200, b'{"message": {"content": "Plain answer, no reasoning."}}')

        with patch.object(main, "urlopen", return_value=response):
            answer = main.ask_ollama(self._settings(), "hi")

        self.assertEqual(answer, "Plain answer, no reasoning.")

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
        return make_settings()


class EvalFileTests(unittest.TestCase):
    def test_parse_eval_file_skips_comments_and_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.txt"
            path.write_text("# header\n\nWhy A? | notes/a.md\nX | Y | notes/b.md\n", encoding="utf-8")
            cases = main.parse_eval_file(path)

        self.assertEqual(cases, [("Why A?", "notes/a.md"), ("X | Y", "notes/b.md")])

    def test_parse_eval_file_rejects_lines_without_separator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.txt"
            path.write_text("no separator here\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                main.parse_eval_file(path)

    def test_parse_eval_file_rejects_an_empty_question_or_expected_file(self) -> None:
        for line in (" | notes/a.md", "|notes/a.md", "Why A? | ", "Why A? |", "Why A? | ./", "|"):
            with self.subTest(line=line), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "q.txt"
                path.write_text(f"# header\n{line}\n", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, r"q\.txt:2: both the question and the expected file"):
                    main.parse_eval_file(path)

    def test_parse_eval_file_strips_a_leading_dot_slash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.txt"
            path.write_text("Why A? | ./notes/a.md\nWhy B? | notes/./b.md\n", encoding="utf-8")
            cases = main.parse_eval_file(path)

        self.assertEqual(cases, [("Why A?", "notes/a.md"), ("Why B?", "notes/./b.md")])

    def test_eval_without_questions_fails(self) -> None:
        for content in ("", "# only a comment\n\n"):
            stderr = io.StringIO()

            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as tmp,
                patch.object(main, "search_index") as search_index,
                patch("sys.stderr", stderr),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                path = Path(tmp) / "q.txt"
                path.write_text(content, encoding="utf-8")
                exit_code = main.run_eval_command(make_settings(), str(path))

            self.assertEqual(exit_code, 1)
            self.assertIn("No questions in", stderr.getvalue())
            search_index.assert_not_called()

    def test_eval_counts_a_dot_slash_expected_path_as_a_hit(self) -> None:
        from store import SearchHit

        hits = [SearchHit("notes/a.md", "", "text", 0.9)]
        stdout = io.StringIO()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(main, "search_index", return_value=(hits, None)),
            contextlib.redirect_stdout(stdout),
        ):
            path = Path(tmp) / "q.txt"
            path.write_text("Why A? | ./notes/a.md\n", encoding="utf-8")
            exit_code = main.run_eval_command(make_settings(), str(path))

        self.assertEqual(exit_code, 0)
        self.assertIn("HIT  #1  Why A?", stdout.getvalue())
        self.assertIn("1/1 questions", stdout.getvalue())
@contextlib.contextmanager
def with_hosts_file(content: str):
    with tempfile.TemporaryDirectory() as tmp:
        hosts = Path(tmp) / "hosts"
        hosts.write_text(content, encoding="utf-8")

        with patch.object(main, "HOSTS_FILE", str(hosts)):
            yield


def load_with(env: dict[str, str]) -> main.Settings:
    with patch.dict(os.environ, env, clear=True), patch.object(main, "load_dotenv"):
        return main.load_settings()


def make_settings() -> main.Settings:
    return main.Settings(
        project_name="mimir-advisor",
        persona="Adjutant_Mimir",
        model_name="gemma4:12b",
        embed_model="bge-m3",
        ollama_base_url="http://ollama:11434",
        data_source_dir="./tests/fixtures/vault",
        index_dir="/tmp/mimir-test-index",
        chunk_max_chars=1200,
        health_timeout_seconds=2,
        chat_timeout_seconds=60,
        embed_timeout_seconds=120,
    )


def json_from_request(request: object) -> dict[str, object]:
    return main.json.loads(request.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
