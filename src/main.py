"""Main entry point for mimir-advisor."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import os
import sys

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        env_path = ".env"

        if not os.path.exists(env_path):
            return

        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                stripped = line.strip()

                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue

                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    project_name: str
    persona: str
    model_name: str
    ollama_base_url: str
    vector_db_url: str
    data_source_dir: str
    health_timeout_seconds: float
    chat_timeout_seconds: float


@dataclass(frozen=True)
class HealthCheck:
    name: str
    url: str
    ok: bool
    detail: str


def load_settings() -> Settings:
    load_dotenv()

    return Settings(
        project_name=os.getenv("PROJECT_NAME", "mimir-advisor"),
        persona=os.getenv("PERSONA", "Adjutant_Mimir"),
        model_name=os.getenv("MODEL_NAME", "llama3"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        vector_db_url=os.getenv("VECTOR_DB_URL", "http://localhost:8000"),
        data_source_dir=os.getenv("DATA_SOURCE_DIR", "./data/raw"),
        health_timeout_seconds=float(os.getenv("HEALTH_TIMEOUT_SECONDS", "2")),
        chat_timeout_seconds=float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "60")),
    )


def check_http_endpoint(name: str, urls: Iterable[str], timeout: float) -> HealthCheck:
    endpoints = list(urls)
    last_error = "no health endpoints configured"

    for url in endpoints:
        try:
            with urlopen(url, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    return HealthCheck(name=name, url=url, ok=True, detail="reachable")

                last_error = f"HTTP {response.status}"
        except HTTPError as error:
            last_error = f"HTTP {error.code}"
        except URLError as error:
            last_error = str(error.reason)
        except TimeoutError:
            last_error = "request timed out"

    return HealthCheck(name=name, url=endpoints[-1], ok=False, detail=last_error)


def run_health_checks(settings: Settings) -> list[HealthCheck]:
    ollama_url = settings.ollama_base_url.rstrip("/")
    vector_db_url = settings.vector_db_url.rstrip("/")

    return [
        check_http_endpoint(
            "Ollama",
            [f"{ollama_url}/api/tags"],
            settings.health_timeout_seconds,
        ),
        check_http_endpoint(
            "ChromaDB",
            [
                f"{vector_db_url}/api/v2/heartbeat",
                f"{vector_db_url}/api/v1/heartbeat",
            ],
            settings.health_timeout_seconds,
        ),
    ]


def print_startup_status(settings: Settings, checks: list[HealthCheck]) -> None:
    print(f"{settings.project_name} startup")
    print(f"Persona: {settings.persona}")
    print(f"Local model: {settings.model_name}")
    print(f"Data source directory: {settings.data_source_dir}")

    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail} ({check.url})")


def run_health_command(settings: Settings) -> int:
    checks = run_health_checks(settings)
    print_startup_status(settings, checks)

    if not all(check.ok for check in checks):
        print("Startup aborted: local services are not ready.", file=sys.stderr)
        return 1

    print("mimir-advisor is ready for local reasoning.")
    return 0


def ask_ollama(settings: Settings, prompt: str) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.model_name,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are mimir-advisor, a local-first private advisor. "
                    "Answer clearly, avoid pretending to know personal facts that "
                    "were not provided, and mention when no source data is available."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=settings.chat_timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))

    return body.get("message", {}).get("content", "").strip()


def run_chat_command(settings: Settings, prompt: str | None = None) -> int:
    ollama_check = check_http_endpoint(
        "Ollama",
        [f"{settings.ollama_base_url.rstrip('/')}/api/tags"],
        settings.health_timeout_seconds,
    )

    if not ollama_check.ok:
        print(f"Ollama is not ready: {ollama_check.detail}", file=sys.stderr)
        return 1

    if prompt:
        try:
            print(ask_ollama(settings, prompt))
            return 0
        except (HTTPError, URLError, TimeoutError) as error:
            print(f"Chat request failed: {error}", file=sys.stderr)
            return 1

    print("mimir-advisor chat. Type /exit to leave.")

    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            print()
            return 0

        if user_input in {"/exit", "/quit"}:
            return 0

        if not user_input:
            continue

        try:
            print(ask_ollama(settings, user_input))
        except (HTTPError, URLError, TimeoutError) as error:
            print(f"Chat request failed: {error}", file=sys.stderr)
            return 1


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="mimir-advisor local runtime")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health", help="check local service readiness")

    chat_parser = subparsers.add_parser("chat", help="chat with the local Ollama model")
    chat_parser.add_argument("prompt", nargs="*", help="optional one-shot prompt")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "chat":
        prompt = " ".join(args.prompt).strip() or None
        return run_chat_command(settings, prompt)

    return run_health_command(settings)


if __name__ == "__main__":
    raise SystemExit(main())
