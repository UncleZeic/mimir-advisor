"""Main entry point for mimir-advisor."""

from dataclasses import dataclass
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

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


def main() -> int:
    settings = load_settings()
    checks = run_health_checks(settings)
    print_startup_status(settings, checks)

    if not all(check.ok for check in checks):
        print("Startup aborted: local services are not ready.", file=sys.stderr)
        return 1

    print("mimir-advisor is ready for local reasoning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
