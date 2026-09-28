"""Main entry point for mimir-advisor."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import os
from pathlib import Path
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


DEFAULT_DATA_SOURCE_DIR = "./tests/fixtures/vault"


@dataclass(frozen=True)
class Settings:
    project_name: str
    persona: str
    model_name: str
    embed_model: str
    ollama_base_url: str
    data_source_dir: str
    index_dir: str
    chunk_max_chars: int
    health_timeout_seconds: float
    chat_timeout_seconds: float
    embed_timeout_seconds: float


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
        embed_model=os.getenv("EMBED_MODEL", "bge-m3"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        # Fixtures by default: anything run without an explicit override only
        # ever sees synthetic notes. Real notes need DATA_SOURCE_DIR on purpose.
        data_source_dir=os.getenv("DATA_SOURCE_DIR", DEFAULT_DATA_SOURCE_DIR),
        index_dir=os.getenv("INDEX_DIR", default_index_dir()),
        chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "1200")),
        health_timeout_seconds=float(os.getenv("HEALTH_TIMEOUT_SECONDS", "2")),
        chat_timeout_seconds=float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "60")),
        embed_timeout_seconds=float(os.getenv("OLLAMA_EMBED_TIMEOUT_SECONDS", "120")),
    )


def default_index_dir() -> str:
    """Indexes hold note text, so they live outside the repo and its workspace."""
    data_home = os.getenv("XDG_DATA_HOME") or os.path.join("~", ".local", "share")
    return os.path.join(data_home, "mimir-advisor", "index")


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


def installed_ollama_models(settings: Settings) -> set[str]:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"

    with urlopen(url, timeout=settings.health_timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))

    return {model.get("name", "") for model in body.get("models", [])}


def model_is_installed(model: str, installed: set[str]) -> bool:
    return model in installed or (":" not in model and f"{model}:latest" in installed)


def run_health_checks(settings: Settings) -> list[HealthCheck]:
    tags_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    ollama = check_http_endpoint("Ollama", [tags_url], settings.health_timeout_seconds)
    checks = [ollama]

    try:
        installed = installed_ollama_models(settings) if ollama.ok else None
    except (HTTPError, URLError, TimeoutError, ValueError):
        installed = None

    for label, model in (("Chat model", settings.model_name), ("Embedding model", settings.embed_model)):
        if installed is None:
            checks.append(HealthCheck(label, model, False, "Ollama not reachable"))
        elif model_is_installed(model, installed):
            checks.append(HealthCheck(label, model, True, "installed"))
        else:
            checks.append(HealthCheck(label, model, False, f"missing; run: ollama pull {model}"))

    return checks


def print_startup_status(settings: Settings, checks: list[HealthCheck]) -> None:
    print(f"{settings.project_name} startup")
    print(f"Persona: {settings.persona}")
    print(f"Local model: {settings.model_name}")
    print(f"Data source directory: {settings.data_source_dir}")
    print(f"Index directory: {settings.index_dir}")

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


def open_index(settings: Settings, must_exist: bool):
    from store import IndexStore, index_path_for

    path = index_path_for(settings.data_source_dir, settings.index_dir)

    if must_exist and not path.exists():
        print(
            f"No index for {Path(settings.data_source_dir).expanduser().resolve()}. "
            "Run ingest first.",
            file=sys.stderr,
        )
        return None

    return IndexStore(path)


def make_embedder(settings: Settings):
    from embeddings import OllamaEmbedder

    return OllamaEmbedder(
        settings.ollama_base_url,
        settings.embed_model,
        settings.embed_timeout_seconds,
    )


def run_ingest_command(settings: Settings, rebuild: bool = False) -> int:
    from embeddings import EmbeddingError
    from ingest import ingest_folder
    from store import StoreError, index_path_for

    root = Path(settings.data_source_dir).expanduser().resolve()

    if not root.is_dir():
        print(f"Source folder not found: {root}", file=sys.stderr)
        return 1

    index_path = index_path_for(settings.data_source_dir, settings.index_dir)

    if rebuild and index_path.exists():
        index_path.unlink()

    print(f"Ingesting {root}")
    print(f"Index: {index_path}")
    store = open_index(settings, must_exist=False)

    try:
        report = ingest_folder(store, make_embedder(settings), root, settings.chunk_max_chars, print)
        files, chunks = store.stats()
    except (EmbeddingError, StoreError) as error:
        print(f"Ingest stopped: {error}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print(
        f"Done: {report.indexed} indexed ({report.chunks} chunks), "
        f"{report.unchanged} unchanged, {report.removed} removed, {len(report.empty)} empty."
    )
    print(f"Index now holds {files} files, {chunks} chunks.")
    return 0


def search_index(settings: Settings, query: str, k: int):
    """Return (hits, error_message). Opens and closes the index."""
    from embeddings import EmbeddingError
    from store import StoreError

    store = open_index(settings, must_exist=True)

    if store is None:
        return None, "no index"

    try:
        embedder = make_embedder(settings)
        store.check_model(embedder.model)
        vector = embedder.embed([query])[0]
        return store.search(vector, k), None
    except (EmbeddingError, StoreError) as error:
        return None, str(error)
    finally:
        store.close()


def run_search_command(settings: Settings, query: str, k: int = 5, full: bool = False) -> int:
    root = Path(settings.data_source_dir).expanduser().resolve()
    print(f"Searching {root}")
    hits, error = search_index(settings, query, k)

    if hits is None:
        if error != "no index":
            print(f"Search failed: {error}", file=sys.stderr)
        return 1

    if not hits:
        print("No results. Is the index empty?")
        return 0

    for rank, hit in enumerate(hits, start=1):
        location = f"{hit.path} › {hit.heading}" if hit.heading else hit.path
        text = hit.text if full or len(hit.text) <= 300 else hit.text[:300].rstrip() + " …"
        print(f"\n{rank}. [{hit.score:.3f}] {location}")
        print("   " + text.replace("\n", "\n   "))

    return 0


def parse_eval_file(path: Path) -> list[tuple[str, str]]:
    """Lines of `question | expected/file.md`. Blank lines and # comments are ignored."""
    cases = []

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if "|" not in stripped:
            raise ValueError(f"{path}:{number}: expected 'question | expected/file.md'")

        question, expected = stripped.rsplit("|", 1)
        cases.append((question.strip(), expected.strip()))

    return cases


def run_eval_command(settings: Settings, eval_file: str, k: int = 5) -> int:
    try:
        cases = parse_eval_file(Path(eval_file).expanduser())
    except (OSError, ValueError) as error:
        print(f"Cannot read eval file: {error}", file=sys.stderr)
        return 1

    print(f"Evaluating {len(cases)} questions against {Path(settings.data_source_dir).expanduser().resolve()}")
    hits_in_top_k = 0

    for question, expected in cases:
        hits, error = search_index(settings, question, k)

        if hits is None:
            if error != "no index":
                print(f"Search failed: {error}", file=sys.stderr)
            return 1

        paths = [hit.path for hit in hits]

        if expected in paths:
            hits_in_top_k += 1
            print(f"  HIT  #{paths.index(expected) + 1}  {question}")
        else:
            top = paths[0] if paths else "nothing"
            print(f"  MISS     {question}  (expected {expected}, top was {top})")

    print(f"\n{hits_in_top_k}/{len(cases)} questions found their file in the top {k}.")
    return 0


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="mimir-advisor local runtime")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health", help="check local service readiness")

    chat_parser = subparsers.add_parser("chat", help="chat with the local Ollama model")
    chat_parser.add_argument("prompt", nargs="*", help="optional one-shot prompt")

    ingest_parser = subparsers.add_parser("ingest", help="index markdown notes from DATA_SOURCE_DIR")
    ingest_parser.add_argument("--rebuild", action="store_true", help="delete the index and re-embed everything")

    search_parser = subparsers.add_parser("search", help="print the chunks closest to a question")
    search_parser.add_argument("query", nargs="+", help="what to look for")
    search_parser.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    search_parser.add_argument("--full", action="store_true", help="print whole chunks")

    eval_parser = subparsers.add_parser("eval", help="score retrieval against a question file")
    eval_parser.add_argument("eval_file", help="lines of: question | expected/file.md")
    eval_parser.add_argument("-k", type=int, default=5, help="top-k to count as a hit (default 5)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "chat":
        prompt = " ".join(args.prompt).strip() or None
        return run_chat_command(settings, prompt)

    if args.command == "ingest":
        return run_ingest_command(settings, args.rebuild)

    if args.command == "search":
        return run_search_command(settings, " ".join(args.query), args.k, args.full)

    if args.command == "eval":
        return run_eval_command(settings, args.eval_file, args.k)

    return run_health_command(settings)


if __name__ == "__main__":
    raise SystemExit(main())
