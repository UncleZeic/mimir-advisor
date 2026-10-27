"""Main entry point for mimir-advisor."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import ipaddress
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from local_http import urlopen
from ollama_models import RemoteModelError, ensure_local_model, fetch_tags, find_entry, is_cloud_name, remote_reason

import os
from pathlib import Path
import re
import sqlite3
import sys

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(env_path) -> None:
        if not os.path.exists(env_path):
            return

        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                stripped = line.strip()

                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue

                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


REPO_ROOT = Path(__file__).resolve().parents[1]
# Only the repo's own .env: python-dotenv would otherwise search upwards, so a
# .env in a parent folder could silently set DATA_SOURCE_DIR or INDEX_DIR.
DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_DATA_SOURCE_DIR = "./tests/fixtures/vault"
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
LOCAL_OLLAMA_HOSTS = ("localhost", "127.0.0.1", "::1", "host.docker.internal")
# Only Docker maps host.docker.internal (via extra_hosts). On the host the name
# goes to DNS, where resolv.conf search domains can turn it into a real,
# remote host such as host.docker.internal.<isp-domain>.
CONTAINER_ONLY_HOSTS = ("host.docker.internal",)
CONTAINER_MARKERS = ("/.dockerenv", "/run/.containerenv")
# Even in a container the name must come from /etc/hosts (compose's
# extra_hosts: host-gateway); a plain `docker run` would send it to DNS.
HOSTS_FILE = "/etc/hosts"
# Must exceed the chunker's 150-char overlap, or hard splits cannot advance.
MIN_CHUNK_MAX_CHARS = 300


class SettingsError(ValueError):
    pass


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
    load_dotenv(DOTENV_PATH)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("MODEL_NAME", "qwen3:8b")
    embed_model = os.getenv("EMBED_MODEL", "bge-m3")
    chunk_max_chars = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
    check_ollama_base_url(ollama_base_url)

    for variable, model in (("MODEL_NAME", model_name), ("EMBED_MODEL", embed_model)):
        if is_cloud_name(model):
            raise SettingsError(
                f"{variable}={model} is an Ollama cloud model: it runs on ollama.com, so note text "
                "would leave this machine. Pick a local model."
            )

    if chunk_max_chars < MIN_CHUNK_MAX_CHARS:
        raise SettingsError(
            f"CHUNK_MAX_CHARS must be at least {MIN_CHUNK_MAX_CHARS} "
            f"(it must exceed the chunk overlap); got {chunk_max_chars}."
        )

    return Settings(
        project_name=os.getenv("PROJECT_NAME", "mimir-advisor"),
        persona=os.getenv("PERSONA", "Adjutant_Mimir"),
        model_name=model_name,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
        # Fixtures by default: anything run without an explicit override only
        # ever sees synthetic notes. Real notes need DATA_SOURCE_DIR on purpose.
        data_source_dir=os.getenv("DATA_SOURCE_DIR", DEFAULT_DATA_SOURCE_DIR),
        # `or`, not a getenv default: an empty INDEX_DIR= in .env must not
        # resolve to the current directory, which is usually the repo.
        index_dir=os.getenv("INDEX_DIR") or default_index_dir(),
        chunk_max_chars=chunk_max_chars,
        health_timeout_seconds=float(os.getenv("HEALTH_TIMEOUT_SECONDS", "2")),
        chat_timeout_seconds=float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "60")),
        embed_timeout_seconds=float(os.getenv("OLLAMA_EMBED_TIMEOUT_SECONDS", "120")),
    )


def check_ollama_base_url(url: str) -> None:
    """Prompts and note text go to this URL, so it must be this machine."""
    parts = urlsplit(url)

    if parts.scheme not in {"http", "https"} or parts.hostname not in LOCAL_OLLAMA_HOSTS:
        raise SettingsError(
            f"OLLAMA_BASE_URL must point at a local Ollama ({', '.join(LOCAL_OLLAMA_HOSTS)}), "
            f"got {url!r}. Note text is never sent to another host."
        )

    if parts.hostname in CONTAINER_ONLY_HOSTS and not running_in_container():
        raise SettingsError(
            f"OLLAMA_BASE_URL uses {parts.hostname}, which only resolves safely inside Docker. "
            "Outside a container use http://localhost:11434."
        )

    if parts.hostname in CONTAINER_ONLY_HOSTS and not is_private_address(hosts_file_address(parts.hostname)):
        raise SettingsError(
            f"OLLAMA_BASE_URL uses {parts.hostname}, but {HOSTS_FILE} does not map it to a private "
            "address, so it would be looked up in DNS. Run with "
            f"--add-host={parts.hostname}:host-gateway (compose sets this via extra_hosts)."
        )


def running_in_container() -> bool:
    return any(os.path.exists(marker) for marker in CONTAINER_MARKERS)


def hosts_file_address(name: str) -> str | None:
    """The first address HOSTS_FILE gives for ``name``, or None."""
    try:
        with open(HOSTS_FILE, encoding="utf-8", errors="replace") as hosts:
            for line in hosts:
                fields = line.split("#", 1)[0].split()

                if len(fields) >= 2 and name.lower() in (alias.lower() for alias in fields[1:]):
                    return fields[0]
    except OSError:
        return None

    return None


def is_private_address(address: str | None) -> bool:
    try:
        ip = ipaddress.ip_address(address or "")
    except ValueError:
        return False

    return ip.is_private or ip.is_link_local or ip.is_loopback


def default_index_dir() -> str:
    """Indexes hold note text, so they live outside the repo and its workspace."""
    data_home = os.getenv("XDG_DATA_HOME") or os.path.join("~", ".local", "share")
    return os.path.join(data_home, "mimir-advisor", "index")


def check_http_endpoint(name: str, url: str, timeout: float) -> HealthCheck:
    try:
        with urlopen(url, timeout=timeout) as response:
            if 200 <= response.status < 300:
                return HealthCheck(name=name, url=url, ok=True, detail="reachable")

            detail = f"HTTP {response.status}"
    except HTTPError as error:
        detail = f"HTTP {error.code}"
    except URLError as error:
        detail = str(error.reason)
    except TimeoutError:
        detail = "request timed out"

    return HealthCheck(name=name, url=url, ok=False, detail=detail)


def installed_ollama_models(settings: Settings) -> list[dict]:
    return fetch_tags(settings.ollama_base_url, settings.health_timeout_seconds, urlopen)


def run_health_checks(settings: Settings) -> list[HealthCheck]:
    tags_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    ollama = check_http_endpoint("Ollama", tags_url, settings.health_timeout_seconds)
    checks = [ollama]

    try:
        installed = installed_ollama_models(settings) if ollama.ok else None
    except (HTTPError, URLError, TimeoutError, ValueError):
        installed = None

    for label, model in (("Chat model", settings.model_name), ("Embedding model", settings.embed_model)):
        if installed is None:
            checks.append(HealthCheck(label, model, False, "Ollama not reachable"))
            continue

        entry = find_entry(model, installed)
        reason = remote_reason(model, entry)

        if reason is not None:
            checks.append(HealthCheck(label, model, False, f"refused: {reason}"))
        elif entry is not None:
            checks.append(HealthCheck(label, model, True, "installed, runs locally"))
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
    ensure_local_model(settings.ollama_base_url, settings.model_name, settings.health_timeout_seconds, urlopen)
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.model_name,
        "stream": False,
        "think": False,
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

    return strip_think(body.get("message", {}).get("content", ""))


def strip_think(text: str) -> str:
    """Older Ollama versions ignore "think" and inline the reasoning instead."""
    return THINK_BLOCK.sub("", text).strip()


def run_chat_command(settings: Settings, prompt: str | None = None) -> int:
    ollama_check = check_http_endpoint(
        "Ollama",
        f"{settings.ollama_base_url.rstrip('/')}/api/tags",
        settings.health_timeout_seconds,
    )

    if not ollama_check.ok:
        print(f"Ollama is not ready: {ollama_check.detail}", file=sys.stderr)
        return 1

    if prompt:
        try:
            print(ask_ollama(settings, prompt))
            return 0
        except (HTTPError, URLError, TimeoutError, ValueError, RemoteModelError) as error:
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
        except (HTTPError, URLError, TimeoutError, ValueError, RemoteModelError) as error:
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
    from store import IndexStore, StoreError, index_path_for, remove_sidecars, temporary_path

    root = Path(settings.data_source_dir).expanduser().resolve()

    if not root.is_dir():
        print(f"Source folder not found: {root}", file=sys.stderr)
        return 1

    try:
        index_path = index_path_for(settings.data_source_dir, settings.index_dir)
    except StoreError as error:
        print(f"Ingest stopped: {error}", file=sys.stderr)
        return 1

    print(f"Ingesting {root}")
    print(f"Index: {index_path}")
    # A rebuild goes into a side file and replaces the old index only once it
    # is complete, so a failed rebuild leaves the previous index usable.
    build_path = temporary_path(index_path) if rebuild else index_path

    try:
        if rebuild:
            build_path.unlink(missing_ok=True)
            remove_sidecars(build_path)

        store = IndexStore(build_path)

        try:
            report = ingest_folder(store, make_embedder(settings), root, settings.chunk_max_chars, print)
            files, chunks = store.stats()
        finally:
            store.close()

        if rebuild:
            remove_sidecars(index_path)
            os.replace(build_path, index_path)
            remove_sidecars(build_path)
    except (EmbeddingError, StoreError, sqlite3.Error, OSError) as error:
        if rebuild:
            discard_index(build_path)

        where = "" if isinstance(error, (EmbeddingError, StoreError)) else f" (index: {build_path})"
        print(f"Ingest stopped: {error}{where}", file=sys.stderr)
        return 1
    except BaseException:
        # Ctrl-C or an unexpected error: still never leave a stray copy of
        # the notes behind in a half-built side file.
        if rebuild:
            discard_index(build_path)

        raise

    print(
        f"Done: {report.indexed} indexed ({report.chunks} chunks), "
        f"{report.unchanged} unchanged, {report.removed} removed, {len(report.empty)} empty."
    )

    if report.unreadable:
        print(f"Skipped {len(report.unreadable)} unreadable: {', '.join(report.unreadable)}", file=sys.stderr)

    print(f"Index now holds {files} files, {chunks} chunks.")
    return 0


def discard_index(path: Path) -> None:
    """Best effort: a half-built index still holds note text."""
    from store import remove_sidecars

    try:
        path.unlink(missing_ok=True)
        remove_sidecars(path)
    except OSError:
        pass


def search_index(settings: Settings, query: str, k: int):
    """Return (hits, error_message). Opens and closes the index."""
    from embeddings import EmbeddingError
    from store import StoreError, index_path_for

    try:
        store = open_index(settings, must_exist=True)

        if store is None:
            return None, "no index"

        try:
            embedder = make_embedder(settings)
            store.check_model(embedder.model)
            vector = embedder.embed([query])[0]
            return store.search(vector, k), None
        finally:
            store.close()
    except (EmbeddingError, StoreError) as error:
        return None, str(error)
    except (sqlite3.Error, OSError) as error:
        index_path = index_path_for(settings.data_source_dir, settings.index_dir)
        return None, f"{error} (index: {index_path})"


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

    try:
        settings = load_settings()
    except SettingsError as error:
        print(f"Invalid settings: {error}", file=sys.stderr)
        return 2

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
