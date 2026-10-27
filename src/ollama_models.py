"""Refuse Ollama models that do not run on this machine.

Ollama cloud models (tags like ``gpt-oss:120b-cloud``) are listed and served by
the local daemon, but every prompt and every chunk sent to them is forwarded to
ollama.com. The URL check cannot see that: the request still goes to localhost.
"""

from __future__ import annotations

import json
import re
from typing import Callable
from urllib.request import Request

CLOUD_NAME_RE = re.compile(r"(^|[:-])cloud$")


class RemoteModelError(RuntimeError):
    pass


def is_cloud_name(model: str) -> bool:
    return CLOUD_NAME_RE.search(model.strip().lower()) is not None


def is_remote_entry(entry: dict) -> bool:
    """An /api/tags entry for a model that runs elsewhere. A cloud model copied
    under a local-looking name keeps its remote_host/remote_model fields."""
    return bool(entry.get("remote_host") or entry.get("remote_model")) or is_cloud_name(str(entry.get("name", "")))


def find_entry(model: str, entries: list[dict]) -> dict | None:
    names = {model} if ":" in model else {model, f"{model}:latest"}
    return next((entry for entry in entries if entry.get("name") in names), None)


def fetch_tags(base_url: str, timeout: float, urlopen: Callable) -> list[dict]:
    """The models Ollama lists. ``urlopen`` is the caller's, so every request
    goes through the one local opener (and the caller's test doubles)."""
    request = Request(f"{base_url.rstrip('/')}/api/tags", method="GET")

    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))

    models = body.get("models") if isinstance(body, dict) else None
    return [entry for entry in models if isinstance(entry, dict)] if isinstance(models, list) else []


def remote_reason(model: str, entry: dict | None) -> str | None:
    """Why ``model`` must not be used, or None when it runs locally (or is not
    installed, which Ollama itself reports when the model is called)."""
    if is_cloud_name(model):
        return f"'{model}' is an Ollama cloud model; it runs on ollama.com, not on this machine"

    if entry is not None and is_remote_entry(entry):
        where = entry.get("remote_host") or "a remote host"
        return f"'{model}' is served from {where}, not from this machine"

    return None


def ensure_local_model(base_url: str, model: str, timeout: float, urlopen: Callable) -> None:
    """Raise RemoteModelError unless ``model`` runs on this machine. Checked
    right before each chat or embed call: models can change between runs."""
    reason = remote_reason(model, None)

    if reason is None:
        reason = remote_reason(model, find_entry(model, fetch_tags(base_url, timeout, urlopen)))

    if reason is not None:
        raise RemoteModelError(f"{reason}. Note text is never sent off this machine; pick a local model.")
