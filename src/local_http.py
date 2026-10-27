"""The one way out to the network: Ollama on a local address, never via a proxy.

urllib's default opener honours HTTP(S)_PROXY from the environment, which would
route prompts and note text through whatever proxy happens to be configured.
"""

from __future__ import annotations

from urllib.request import ProxyHandler, Request, build_opener

OPENER = build_opener(ProxyHandler({}))


def urlopen(request: Request | str, timeout: float):
    return OPENER.open(request, timeout=timeout)
