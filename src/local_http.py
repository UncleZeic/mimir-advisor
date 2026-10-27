"""The one way out to the network: Ollama on a local address, never via a proxy.

urllib's default opener honours HTTP(S)_PROXY from the environment, which would
route prompts and note text through whatever proxy happens to be configured,
and follows redirects to any host. This opener is assembled by hand instead:
plain HTTP(S) only, no proxy, no redirects (a 3xx raises HTTPError), and no
ftp:// or file:// handlers.
"""

from __future__ import annotations

from urllib.request import (
    HTTPDefaultErrorHandler,
    HTTPErrorProcessor,
    HTTPHandler,
    HTTPSHandler,
    OpenerDirector,
    Request,
    UnknownHandler,
)


def build_local_opener() -> OpenerDirector:
    opener = OpenerDirector()

    for handler in (UnknownHandler(), HTTPHandler(), HTTPSHandler(), HTTPDefaultErrorHandler(), HTTPErrorProcessor()):
        opener.add_handler(handler)

    return opener


OPENER = build_local_opener()


def urlopen(request: Request | str, timeout: float):
    return OPENER.open(request, timeout=timeout)
