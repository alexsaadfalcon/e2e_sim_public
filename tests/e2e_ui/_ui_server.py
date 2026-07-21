"""Boot the real Dash webapp in a background thread for browser-driven tests.

This is the server half of the harness (the browser half lives in conftest.py).
It mirrors what ``dash.testing``'s threaded runner does internally -- run the app's
WSGI server on a free port in a daemon thread -- but without pulling in Selenium,
so we can drive the app with Playwright instead.

No torch/Sionna import happens here: importing ``webapp.app`` only builds the Dash
shell (heavy imports are lazy, inside the Run callback), so the light UI journeys
run on a machine without torch. Only the pipeline-Run journey needs torch + frames.
"""

from __future__ import annotations

import socket
import threading
import time
from urllib.request import urlopen


def _free_port() -> int:
    """Ask the OS for an unused localhost port (avoids clashing with a dev server)."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class LiveDashServer:
    """Context manager that serves ``webapp.app`` on a free port in a daemon thread.

    Usage::

        with LiveDashServer() as server:
            page.goto(server.url)

    ``.url`` is ready to hit only after ``__enter__`` has polled the server to a
    live 200 response, so tests never race the server's startup.
    """

    def __init__(self, host: str = "127.0.0.1"):
        self.host = host
        self.port = _free_port()
        self._srv = None
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "LiveDashServer":
        # Imported lazily so merely importing this module stays cheap/torch-free.
        from werkzeug.serving import make_server
        from webapp.app import app  # app.server is the Flask WSGI application

        # threaded=True so Dash's concurrent callback XHRs don't serialize behind
        # one another and stall the browser.
        self._srv = make_server(self.host, self.port, app.server, threaded=True)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        self._wait_until_live()
        return self

    def _wait_until_live(self, timeout: float = 30.0) -> None:
        """Poll the index route until it answers, so callers never race startup."""
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                with urlopen(self.url, timeout=1) as r:  # noqa: S310 (localhost only)
                    if r.status == 200:
                        return
            except Exception as e:  # connection refused until the server is up
                last_err = e
                time.sleep(0.1)
        raise RuntimeError(f"Dash server did not come up at {self.url}: {last_err}")

    def __exit__(self, *exc) -> None:
        if self._srv is not None:
            self._srv.shutdown()          # unblocks serve_forever()
            # shutdown() only stops the loop; server_close() releases the listening
            # socket deterministically instead of leaving it for GC.
            self._srv.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
