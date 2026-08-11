"""Local dashboard server with one safe operator action: retry implementation."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .harvest import CandidateStore, to_dashboard

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD), **kwargs)

    def do_POST(self):  # noqa: N802
        if self.path not in ("/api/retry-implementation", "/api/hide"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            cid = str(json.loads(self.rfile.read(size)).get("id", ""))
            st = CandidateStore()
            row = next((r for r in st.all() if r["id"] == cid), None)
            if not row:
                raise ValueError("strategy not found")
            if self.path == "/api/hide":
                st.update_result(cid, status="archived", verdict="archived", note="hidden by user")
                st.append_audit(cid, "hidden by user")
            else:
                if row["status"] not in ("blocked", "implementing"):
                    raise ValueError("strategy is not parked for implementation")
                st.update_result(cid, status="blocked", verdict="blocked", implementation_attempts=0,
                                 note="manual implementation retry requested")
                st.append_audit(cid, "manual retry requested")
            to_dashboard(st)
            body = b'{"ok":true}'
            self.send_response(HTTPStatus.OK)
        except Exception as exc:  # local UI gets a useful, non-sensitive message
            body = json.dumps({"ok": False, "error": str(exc)}).encode()
            self.send_response(HTTPStatus.BAD_REQUEST)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    with ThreadingHTTPServer(("127.0.0.1", 8777), Handler) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
