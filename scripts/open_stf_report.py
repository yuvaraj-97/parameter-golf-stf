#!/usr/bin/env python3
"""Open the STF report, starting the refresh server if needed."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def server_url(host: str, port: int, page: str = "vast_formula_summary.html") -> str:
    return f"http://{host}:{port}/{page}"


def server_is_ready(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(server_url(host, port), timeout=0.7) as response:
            return 200 <= response.status < 400
    except (OSError, urllib.error.URLError):
        return False


def start_server(host: str, port: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "scripts/serve_stf_report.py", "--host", host, "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_until_ready(host: str, port: int, timeout_seconds: float = 6.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server_is_ready(host, port):
            return True
        time.sleep(0.2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--page", default="vast_formula_summary.html", choices=["vast_formula_summary.html", "index.html"])
    parser.add_argument("--no-open", action="store_true", help="start/check the server without opening a browser")
    args = parser.parse_args()

    if not server_is_ready(args.host, args.port):
        start_server(args.host, args.port)
        if not wait_until_ready(args.host, args.port):
            raise SystemExit(f"Could not start report server at {server_url(args.host, args.port, args.page)}")

    url = server_url(args.host, args.port, args.page)
    if not args.no_open:
        webbrowser.open(url)
    print(url)


if __name__ == "__main__":
    main()
