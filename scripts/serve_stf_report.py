#!/usr/bin/env python3
"""Serve the STF HTML report and refresh it on demand."""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def report_inputs() -> list[str]:
    patterns = ["vast*.log", "vast/experiments/**/console.log"]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(str(ROOT / pattern), recursive=True))
    return sorted(str(Path(path).relative_to(ROOT)) for path in paths)


def refresh_report() -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/summarize_vast_formula_logs.py",
        *report_inputs(),
        "--html",
        "index.html",
    ]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


class ReportHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self) -> None:
        if self.path != "/refresh":
            self.send_error(404, "unknown endpoint")
            return

        result = refresh_report()
        body = (result.stdout + "\n" + result.stderr).encode("utf-8", errors="replace")
        self.send_response(200 if result.returncode == 0 else 500)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ReportHandler)
    print(f"Serving STF report at http://{args.host}:{args.port}/index.html")
    print("Use the floating Refresh data button to regenerate index.html.")
    server.serve_forever()


if __name__ == "__main__":
    main()
