from __future__ import annotations

import json
import re
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "My_approch" / "analysis"
GENERATOR = ROOT / "scripts" / "generate_stf_analysis.py"
ITERATION_RE = re.compile(r"^[0-9]{1,7}$")
SHARDS_RE = re.compile(r"^(preferred|auto|[0-9]{1,5})$")


def normalize_shards(value: str) -> str:
    value = (value or "preferred").strip().lower()
    return "preferred" if value in {"", "auto", "preferred"} else value


def report_stem(iterations: str, shards: str) -> str:
    return f"stf_{iterations}_branch_compare" if shards == "preferred" else f"stf_{iterations}_shards{shards}_branch_compare"


class AnalysisHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(ANALYSIS_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/report":
            self.handle_report(parse_qs(parsed.query))
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def handle_report(self, query: dict[str, list[str]]) -> None:
        iterations = (query.get("iterations") or [""])[0]
        shards = normalize_shards((query.get("shards") or ["preferred"])[0])
        if not ITERATION_RE.match(iterations):
            self.write_json({"ok": False, "error": "iterations must be a numeric value"}, status=400)
            return
        if not SHARDS_RE.match(shards):
            self.write_json({"ok": False, "error": "shards must be preferred/auto or a numeric shard count"}, status=400)
            return

        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(GENERATOR), iterations, shards],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            self.write_json(
                {
                    "ok": False,
                    "error": "generator failed",
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                },
                status=500,
            )
            return

        stem = report_stem(iterations, shards)
        report = ANALYSIS_DIR / f"{stem}.html"
        data = ANALYSIS_DIR / f"{stem}_data.json"
        self.write_json(
            {
                "ok": True,
                "iterations": iterations,
                "shards": shards,
                "report": report.name,
                "data": data.name,
                "stdout": result.stdout[-4000:],
            }
        )

    def write_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer((host, port), AnalysisHandler)
    print(f"STF analysis server: http://{host}:{port}/")
    print("Select an iteration in the page to generate missing reports from branch logs.")
    server.serve_forever()


if __name__ == "__main__":
    main()
