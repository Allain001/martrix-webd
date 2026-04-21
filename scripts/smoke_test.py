from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIST = ROOT / "frontend" / "dist"
BASE_URL = "http://127.0.0.1:8012"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _wait_until_ready() -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            health = _get_json(f"{BASE_URL}/api/health")
            if health.get("status") == "ok":
                return
        except (urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.5)
    raise RuntimeError("Timed out waiting for the FastAPI service to become ready.")


def main() -> int:
    if not FRONTEND_DIST.exists():
        print("[smoke] frontend/dist is missing. Run `npm run build` in frontend first.")
        return 1

    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8012",
        ],
        cwd=BACKEND_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_until_ready()
        site_content = _get_json(f"{BASE_URL}/api/site-content")
        demo_cases = _get_json(f"{BASE_URL}/api/demo-cases")
        analysis = _post_json(
            f"{BASE_URL}/api/analyze",
            {
                "matrix": [[1.2, 0.4], [-0.3, 1.1]],
                "operation": "all",
                "b": [2, 1],
            },
        )
        with urllib.request.urlopen(f"{BASE_URL}/?view=defense", timeout=10) as response:  # noqa: S310
            homepage_status = response.status
        with urllib.request.urlopen(f"{BASE_URL}/favicon.svg", timeout=10) as response:  # noqa: S310
            favicon_status = response.status
        with urllib.request.urlopen(f"{BASE_URL}/site.webmanifest", timeout=10) as response:  # noqa: S310
            manifest_status = response.status

        print("[smoke] health ok")
        print(f"[smoke] homepage status: {homepage_status}")
        print(f"[smoke] favicon status: {favicon_status}")
        print(f"[smoke] manifest status: {manifest_status}")
        print(f"[smoke] site sections: {len(site_content.get('sections', []))}")
        print(f"[smoke] quickstart items: {len(site_content.get('quickstart', []))}")
        print(f"[smoke] demo cases: {len(demo_cases.get('items', []))}")
        print(f"[smoke] analysis result keys: {sorted(analysis.get('results', {}).keys())}")
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
