"""
Smoke test for the OmniParser API: posts a base64 encoded screenshot to /parse/
and prints the number of unique icons found.

Usage (from anywhere): python test/test.py
Reads OMNIPARSER_HOST / OMNIPARSER_PORT from the environment or the repo-root .env.
"""
import base64
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = REPO_ROOT / "images" / "windows_desktop.png"
REQUEST_TIMEOUT = (5, 120)  # (connect, read) seconds; the first call can be slow on GPU warm-up

load_dotenv(REPO_ROOT / ".env")


def test_run(num_of_testing: int = 10) -> None:
    """
    With NVIDIA GeForce RTX 4050 Laptop GPU, this might take around 22 seconds
    """
    host = os.getenv("OMNIPARSER_HOST", "localhost")
    port = os.getenv("OMNIPARSER_PORT", "8081")
    # 0.0.0.0 is a bind address, not a valid destination on every OS
    if host == "0.0.0.0":
        host = "localhost"
    florence_url = f"http://{host}:{port}/parse/"
    print("=" * 66)
    print(f"Florence API URL: {florence_url}")
    print("=" * 66)

    base64_image = base64.b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")
    failures = 0
    for i in range(num_of_testing):
        started = time.time()
        try:
            response = requests.post(
                florence_url, json={"base64_image": base64_image}, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            print(f"[{i + 1}/{num_of_testing}] FAILED: {exc}")
            failures += 1
            continue

        elapsed = time.time() - started
        if response.status_code != 200:
            print(f"[{i + 1}/{num_of_testing}] FAILED: HTTP {response.status_code} {response.text[:200]}")
            failures += 1
            continue

        parsed = response.json().get("parsed_content_list", [])
        icons = {item["content"] for item in parsed if item.get("type") == "icon"}
        print(f"[{i + 1}/{num_of_testing}] OK in {elapsed:.1f}s - {len(parsed)} elements, {len(icons)} unique icons")

    assert failures == 0, f"{failures}/{num_of_testing} requests failed"


if __name__ == "__main__":
    start_time = time.time()
    try:
        test_run()
        ok = True
    except AssertionError as exc:
        print(exc)
        ok = False
    print("=" * 33)
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 33)
    sys.exit(0 if ok else 1)
