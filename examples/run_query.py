"""
Interactive client that runs the local Flask app in-process and sends /message
requests so you can type a sample query, get the assistant response, edit your
query and re-run — without starting an external server.

Usage:
  python examples/run_query.py            # uses config.yaml in repo root
  python examples/run_query.py /path/to/config.yaml

This uses Flask's test_client() so everything runs in-process and keeps data
local (privacy-first). It will create DB/key if missing (per config).
"""
import sys
import json
import time
import yaml
from typing import Optional
from ai_companion.api import create_app


def pretty_print(resp_json: dict) -> None:
    print("\n--- Assistant Response ---")
    print(json.dumps(resp_json, indent=2, ensure_ascii=False))
    print("--------------------------\n")


def run_interactive(config_path: Optional[str] = "config.yaml") -> None:
    # Load config to ensure path validity (create_app will also load it)
    cfg = yaml.safe_load(open(config_path))
    app = create_app(config_path)

    print("AI Companion interactive client (in-process). Type messages and press Enter.")
    print("Type 'exit' or Ctrl+C to quit.\n")

    client = app.test_client()
    try:
        while True:
            message = input("You: ").strip()
            if not message:
                continue
            if message.lower() in ("exit", "quit"):
                print("Exiting.")
                break

            # naive typing time estimate: 50 cps baseline, add a bit of jitter
            typing_time_ms = max(30, int(len(message) / 5 * 100))  # simple heuristic

            payload = {
                "message": message,
                "typing_time_ms": typing_time_ms,
                # optional metadata you can edit inline: e.g., prefers_direct
                "metadata": {"prefers_direct": False}
            }
            resp = client.post("/message", json=payload)
            if resp.status_code != 200:
                print(f"Error ({resp.status_code}): {resp.get_data(as_text=True)}")
                continue
            data = resp.get_json()
            pretty_print(data)

            # Give you the option to modify the last message and re-run quickly
            edit = input("Edit and re-run? (enter new message or press Enter to continue) ").strip()
            if edit:
                message = edit
                payload["message"] = message
                payload["typing_time_ms"] = max(30, int(len(message) / 5 * 100))
                resp = client.post("/message", json=payload)
                data = resp.get_json()
                pretty_print(data)

    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_interactive(cfg_path)