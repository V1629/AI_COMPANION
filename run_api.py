"""
Run the AI Companion Flask API locally.

Usage:
  python run_api.py

This will start the app on http://127.0.0.1:5000
"""
from ai_companion.api import create_app

app = create_app("config.yaml")

if __name__ == "__main__":
    # In production use gunicorn / uvicorn behind a reverse proxy.
    app.run(host="127.0.0.1", port=5000, debug=True)