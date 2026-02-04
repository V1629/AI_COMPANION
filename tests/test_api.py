import json
import tempfile
import os
from ai_companion.api import create_app
from ai_companion.data_layer import initialize_user_profile
import yaml

def test_message_endpoint(tmp_path):
    # create temporary config and DB to avoid clobbering local files
    cfg = {
        "database": {"path": str(tmp_path / "test_db.sqlite")},
        "encryption": {"key_file": str(tmp_path / "test_key.key")},
        "embeddings": {"dim": 64, "ngram_min": 3, "ngram_max": 4},
        "learning": {"short_term_window": 50, "long_term_window": 500}
    }
    cfg_path = tmp_path / "config_test.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)

    app = create_app(str(cfg_path))
    client = app.test_client()

    # Post a message
    resp = client.post("/message", json={"message": "I'm fine", "typing_time_ms": 300})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "response" in data
    assert "intent" in data
    assert "emotional_probs" in data

def test_health(tmp_path):
    cfg_path = tmp_path / "config_test.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump({
            "database": {"path": str(tmp_path / "test_db.sqlite")},
            "encryption": {"key_file": str(tmp_path / "test_key.key")},
            "embeddings": {"dim": 32, "ngram_min": 3, "ngram_max": 4}
        }, fh)
    app = create_app(str(cfg_path))
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"