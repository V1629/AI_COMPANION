"""
Data Layer: SQLite storage with encrypted sensitive fields.
Provides functions to initialize profiles, store interactions,
retrieve contextually similar history, and update aggregated user model.
"""
import sqlite3
import json
import time
from typing import Dict, Any, List, Optional
from .crypto_store import load_or_create_key, EncryptedCodec
from .embeddings import CharNGramHasher
import numpy as np
import os

CONFIG_PATH = "config.yaml"


def initialize_user_profile(db_path: str, enc_key_path: str) -> Dict[str, Any]:
    """
    Create database file if missing and initialize tables.
    Returns a minimal profile dictionary for the new user.
    """
    key = load_or_create_key(enc_key_path)
    codec = EncryptedCodec(key)
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    # tables: users, interactions, phrase_map, personality_model
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        created_at REAL,
        profile_json BLOB
    )
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        ts REAL,
        message_encrypted BLOB,
        metadata_encrypted BLOB,
        embedding BLOB,
        behavioral_json BLOB
    )
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS phrase_map (
        user_id TEXT,
        phrase TEXT,
        weight REAL,
        contexts_json BLOB,
        PRIMARY KEY(user_id, phrase)
    )
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS personality_model (
        user_id TEXT PRIMARY KEY,
        model_blob BLOB,
        updated_at REAL
    )
    """
    )
    conn.commit()
    conn.close()
    # minimal profile
    profile = {
        "user_id": "local_user",
        "created_at": time.time(),
        "communication_baseline": {},
        "emotional_baseline": {},
        "personality": {},
    }
    # store profile encrypted
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO users(user_id, created_at, profile_json) VALUES (?, ?, ?)",
                (profile["user_id"], profile["created_at"], codec.encrypt(json.dumps(profile))))
    conn.commit()
    conn.close()
    return profile


def _serialize_np(arr: np.ndarray) -> bytes:
    return arr.tobytes()


def _deserialize_np(b: bytes, dtype=np.float64, shape=None) -> np.ndarray:
    if shape is None:
        return np.frombuffer(b, dtype=dtype)
    return np.frombuffer(b, dtype=dtype).reshape(shape)


def store_interaction(db_path: str, enc_key_path: str, user_id: str, message: str,
                      metadata: Dict[str, Any], embedding: np.ndarray, behavioral: Dict[str, Any]) -> int:
    """
    Save conversation message with encrypted message and metadata, store embedding and behavioral JSON.
    Returns inserted interaction ID.
    """
    key = load_or_create_key(enc_key_path)
    codec = EncryptedCodec(key)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = time.time()
    cur.execute(
        "INSERT INTO interactions(user_id, ts, message_encrypted, metadata_encrypted, embedding, behavioral_json) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, ts, codec.encrypt(message), codec.encrypt(json.dumps(metadata)), _serialize_np(embedding.astype(np.float32)),
         codec.encrypt(json.dumps(behavioral))),
    )
    iid = cur.lastrowid
    conn.commit()
    conn.close()
    return iid


def retrieve_relevant_history(db_path: str, enc_key_path: str, user_id: str,
                              query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch top-k most similar past interactions by cosine similarity on stored embeddings.
    Returns list of dicts with decrypted message, metadata, ts and behavioral features.
    """
    key = load_or_create_key(enc_key_path)
    codec = EncryptedCodec(key)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, ts, message_encrypted, metadata_encrypted, embedding, behavioral_json FROM interactions WHERE user_id = ?", (user_id,))
    rows = cur.fetchall()
    results = []
    q = query_embedding.astype(np.float32)
    qnorm = np.linalg.norm(q) + 1e-9
    for r in rows:
        iid, ts, msg_enc, meta_enc, emb_blob, beh_enc = r
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = float(np.dot(q, emb) / (qnorm * (np.linalg.norm(emb) + 1e-9)))
        results.append((sim, iid, ts, msg_enc, meta_enc, beh_enc))
    results.sort(key=lambda x: x[0], reverse=True)
    out = []
    for sim, iid, ts, msg_enc, meta_enc, beh_enc in results[:top_k]:
        try:
            msg = codec.decrypt(msg_enc)
            meta = json.loads(codec.decrypt(meta_enc))
            beh = json.loads(codec.decrypt(beh_enc))
        except Exception:
            # if decryption fails, skip
            continue
        out.append({"id": iid, "ts": ts, "message": msg, "metadata": meta, "behavioral": beh, "sim": sim})
    conn.close()
    return out


def update_user_model(db_path: str, enc_key_path: str, user_id: str, updates: Dict[str, Any]) -> None:
    """
    Merge provided updates into the stored profile JSON for user.
    """
    key = load_or_create_key(enc_key_path)
    codec = EncryptedCodec(key)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT profile_json FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise ValueError("User not found")
    profile = json.loads(codec.decrypt(row[0]))
    # shallow merge, but allow nested dict updates
    def deep_merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                deep_merge(a[k], v)
            else:
                a[k] = v
    deep_merge(profile, updates)
    cur.execute("UPDATE users SET profile_json = ? WHERE user_id = ?", (codec.encrypt(json.dumps(profile)), user_id))
    conn.commit()
    conn.close()