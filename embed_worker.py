#!/usr/bin/env python3
"""
Embedding worker (Airtable -> OpenAI embeddings -> Airtable)

What it does:
- Finds records in your "Client Style Library" table where {Embedding Status} = "Pending"
- Creates an embedding from {Chunk Text} using OpenAI embeddings
- Writes back:
    - {Embedding Vector} = JSON array (as a string)
    - {Embedding Status} = "Embedded"
    - {Embedding Error} = "" (cleared)
- On failure:
    - {Embedding Status} = "Failed"
    - {Embedding Error} = error message

Assumptions (field names must match Airtable exactly):
- Chunk Text (long text)
- Embedding Vector (long text)
- Embedding Status (single select): Pending / Embedded / Failed
- Embedding Error (long text)

Env vars required:
- AIRTABLE_TOKEN
- AIRTABLE_BASE_ID
- AIRTABLE_TABLE_ID_STYLE_LIBRARY   (table id or table name)
- OPENAI_API_KEY

Optional env vars:
- OPENAI_EMBED_MODEL (default: text-embedding-3-small)
- AIRTABLE_TIMEOUT (default: 20)
- OPENAI_TIMEOUT (default: 30)
- MAX_RECORDS (default: 100)
- SLEEP_BETWEEN (default: 0.2)  # seconds
"""

import os
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import requests

# ----------------------------
# Config
# ----------------------------
AIRTABLE_TOKEN = os.environ["AIRTABLE_TOKEN"]
AIRTABLE_BASE_ID = os.environ["AIRTABLE_BASE_ID"]
STYLE_TABLE_ID = os.environ["AIRTABLE_TABLE_ID_STYLE_LIBRARY"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

AIRTABLE_TIMEOUT = int(os.getenv("AIRTABLE_TIMEOUT", "20"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "100"))
SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0.2"))

# Airtable field names (must match exactly)
F_CHUNK_TEXT = "Chunk Text"
F_EMBED_VECTOR = "Embedding Vector"
F_EMBED_STATUS = "Embedding Status"
F_EMBED_ERROR = "Embedding Error"

STATUS_PENDING = "Pending"
STATUS_EMBEDDED = "Embedded"
STATUS_FAILED = "Failed"

AIRTABLE_API_BASE = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{STYLE_TABLE_ID}"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

# Reuse sessions
air_sess = requests.Session()
air_sess.headers.update({
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json",
})

# ----------------------------
# Helpers
# ----------------------------
def _safe_str(x: Any) -> str:
    s = str(x) if x is not None else ""
    return s[:10000]  # keep Airtable error fields reasonable

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def airtable_list_pending(page_size: int = 100) -> List[Dict[str, Any]]:
    """Fetch up to MAX_RECORDS pending records (handles pagination)."""
    records: List[Dict[str, Any]] = []
    offset: Optional[str] = None

    while True:
        params = {
            "pageSize": page_size,
            "filterByFormula": f"{{{F_EMBED_STATUS}}}='{STATUS_PENDING}'",
            # Optional: only fetch needed fields to reduce payload
            "fields[]": [F_CHUNK_TEXT, F_EMBED_STATUS],
        }
        if offset:
            params["offset"] = offset

        resp = air_sess.get(AIRTABLE_API_BASE, params=params, timeout=AIRTABLE_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("records", [])
        records.extend(batch)

        if len(records) >= MAX_RECORDS:
            return records[:MAX_RECORDS]

        offset = data.get("offset")
        if not offset:
            break

    return records

def airtable_patch(record_id: str, fields: Dict[str, Any]) -> None:
    url = f"{AIRTABLE_API_BASE}/{record_id}"
    resp = air_sess.patch(url, json={"fields": fields}, timeout=AIRTABLE_TIMEOUT)
    resp.raise_for_status()

def openai_create_embedding(text: str) -> List[float]:
    """
    Create an embedding using the OpenAI embeddings endpoint.
    Docs: https://platform.openai.com/docs/api-reference/embeddings/create
    Uses model text-embedding-3-small by default. :contentReference[oaicite:0]{index=0}
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_EMBED_MODEL,
        "input": text,
    }
    resp = requests.post(OPENAI_EMBED_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
    # Try to parse JSON even on error for better messages
    try:
        j = resp.json()
    except Exception:
        j = {"error": {"message": f"Non-JSON response: {resp.text[:300]}"}}

    if resp.status_code >= 400:
        msg = j.get("error", {}).get("message") or f"HTTP {resp.status_code}"
        raise RuntimeError(f"OpenAI embeddings error: {msg}")

    emb = j["data"][0]["embedding"]
    if not isinstance(emb, list) or not emb:
        raise RuntimeError("OpenAI embeddings: missing/invalid embedding array")
    return emb

# ----------------------------
# Main
# ----------------------------
def process_one(record: Dict[str, Any]) -> Tuple[bool, str]:
    record_id = record["id"]
    fields = record.get("fields", {}) or {}
    text = fields.get(F_CHUNK_TEXT)

    if not isinstance(text, str) or not text.strip():
        airtable_patch(record_id, {
            F_EMBED_STATUS: STATUS_FAILED,
            F_EMBED_ERROR: "Missing or empty Chunk Text",
        })
        return False, f"{record_id}: missing Chunk Text"

    try:
        # Embeddings input must not be empty; keep text as-is, but strip whitespace.
        clean = text.strip()

        emb = openai_create_embedding(clean)
        airtable_patch(record_id, {
            F_EMBED_VECTOR: json.dumps(emb, ensure_ascii=False),
            F_EMBED_STATUS: STATUS_EMBEDDED,
            F_EMBED_ERROR: "",
        })
        return True, f"{record_id}: embedded ({len(emb)} dims)"
    except Exception as e:
        airtable_patch(record_id, {
            F_EMBED_STATUS: STATUS_FAILED,
            F_EMBED_ERROR: _safe_str(e),
        })
        return False, f"{record_id}: FAILED - {e}"

def main() -> None:
    pending = airtable_list_pending(page_size=100)
    if not pending:
        print("No pending samples found.")
        return

    print(f"Found {len(pending)} pending sample(s).")

    ok = 0
    fail = 0
    for i, rec in enumerate(pending, start=1):
        success, msg = process_one(rec)
        print(f"[{i}/{len(pending)}] {msg}")
        if success:
            ok += 1
        else:
            fail += 1
        time.sleep(SLEEP_BETWEEN)

    print(f"Done. Embedded: {ok}, Failed: {fail}")

if __name__ == "__main__":
    main()
