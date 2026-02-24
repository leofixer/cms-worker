import os
import time
import json
import requests
from typing import Any, Dict, Optional, Tuple

AIRTABLE_API_KEY = os.environ["AIRTABLE_API_KEY"]
AIRTABLE_BASE_ID = os.environ["AIRTABLE_BASE_ID"]
AIRTABLE_TABLE_ID = os.environ["AIRTABLE_TABLE_ID"]  # or table name
ZEROGPT_API_KEY = os.environ["ZEROGPT_API_KEY"]

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
QC_THRESHOLD = float(os.getenv("QC_THRESHOLD", "15"))  # MUST be 15 per your rule
AIRTABLE_TIMEOUT = int(os.getenv("AIRTABLE_TIMEOUT", "20"))
ZEROGPT_TIMEOUT = int(os.getenv("ZEROGPT_TIMEOUT", "30"))

AIRTABLE_API_BASE = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json",
})

def airtable_patch(record_id: str, fields: Dict[str, Any]) -> None:
    url = f"{AIRTABLE_API_BASE}/{record_id}"
    resp = SESSION.patch(url, json={"fields": fields}, timeout=AIRTABLE_TIMEOUT)
    resp.raise_for_status()

def airtable_find_ready_batch(max_records: int = 10) -> Dict[str, Any]:
    # Fetch records where Status = "Ready"
    # NOTE: Airtable requires single quotes around strings inside formula.
    params = {
        "pageSize": max_records,
        "filterByFormula": "{Status}='Ready'",
        "sort[0][field]": "Last Modified Time"  # optional if you have such a field
    }
    url = AIRTABLE_API_BASE
    resp = SESSION.get(url, params=params, timeout=AIRTABLE_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def zerogpt_detect(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (ai_score, raw_json)
    AI score is raw_json["data"]["fakePercentage"].
    """
    url = "https://api.zerogpt.com/api/detect/detectText"
    headers = {"ApiKey": ZEROGPT_API_KEY, "Content-Type": "application/json"}
    payload = {"input_text": text}

    resp = requests.post(url, headers=headers, json=payload, timeout=ZEROGPT_TIMEOUT)
    # Save raw JSON even when non-200 if possible
    try:
        raw = resp.json()
    except Exception:
        raw = {"success": False, "message": "Non-JSON response", "status_code": resp.status_code, "text": resp.text}

    resp.raise_for_status()

    # Your confirmed path:
    # raw_json.data.fakePercentage
    data = raw.get("data") or {}
    score = data.get("fakePercentage")

    if score is None:
        raise ValueError(f"ZeroGPT response missing data.fakePercentage. Raw: {raw}")

    return float(score), raw

def process_record(record: Dict[str, Any]) -> None:
    record_id = record["id"]
    fields = record.get("fields", {})

    article = fields.get("Final Article Text", "")
    if not isinstance(article, str) or not article.strip():
        airtable_patch(record_id, {
            "Status": "Failed",
            "Last Error": "Final Article Text is empty or missing.",
        })
        return

    # Mark Drafting while we analyze (optional but recommended)
    airtable_patch(record_id, {
        "Status": "Drafting",
        "Last Error": "",
    })

    raw_json: Optional[Dict[str, Any]] = None
    try:
        score, raw_json = zerogpt_detect(article)

        new_status = "Delivered" if score < QC_THRESHOLD else "Needs Review"

        airtable_patch(record_id, {
            "Status": new_status,
            "AI Score": score,
            "AI Raw Response": json.dumps(raw_json, ensure_ascii=False),
            "Last Error": "",
        })

    except Exception as e:
        patch_fields = {
            "Status": "Failed",
            "Last Error": str(e)[:10000],  # avoid huge error strings
        }
        if raw_json is not None:
            patch_fields["AI Raw Response"] = json.dumps(raw_json, ensure_ascii=False)

        airtable_patch(record_id, patch_fields)

def main() -> None:
    while True:
        try:
            batch = airtable_find_ready_batch(max_records=10)
            records = batch.get("records", [])

            if not records:
                time.sleep(POLL_SECONDS)
                continue

            for r in records:
                process_record(r)

        except Exception as loop_err:
            # Don’t crash the worker on transient errors
            print(f"[loop error] {loop_err}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
