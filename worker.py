import os
import time
import json
import requests
from urllib.parse import quote

import httpx
from openai import OpenAI
import requests
print("Server IP:", requests.get("https://api.ipify.org").text, flush=True)

# ==================================================
# VERSION STAMP
# ==================================================
WORKER_VERSION = "v5-zerogpt-detectText-2026-02-23"

# ==================================================
# REQUIRED ENV VARS
# ==================================================
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")  # tblXXXXXXXX recommended
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==================================================
# OPTIONAL ENV VARS
# ==================================================
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "120"))
MAX_RECORDS_PER_CYCLE = int(os.getenv("MAX_RECORDS_PER_CYCLE", "3"))

STATUS_FIELD = os.getenv("STATUS_FIELD", "Status")
READY_VALUE = os.getenv("READY_VALUE", "Ready")
DRAFTING_VALUE = os.getenv("DRAFTING_VALUE", "Drafting")
DELIVERED_VALUE = os.getenv("DELIVERED_VALUE", "Delivered")
FAILED_VALUE = os.getenv("FAILED_VALUE", "Failed")

FINAL_TEXT_FIELD = os.getenv("FINAL_TEXT_FIELD", "Final Article Text")
LAST_ERROR_FIELD = os.getenv("LAST_ERROR_FIELD", "Last Error")

# ==================================================
# ZeroGPT (QC step)
# ==================================================
ZEROGPT_API_KEY = os.getenv("ZEROGPT_API_KEY")

# Default to the commonly referenced endpoint; you can override in Render env var.
ZEROGPT_API_URL = os.getenv(
    "ZEROGPT_API_URL",
    "https://api.zerogpt.com/api/detect/detectText"
)

ZEROGPT_THRESHOLD = float(os.getenv("ZEROGPT_THRESHOLD", "15"))

AI_SCORE_FIELD = os.getenv("AI_SCORE_FIELD", "AI Score")
AI_RAW_FIELD = os.getenv("AI_RAW_FIELD", "AI Raw Response")
REVIEW_STATUS_FIELD = os.getenv("REVIEW_STATUS_FIELD", "Review Status")
REVIEW_APPROVED = os.getenv("REVIEW_APPROVED", "Approved")
REVIEW_NEEDS = os.getenv("REVIEW_NEEDS", "Needs Review")


def require_env():
    missing = []
    for k, v in {
        "AIRTABLE_TOKEN": AIRTABLE_TOKEN,
        "AIRTABLE_BASE_ID": AIRTABLE_BASE_ID,
        "AIRTABLE_TABLE": AIRTABLE_TABLE,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }.items():
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


require_env()

# OpenAI client (proxy-safe)
http_client = httpx.Client(timeout=60.0, follow_redirects=True, trust_env=False)
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)


# ==================================================
# AIRTABLE HELPERS
# ==================================================
def airtable_headers():
    return {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json"
    }


def airtable_base_url():
    table = quote(AIRTABLE_TABLE, safe="")
    return f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table}"


def airtable_get(params: dict):
    url = airtable_base_url()
    r = requests.get(url, headers=airtable_headers(), params=params, timeout=30)
    if r.status_code != 200:
        print("Airtable GET URL:", r.url, flush=True)
        print("Airtable GET status:", r.status_code, flush=True)
        print("Airtable GET response:", r.text, flush=True)
        r.raise_for_status()
    return r.json()


def list_ready(max_records: int):
    filter_formula = f'{{{STATUS_FIELD}}}="{READY_VALUE}"'
    params = {"maxRecords": max_records, "filterByFormula": filter_formula}
    data = airtable_get(params)
    return data.get("records", [])


def update_record(record_id: str, fields: dict):
    url = f"{airtable_base_url()}/{record_id}"
    r = requests.patch(url, headers=airtable_headers(), json={"fields": fields}, timeout=30)
    if r.status_code != 200:
        print("Airtable PATCH URL:", url, flush=True)
        print("Airtable PATCH status:", r.status_code, flush=True)
        print("Airtable PATCH response:", r.text, flush=True)
        r.raise_for_status()
    return r.json()


# ==================================================
# CONTENT GENERATION
# ==================================================
def build_prompt(fields: dict) -> str:
    topic = fields.get("Topics") or fields.get("Topic") or "Write an original article."
    word_count = fields.get("word count") or fields.get("Word Count") or 700
    tone = fields.get("Tone") or ""
    special = fields.get("Special Content Instructions") or ""

    anchor = fields.get("Anchor Text")
    target_url = fields.get("Target URL")

    link_rule = ""
    if anchor and target_url:
        link_rule = f'Include this exact anchor text once: "{anchor}" linking to {target_url}. Make it natural.'

    prompt = f"""
Write an original article.

Topic:
{topic}

Target length:
About {word_count} words.

Tone:
{tone}

Special instructions:
{special}

Link requirement:
{link_rule}

Formatting rules:
- Put a clear title on the first line.
- Use subheadings.
- Avoid repetitive filler.
- Do not mention AI or detectors.
""".strip()
    return prompt


def generate_article(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a professional content writer who follows instructions exactly."},
            {"role": "user", "content": prompt}
        ]
    )
    return (response.choices[0].message.content or "").strip()


# ==================================================
# ZeroGPT QC
# ==================================================
def _extract_score_from_zerogpt(data: dict):
    """
    Try common response shapes:
    - isHuman: boolean
    - aiPercentage / ai_percentage: number 0-100
    - score/probability: 0-1 or 0-100
    """
    # direct booleans
    if isinstance(data, dict) and "isHuman" in data and isinstance(data["isHuman"], bool):
        # If it's human, AI score ~ 0; if not, AI score ~ 100 (coarse)
        return 0.0 if data["isHuman"] else 100.0

    # common % keys
    for k in ["aiPercentage", "ai_percentage", "aiScore", "ai_score"]:
        if k in data:
            try:
                v = data[k]
                if isinstance(v, str):
                    v = v.strip().replace("%", "")
                return float(v)
            except Exception:
                pass

    # generic nested keys
    candidates = [
        ("data", "aiPercentage"),
        ("data", "ai_percentage"),
        ("result", "aiPercentage"),
        ("result", "ai_percentage"),
        ("result", "score"),
        ("score",),
        ("probability",),
    ]
    for path in candidates:
        cur = data
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            try:
                if isinstance(cur, str):
                    cur = cur.strip().replace("%", "")
                val = float(cur)
                # If returned 0-1, convert to %
                if 0.0 <= val <= 1.0:
                    return val * 100.0
                return val
            except Exception:
                pass

    return None


def zerogpt_detect(text: str) -> dict:
    if not ZEROGPT_API_KEY:
        return {"enabled": False, "reason": "ZEROGPT_API_KEY not set"}

    # Try both common auth styles
    auth_headers_list = [
        {"Authorization": f"Bearer {ZEROGPT_API_KEY}"},
        {"x-api-key": ZEROGPT_API_KEY},
    ]

    payloads = [
        {"text": text},
        {"input_text": text},
        {"document": text},
    ]

    last_error = None

    for auth_headers in auth_headers_list:
        headers = {"Content-Type": "application/json", **auth_headers}

        for payload in payloads:
            try:
                r = requests.post(ZEROGPT_API_URL, headers=headers, json=payload, timeout=60)
                raw = r.text

                if r.status_code >= 400:
                    last_error = {
                        "enabled": True,
                        "error": True,
                        "http_status": r.status_code,
                        "raw": raw[:20000],
                        "tried_payload_keys": list(payload.keys()),
                        "tried_auth": list(auth_headers.keys())[0],
                        "url": ZEROGPT_API_URL
                    }
                    continue

                try:
                    data = r.json()
                except Exception:
                    last_error = {
                        "enabled": True,
                        "error": True,
                        "http_status": r.status_code,
                        "raw": raw[:20000],
                        "tried_payload_keys": list(payload.keys()),
                        "tried_auth": list(auth_headers.keys())[0],
                        "url": ZEROGPT_API_URL
                    }
                    continue

                score = _extract_score_from_zerogpt(data)
                return {
                    "enabled": True,
                    "error": False,
                    "score": score,
                    "raw_json": data,
                    "url": ZEROGPT_API_URL,
                    "used_payload_keys": list(payload.keys()),
                    "used_auth": list(auth_headers.keys())[0]
                }

            except Exception as e:
                last_error = {
                    "enabled": True,
                    "error": True,
                    "exception": str(e),
                    "tried_payload_keys": list(payload.keys()),
                    "tried_auth": list(auth_headers.keys())[0],
                    "url": ZEROGPT_API_URL
                }

    return last_error or {"enabled": True, "error": True, "raw": "Unknown ZeroGPT failure"}


# ==================================================
# PROCESS ONE RECORD
# ==================================================
def process_one(record: dict):
    record_id = record["id"]
    fields = record.get("fields", {})

    update_record(record_id, {STATUS_FIELD: DRAFTING_VALUE, LAST_ERROR_FIELD: ""})

    prompt = build_prompt(fields)
    article = generate_article(prompt)

    if not article or len(article) < 200:
        raise RuntimeError("Generated article too short.")

    qc = zerogpt_detect(article)

    patch = {
        FINAL_TEXT_FIELD: article,
        AI_RAW_FIELD: json.dumps(qc, ensure_ascii=False)
    }

    if qc.get("enabled") and not qc.get("error"):
        score = qc.get("score")
        if score is not None:
            patch[AI_SCORE_FIELD] = float(score)

            # QC routing only (no auto “rewrite until pass” loops)
            if float(score) > ZEROGPT_THRESHOLD:
                patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS
            else:
                patch[REVIEW_STATUS_FIELD] = REVIEW_APPROVED
        else:
            patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS
    else:
        # If detector fails, route to review (don’t block delivery entirely)
        patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS

    patch[STATUS_FIELD] = DELIVERED_VALUE
    update_record(record_id, patch)


# ==================================================
# MAIN LOOP
# ==================================================
def main():
    print("========================================", flush=True)
    print("WORKER_VERSION:", WORKER_VERSION, flush=True)
    print("BASE:", AIRTABLE_BASE_ID, flush=True)
    print("TABLE:", AIRTABLE_TABLE, flush=True)
    print("STATUS_FIELD:", STATUS_FIELD, "READY_VALUE:", READY_VALUE, flush=True)
    print("POLL_SECONDS:", POLL_SECONDS, flush=True)
    print("MAX_RECORDS_PER_CYCLE:", MAX_RECORDS_PER_CYCLE, flush=True)
    print("httpx version:", httpx.__version__, flush=True)
    print("ZeroGPT URL:", ZEROGPT_API_URL, flush=True)
    print("ZeroGPT enabled:", bool(ZEROGPT_API_KEY), flush=True)
    print("========================================", flush=True)

    while True:
        try:
            print("Polling Airtable...", flush=True)
            ready_records = list_ready(MAX_RECORDS_PER_CYCLE)
            print("Ready records found:", len(ready_records), flush=True)

            for record in ready_records:
                rid = record["id"]
                try:
                    print("Processing:", rid, flush=True)
                    process_one(record)
                    print("Delivered:", rid, flush=True)
                except Exception as e:
                    msg = str(e)
                    print("FAILED:", rid, msg, flush=True)
                    try:
                        update_record(rid, {STATUS_FIELD: FAILED_VALUE, LAST_ERROR_FIELD: msg[:9000]})
                    except Exception as inner:
                        print("Also failed to update Airtable error:", str(inner), flush=True)

        except Exception as cycle_error:
            print("Cycle error:", str(cycle_error), flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

