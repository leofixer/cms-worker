import os
import time
import json
import requests
from urllib.parse import quote

import httpx
from openai import OpenAI

# ==================================================
# VERSION
# ==================================================
WORKER_VERSION = "v9-qc-under15-needsreview-2026-02-24"

# ==================================================
# REQUIRED ENV VARS (Render)
# ==================================================
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")  # Airtable Personal Access Token
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
NEEDS_REVIEW_VALUE = os.getenv("NEEDS_REVIEW_VALUE", "Needs Review")

FINAL_TEXT_FIELD = os.getenv("FINAL_TEXT_FIELD", "Final Article Text")
LAST_ERROR_FIELD = os.getenv("LAST_ERROR_FIELD", "Last Error")

# optional timestamp field (recommended)
DRAFTING_STARTED_AT_FIELD = os.getenv("DRAFTING_STARTED_AT_FIELD", "Drafting Started At")

# record watchdog
RECORD_TIMEOUT_SECONDS = int(os.getenv("RECORD_TIMEOUT_SECONDS", "240"))  # 4 min

# ==================================================
# ZeroGPT (per your docs)
# ==================================================
ZEROGPT_API_KEY = os.getenv("ZEROGPT_API_KEY")
ZEROGPT_API_URL = os.getenv("ZEROGPT_API_URL", "https://api.zerogpt.com/api/detect/detectText")
ZEROGPT_THRESHOLD = float(os.getenv("ZEROGPT_THRESHOLD", "15"))  # must be < 15 to deliver

AI_SCORE_FIELD = os.getenv("AI_SCORE_FIELD", "AI Score")
AI_RAW_FIELD = os.getenv("AI_RAW_FIELD", "AI Raw Response")


def require_env():
    missing = []
    required = {
        "AIRTABLE_TOKEN": AIRTABLE_TOKEN,
        "AIRTABLE_BASE_ID": AIRTABLE_BASE_ID,
        "AIRTABLE_TABLE": AIRTABLE_TABLE,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "ZEROGPT_API_KEY": ZEROGPT_API_KEY,
    }
    for k, v in required.items():
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


require_env()

# ==================================================
# OPENAI CLIENT (proxy-safe)
# ==================================================
http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, connect=20.0, read=60.0, write=60.0, pool=60.0),
    follow_redirects=True,
    trust_env=False,
)
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)


# ==================================================
# AIRTABLE HELPERS
# ==================================================
def airtable_headers():
    return {"Authorization": f"Bearer {AIRTABLE_TOKEN}", "Content-Type": "application/json"}


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


def update_record(record_id: str, fields: dict):
    url = f"{airtable_base_url()}/{record_id}"
    r = requests.patch(url, headers=airtable_headers(), json={"fields": fields}, timeout=30)
    if r.status_code != 200:
        print("Airtable PATCH URL:", url, flush=True)
        print("Airtable PATCH status:", r.status_code, flush=True)
        print("Airtable PATCH response:", r.text, flush=True)
        r.raise_for_status()
    return r.json()


def safe_mark_failed(record_id: str, msg: str):
    msg = (msg or "Unknown error")[:9000]
    try:
        update_record(record_id, {STATUS_FIELD: FAILED_VALUE, LAST_ERROR_FIELD: msg})
    except Exception as e:
        print("Failed to mark failed:", str(e), flush=True)
        try:
            update_record(record_id, {STATUS_FIELD: FAILED_VALUE})
        except Exception as e2:
            print("Failed even minimal mark failed:", str(e2), flush=True)


def list_ready(max_records: int):
    formula = f'{{{STATUS_FIELD}}}="{READY_VALUE}"'
    data = airtable_get({"maxRecords": max_records, "filterByFormula": formula})
    return data.get("records", [])


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

    return f"""
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
- Do not use bullet lists or hyphen bullets.
- Avoid repetitive filler.
- Do not mention AI or detectors.
""".strip()


def generate_article(prompt: str) -> str:
    # gpt-5: defaults only (no temperature)
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a professional content writer who follows instructions exactly."},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ==================================================
# ZeroGPT DETECTION (correct request format)
# ==================================================
def zerogpt_detect(text: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "ApiKey": ZEROGPT_API_KEY,  # per docs
    }
    payload = {"input_text": text}  # per docs

    r = requests.post(ZEROGPT_API_URL, headers=headers, json=payload, timeout=60)
    raw = r.text

    try:
        data = r.json()
    except Exception:
        return {"enabled": True, "error": True, "http_status": r.status_code, "raw": raw[:20000]}

    # API uses success boolean in JSON
    if isinstance(data, dict) and data.get("success") is False:
        return {"enabled": True, "error": True, "http_status": data.get("code", r.status_code), "raw_json": data}

    # score = fakePercentage (AI-likelihood %)
    score = None
    try:
        d = data.get("data") if isinstance(data, dict) else None
        if isinstance(d, dict) and "fakePercentage" in d:
            v = str(d["fakePercentage"]).strip().replace("%", "")
            score = float(v)
    except Exception:
        score = None

    return {"enabled": True, "error": False, "score": score, "raw_json": data}


# ==================================================
# PROCESS ONE RECORD
# ==================================================
def process_one(record: dict):
    record_id = record["id"]
    fields = record.get("fields", {})

    start = time.time()

    def check_timeout(step: str):
        if time.time() - start > RECORD_TIMEOUT_SECONDS:
            raise TimeoutError(f"Timeout after {RECORD_TIMEOUT_SECONDS}s at step: {step}")

    # set Drafting + timestamp
    drafting_patch = {
        STATUS_FIELD: DRAFTING_VALUE,
        LAST_ERROR_FIELD: "",
        DRAFTING_STARTED_AT_FIELD: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("Setting Drafting:", record_id, flush=True)
    update_record(record_id, drafting_patch)
    check_timeout("drafting_patch")

    prompt = build_prompt(fields)

    print("OpenAI start:", record_id, flush=True)
    article = generate_article(prompt)
    print("OpenAI done:", record_id, "len:", len(article), flush=True)
    check_timeout("openai_done")

    if not article or len(article) < 200:
        raise RuntimeError("Generated article too short.")

    print("ZeroGPT start:", record_id, flush=True)
    qc = zerogpt_detect(article)
    print("ZeroGPT done:", record_id, "qc_error:", qc.get("error"), "score:", qc.get("score"), flush=True)
    check_timeout("zerogpt_done")

    # Always store raw response + article
    patch = {
        FINAL_TEXT_FIELD: article,
        AI_RAW_FIELD: json.dumps(qc, ensure_ascii=False),
    }

    if qc.get("error"):
        # Detector error → Needs Review (do not deliver)
        patch[STATUS_FIELD] = NEEDS_REVIEW_VALUE
        patch[LAST_ERROR_FIELD] = f"ZeroGPT error: {qc.get('http_status')}"
        update_record(record_id, patch)
        return

    score = qc.get("score")
    if score is None:
        patch[STATUS_FIELD] = NEEDS_REVIEW_VALUE
        patch[LAST_ERROR_FIELD] = "ZeroGPT did not return a score."
        update_record(record_id, patch)
        return

    patch[AI_SCORE_FIELD] = float(score)

    # RULE: MUST be under 15 to be acceptable
    if float(score) >= ZEROGPT_THRESHOLD:
        patch[STATUS_FIELD] = NEEDS_REVIEW_VALUE
    else:
        patch[STATUS_FIELD] = DELIVERED_VALUE

    print("Writing final status:", record_id, patch[STATUS_FIELD], flush=True)
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
    print("POLL_SECONDS:", POLL_SECONDS, "MAX_RECORDS_PER_CYCLE:", MAX_RECORDS_PER_CYCLE, flush=True)
    print("ZeroGPT URL:", ZEROGPT_API_URL, flush=True)
    print("ZeroGPT threshold (<):", ZEROGPT_THRESHOLD, flush=True)
    print("========================================", flush=True)

    while True:
        try:
            print("Polling Airtable...", flush=True)
            ready = list_ready(MAX_RECORDS_PER_CYCLE)
            print("Ready records found:", len(ready), flush=True)

            for rec in ready:
                rid = rec["id"]
                try:
                    print("Processing:", rid, flush=True)
                    process_one(rec)
                except Exception as e:
                    print("FAILED:", rid, str(e), flush=True)
                    safe_mark_failed(rid, str(e))

        except Exception as cycle_error:
            print("Cycle error:", str(cycle_error), flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
