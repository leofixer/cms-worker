import os
import time
import json
import requests
from urllib.parse import quote

import httpx
from openai import OpenAI

# ==================================================
# VERSION STAMP
# ==================================================
WORKER_VERSION = "v8-zerogpt-docs-apikey-inputtext-2026-02-24"

# ==================================================
# REQUIRED ENV VARS
# ==================================================
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")  # tbl... recommended
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

# Optional timestamp field in Airtable (recommended)
DRAFTING_STARTED_AT_FIELD = os.getenv("DRAFTING_STARTED_AT_FIELD", "Drafting Started At")

# Prevent infinite hangs per record
RECORD_TIMEOUT_SECONDS = int(os.getenv("RECORD_TIMEOUT_SECONDS", "240"))  # 4 minutes

# Auto-fail Drafting if older than this (minutes). Set 0 to disable.
STUCK_MINUTES = int(os.getenv("STUCK_MINUTES", "15"))

# ==================================================
# ZeroGPT (per your uploaded docs)
# Base URL: https://api.zerogpt.com
# Endpoint: POST /api/detect/detectText
# Header: ApiKey: <key>
# Body: { "input_text": "..." }
# Response: data.fakePercentage is the AI% (per docs)
# ==================================================
ZEROGPT_API_KEY = os.getenv("ZEROGPT_API_KEY")
ZEROGPT_API_URL = os.getenv("ZEROGPT_API_URL", "https://api.zerogpt.com/api/detect/detectText")
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

# ==================================================
# OPENAI CLIENT (proxy-safe)
# ==================================================
http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, connect=20.0, read=60.0, write=60.0, pool=60.0),
    follow_redirects=True,
    trust_env=False
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
        return
    except Exception as e1:
        print("Failed to write full failure update:", str(e1), flush=True)

    try:
        update_record(record_id, {STATUS_FIELD: FAILED_VALUE})
        return
    except Exception as e2:
        print("Failed to write minimal failure update:", str(e2), flush=True)


def list_ready(max_records: int):
    filter_formula = f'{{{STATUS_FIELD}}}="{READY_VALUE}"'
    params = {"maxRecords": max_records, "filterByFormula": filter_formula}
    data = airtable_get(params)
    return data.get("records", [])


def list_drafting(max_records: int):
    filter_formula = f'{{{STATUS_FIELD}}}="{DRAFTING_VALUE}"'
    params = {"maxRecords": max_records, "filterByFormula": filter_formula}
    data = airtable_get(params)
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
    # gpt-5: keep default parameters (no temperature)
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a professional content writer who follows instructions exactly."},
            {"role": "user", "content": prompt}
        ]
    )
    return (resp.choices[0].message.content or "").strip()


# ==================================================
# ZeroGPT DETECTION (matches your ZeroGPT docs)
# ==================================================
def zerogpt_detect(text: str) -> dict:
    if not ZEROGPT_API_KEY:
        return {"enabled": False, "reason": "ZEROGPT_API_KEY not set"}

    headers = {
        "Content-Type": "application/json",
        "ApiKey": ZEROGPT_API_KEY,          # <-- per docs
    }
    payload = {
        "input_text": text                 # <-- per docs
    }

    r = requests.post(ZEROGPT_API_URL, headers=headers, json=payload, timeout=60)
    raw = r.text

    try:
        data = r.json()
    except Exception:
        return {
            "enabled": True,
            "error": True,
            "http_status": r.status_code,
            "raw": raw[:20000],
        }

    # Treat success:false as error (even if HTTP 200)
    if isinstance(data, dict) and data.get("success") is False:
        return {
            "enabled": True,
            "error": True,
            "http_status": data.get("code", r.status_code),
            "raw_json": data,
        }

    # Extract score per docs: data.fakePercentage
    score = None
    try:
        if isinstance(data, dict):
            d = data.get("data")
            if isinstance(d, dict) and "fakePercentage" in d:
                v = str(d["fakePercentage"]).strip().replace("%", "")
                score = float(v)
    except Exception:
        score = None

    return {
        "enabled": True,
        "error": False,
        "score": score,
        "raw_json": data,
    }


# ==================================================
# PROCESS ONE RECORD (WITH WATCHDOG)
# ==================================================
def process_one(record: dict):
    record_id = record["id"]
    fields = record.get("fields", {})

    start = time.time()

    def check_timeout(step_name: str):
        if time.time() - start > RECORD_TIMEOUT_SECONDS:
            raise TimeoutError(f"Record timed out after {RECORD_TIMEOUT_SECONDS}s during: {step_name}")

    # Mark Drafting + timestamp
    drafting_patch = {
        STATUS_FIELD: DRAFTING_VALUE,
        LAST_ERROR_FIELD: "",
        DRAFTING_STARTED_AT_FIELD: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print("Setting Drafting:", record_id, flush=True)
    update_record(record_id, drafting_patch)
    check_timeout("after drafting patch")

    prompt = build_prompt(fields)

    print("OpenAI start:", record_id, flush=True)
    article = generate_article(prompt)
    print("OpenAI done:", record_id, "len:", len(article), flush=True)
    check_timeout("after OpenAI")

    if not article or len(article) < 200:
        raise RuntimeError("Generated article too short.")

    qc = {}
    if ZEROGPT_API_KEY:
        print("ZeroGPT start:", record_id, flush=True)
        qc = zerogpt_detect(article)
        print("ZeroGPT done:", record_id, "qc_error:", qc.get("error"), "score:", qc.get("score"), flush=True)
        check_timeout("after ZeroGPT")

    # Final patch
    final_patch = {
        FINAL_TEXT_FIELD: article,
        STATUS_FIELD: DELIVERED_VALUE,
    }

    if qc:
        final_patch[AI_RAW_FIELD] = json.dumps(qc, ensure_ascii=False)

        if qc.get("error"):
            final_patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS
        else:
            score = qc.get("score")
            if score is not None:
                final_patch[AI_SCORE_FIELD] = float(score)
                final_patch[REVIEW_STATUS_FIELD] = (
                    REVIEW_NEEDS if float(score) > ZEROGPT_THRESHOLD else REVIEW_APPROVED
                )
            else:
                final_patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS

    print("Writing Delivered:", record_id, flush=True)
    update_record(record_id, final_patch)


# ==================================================
# STUCK DRAFTING RECOVERY (OPTIONAL)
# ==================================================
def stuck_recovery():
    if STUCK_MINUTES <= 0:
        return

    try:
        drafting = list_drafting(10)
    except Exception as e:
        print("Stuck recovery: failed to list drafting:", str(e), flush=True)
        return

    now = time.time()
    for rec in drafting:
        rid = rec["id"]
        f = rec.get("fields", {})
        started = f.get(DRAFTING_STARTED_AT_FIELD)
        if not started:
            continue

        # Airtable may return e.g. 2026-02-24T09:12:30.000Z
        ts = started.replace(".000Z", "Z")

        try:
            t_struct = time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            started_epoch = time.mktime(t_struct)  # local; good enough for stuck detection
            age_min = (now - started_epoch) / 60.0
            if age_min >= STUCK_MINUTES:
                safe_mark_failed(rid, f"Stuck in Drafting for {int(age_min)} minutes. Auto-failed.")
        except Exception:
            continue


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
    print("RECORD_TIMEOUT_SECONDS:", RECORD_TIMEOUT_SECONDS, flush=True)
    print("ZeroGPT URL:", ZEROGPT_API_URL, flush=True)
    print("ZeroGPT enabled:", bool(ZEROGPT_API_KEY), flush=True)
    print("========================================", flush=True)

    while True:
        try:
            stuck_recovery()

            print("Polling Airtable...", flush=True)
            ready = list_ready(MAX_RECORDS_PER_CYCLE)
            print("Ready records found:", len(ready), flush=True)

            for rec in ready:
                rid = rec["id"]
                try:
                    print("Processing:", rid, flush=True)
                    process_one(rec)
                    print("Delivered:", rid, flush=True)
                except Exception as e:
                    print("FAILED:", rid, str(e), flush=True)
                    safe_mark_failed(rid, str(e))

        except Exception as cycle_error:
            print("Cycle error:", str(cycle_error), flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
