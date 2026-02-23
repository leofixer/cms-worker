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
WORKER_VERSION = "v4-zerogpt-qc-routing-2026-02-23"

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
NEEDS_REVIEW_VALUE = os.getenv("NEEDS_REVIEW_VALUE", "Needs Review")  # if you add it to Status options

FINAL_TEXT_FIELD = os.getenv("FINAL_TEXT_FIELD", "Final Article Text")
LAST_ERROR_FIELD = os.getenv("LAST_ERROR_FIELD", "Last Error")

# ZeroGPT (QC step)
ZEROGPT_API_KEY = os.getenv("ZEROGPT_API_KEY")
ZEROGPT_API_URL = os.getenv("ZEROGPT_API_URL")  # YOU set this from zerogpt.com dashboard docs
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
    # Note: gpt-5 doesn't allow non-default temperature. Keep defaults.
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a professional content writer who follows instructions exactly."},
            {"role": "user", "content": prompt}
        ]
    )
    return (response.choices[0].message.content or "").strip()


# ==================================================
# ZeroGPT QC (store score + response; route to review)
# ==================================================
def zerogpt_detect(text: str) -> dict:
    """
    Calls ZeroGPT using the URL you set in ZEROGPT_API_URL.
    Because providers vary payload shapes, we:
    - send a simple JSON body containing the text
    - store raw JSON response
    - try to extract a numeric "score" from common keys
    """
    if not ZEROGPT_API_KEY or not ZEROGPT_API_URL:
        return {"enabled": False, "reason": "ZeroGPT not configured"}

    headers = {
        "Authorization": f"Bearer {ZEROGPT_API_KEY}",
        "Content-Type": "application/json"
    }

    # Most common pattern: send text in a field like "text" or "content".
    # If your ZeroGPT docs require a different key, tell me and I’ll adjust.
    payload = {"text": text}

    r = requests.post(ZEROGPT_API_URL, headers=headers, json=payload, timeout=60)
    raw_text = r.text

    if r.status_code >= 400:
        return {
            "enabled": True,
            "error": True,
            "http_status": r.status_code,
            "raw": raw_text[:20000]
        }

    try:
        data = r.json()
    except Exception:
        return {
            "enabled": True,
            "error": True,
            "http_status": r.status_code,
            "raw": raw_text[:20000]
        }

    # Try common keys for an AI % score
    # (You may need to tweak once we see your real response schema.)
    candidates = [
        ("ai_score",),
        ("score",),
        ("probability",),
        ("ai_probability",),
        ("data", "ai_score"),
        ("data", "score"),
        ("result", "ai_score"),
        ("result", "score"),
        ("prediction", "ai"),
        ("ai",),
    ]

    score = None
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
            # If it's a string like "23%" or "0.23", normalize
            try:
                if isinstance(cur, str):
                    val = cur.strip().replace("%", "")
                    score = float(val)
                else:
                    score = float(cur)
                break
            except Exception:
                pass

    return {
        "enabled": True,
        "error": False,
        "score": score,          # may be None until we map the exact schema
        "raw_json": data
    }


# ==================================================
# PROCESS ONE RECORD
# ==================================================
def process_one(record: dict):
    record_id = record["id"]
    fields = record.get("fields", {})

    update_record(record_id, {
        STATUS_FIELD: DRAFTING_VALUE,
        LAST_ERROR_FIELD: ""
    })

    prompt = build_prompt(fields)
    article = generate_article(prompt)

    if not article or len(article) < 200:
        raise RuntimeError("Generated article too short.")

    # Run ZeroGPT QC (if configured)
    qc = zerogpt_detect(article)

    patch = {
        FINAL_TEXT_FIELD: article
    }

    # Always store raw response (even if error) so you can debug
    if qc.get("enabled"):
        if qc.get("error"):
            patch[AI_RAW_FIELD] = json.dumps(qc, ensure_ascii=False)
            # Don’t fail the job because detector is down; route to review
            patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS
        else:
            patch[AI_RAW_FIELD] = json.dumps(qc.get("raw_json", {}), ensure_ascii=False)

            score = qc.get("score")
            if score is not None:
                patch[AI_SCORE_FIELD] = score

                # If score exceeds threshold, route to review (don’t auto “fix”)
                if score > ZEROGPT_THRESHOLD:
                    patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS
                else:
                    patch[REVIEW_STATUS_FIELD] = REVIEW_APPROVED
            else:
                # Could not parse score yet; still store raw
                patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS

    # Decide final Status
    # If Review Status is Needs Review, keep status as Delivered (or use Needs Review if you added it)
    if patch.get(REVIEW_STATUS_FIELD) == REVIEW_NEEDS:
        # Option A: Keep Delivered but flagged for review
        patch[STATUS_FIELD] = DELIVERED_VALUE

        # Option B (if you added "Needs Review" to Status single-select):
        # patch[STATUS_FIELD] = NEEDS_REVIEW_VALUE
    else:
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
    print("ZeroGPT enabled:", bool(ZEROGPT_API_KEY and ZEROGPT_API_URL), flush=True)
    if ZEROGPT_API_URL:
        print("ZeroGPT URL:", ZEROGPT_API_URL, flush=True)
    print("========================================", flush=True)

    while True:
        try:
            print("Polling Airtable...", flush=True)
            ready_records = list_ready(MAX_RECORDS_PER_CYCLE)
            print("Ready records found:", len(ready_records), flush=True)

            for record in ready_records:
                record_id = record["id"]
                try:
                    print("Processing:", record_id, flush=True)
                    process_one(record)
                    print("Delivered:", record_id, flush=True)
                except Exception as e:
                    error_msg = str(e)
                    print("FAILED:", record_id, error_msg, flush=True)
                    try:
                        update_record(record_id, {
                            STATUS_FIELD: FAILED_VALUE,
                            LAST_ERROR_FIELD: error_msg[:9000]
                        })
                    except Exception as inner:
                        print("Also failed to update Airtable error:", str(inner), flush=True)

        except Exception as cycle_error:
            print("Cycle error:", str(cycle_error), flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
