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
WORKER_VERSION = "v6-zerogpt-success-fix-2026-02-23"

# ==================================================
# REQUIRED ENV VARS
# ==================================================
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")
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
# ZeroGPT
# ==================================================
ZEROGPT_API_KEY = os.getenv("ZEROGPT_API_KEY")
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

# OpenAI client
http_client = httpx.Client(timeout=60.0, follow_redirects=True, trust_env=False)
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)


# ==================================================
# Airtable helpers
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
    r = requests.get(
        airtable_base_url(),
        headers=airtable_headers(),
        params=params,
        timeout=30
    )
    if r.status_code != 200:
        print("Airtable GET error:", r.text, flush=True)
        r.raise_for_status()
    return r.json()


def list_ready(max_records: int):
    formula = f'{{{STATUS_FIELD}}}="{READY_VALUE}"'
    data = airtable_get({
        "maxRecords": max_records,
        "filterByFormula": formula
    })
    return data.get("records", [])


def update_record(record_id: str, fields: dict):
    url = f"{airtable_base_url()}/{record_id}"
    r = requests.patch(
        url,
        headers=airtable_headers(),
        json={"fields": fields},
        timeout=30
    )
    if r.status_code != 200:
        print("Airtable PATCH error:", r.text, flush=True)
        r.raise_for_status()
    return r.json()


# ==================================================
# Content generation
# ==================================================
def build_prompt(fields: dict) -> str:
    topic = fields.get("Topics") or fields.get("Topic") or "Write an article."
    word_count = fields.get("word count") or fields.get("Word Count") or 700
    tone = fields.get("Tone") or ""
    special = fields.get("Special Content Instructions") or ""

    anchor = fields.get("Anchor Text")
    target_url = fields.get("Target URL")

    link_rule = ""
    if anchor and target_url:
        link_rule = f'Include this anchor once: "{anchor}" linking to {target_url}.'

    return f"""
Write an original article.

Topic:
{topic}

Length:
About {word_count} words.

Tone:
{tone}

Special instructions:
{special}

Link requirement:
{link_rule}

Formatting:
- Title on first line
- Use subheadings
- Avoid repetitive filler
""".strip()


def generate_article(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a professional content writer."},
            {"role": "user", "content": prompt}
        ]
    )
    return (response.choices[0].message.content or "").strip()


# ==================================================
# ZeroGPT
# ==================================================
def zerogpt_detect(text: str):
    if not ZEROGPT_API_KEY:
        return {"enabled": False}

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ZEROGPT_API_KEY
    }

    payload = {"text": text}

    r = requests.post(ZEROGPT_API_URL, headers=headers, json=payload, timeout=60)

    raw_text = r.text

    try:
        data = r.json()
    except Exception:
        return {
            "enabled": True,
            "error": True,
            "raw_json": raw_text
        }

    # Treat success:false as error
    if isinstance(data, dict) and data.get("success") is False:
        return {
            "enabled": True,
            "error": True,
            "raw_json": data
        }

    score = None
    if isinstance(data, dict):
        if "aiPercentage" in data:
            score = float(str(data["aiPercentage"]).replace("%", ""))

    return {
        "enabled": True,
        "error": False,
        "score": score,
        "raw_json": data
    }


# ==================================================
# Processing
# ==================================================
def process_one(record: dict):
    record_id = record["id"]
    fields = record.get("fields", {})

    update_record(record_id, {
        STATUS_FIELD: DRAFTING_VALUE,
        LAST_ERROR_FIELD: ""
    })

    article = generate_article(build_prompt(fields))

    qc = zerogpt_detect(article)

    patch = {
        FINAL_TEXT_FIELD: article,
        AI_RAW_FIELD: json.dumps(qc, ensure_ascii=False)
    }

    if qc.get("error"):
        patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS
    else:
        score = qc.get("score")
        if score is not None:
            patch[AI_SCORE_FIELD] = score
            patch[REVIEW_STATUS_FIELD] = (
                REVIEW_NEEDS if score > ZEROGPT_THRESHOLD else REVIEW_APPROVED
            )
        else:
            patch[REVIEW_STATUS_FIELD] = REVIEW_NEEDS

    patch[STATUS_FIELD] = DELIVERED_VALUE
    update_record(record_id, patch)


# ==================================================
# Main loop
# ==================================================
def main():
    print("========================================", flush=True)
    print("WORKER_VERSION:", WORKER_VERSION, flush=True)
    print("ZeroGPT URL:", ZEROGPT_API_URL, flush=True)
    print("========================================", flush=True)

    while True:
        try:
            ready = list_ready(MAX_RECORDS_PER_CYCLE)
            print("Ready records:", len(ready), flush=True)

            for record in ready:
                try:
                    process_one(record)
                    print("Delivered:", record["id"], flush=True)
                except Exception as e:
                    update_record(record["id"], {
                        STATUS_FIELD: FAILED_VALUE,
                        LAST_ERROR_FIELD: str(e)[:9000]
                    })

        except Exception as e:
            print("Cycle error:", str(e), flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
