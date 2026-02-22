import os
import time
import requests
from urllib.parse import quote

import httpx
from openai import OpenAI


# -------------------------
# ENV VARS (set on Render)
# -------------------------
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")  # exact Airtable table name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional tuning
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "120"))
MAX_RECORDS_PER_CYCLE = int(os.getenv("MAX_RECORDS_PER_CYCLE", "3"))

# Airtable field names / statuses (configurable)
STATUS_FIELD = os.getenv("STATUS_FIELD", "Status")           # column name in Airtable
READY_VALUE = os.getenv("READY_VALUE", "Ready")              # single-select value
DRAFTING_VALUE = os.getenv("DRAFTING_VALUE", "Drafting")
DELIVERED_VALUE = os.getenv("DELIVERED_VALUE", "Delivered")
FAILED_VALUE = os.getenv("FAILED_VALUE", "Failed")

# Output/error fields in Airtable
FINAL_TEXT_FIELD = os.getenv("FINAL_TEXT_FIELD", "Final Article Text")
LAST_ERROR_FIELD = os.getenv("LAST_ERROR_FIELD", "Last Error")


def require_env():
    missing = [k for k, v in {
        "AIRTABLE_TOKEN": AIRTABLE_TOKEN,
        "AIRTABLE_BASE_ID": AIRTABLE_BASE_ID,
        "AIRTABLE_TABLE": AIRTABLE_TABLE,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


require_env()

# ---- OpenAI client fix: ignore proxy env vars that can break httpx/OpenAI on hosts ----
http_client = httpx.Client(
    timeout=60.0,
    follow_redirects=True,
    trust_env=False  # IMPORTANT: prevents proxy env vars from causing the 'proxies' crash
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=http_client
)


# -------------------------
# Airtable helpers
# -------------------------
def airtable_headers():
    return {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json",
    }


def airtable_base_url():
    # URL-encode table name to safely handle spaces and special chars
    table = quote(AIRTABLE_TABLE, safe="")
    return f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table}"


def list_ready(max_records: int):
    # Airtable filter formula. Note: field name and value are case-sensitive.
    filter_formula = f'{{{STATUS_FIELD}}}="{READY_VALUE}"'
    params = {
        "maxRecords": max_records,
        "filterByFormula": filter_formula
    }
    r = requests.get(airtable_base_url(), headers=airtable_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("records", [])


def update_record(record_id: str, fields: dict):
    url = f"{airtable_base_url()}/{record_id}"
    r = requests.patch(url, headers=airtable_headers(), json={"fields": fields}, timeout=30)
    r.raise_for_status()
    return r.json()


# -------------------------
# Content generation logic
# -------------------------
def build_prompt(fields: dict) -> str:
    # Try multiple possible column names to avoid breaking if you rename
    topic = fields.get("Topics") or fields.get("Topic") or "Write an original article on the provided topic."
    word_count = fields.get("word count") or fields.get("Word Count") or fields.get("Word count") or 700
    tone = fields.get("Tone") or ""
    special = fields.get("Special Content Instructions") or ""

    anchor = fields.get("Anchor Text")
    target_url = fields.get("Target URL")

    link_rule = ""
    if anchor and target_url:
        link_rule = f'Include this exact anchor text once: "{anchor}" linking to {target_url}. Make it natural.'

    prompt = f"""
Write an original article that follows the instructions below.

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

Formatting rules:
- Include a clear title on the first line.
- Use subheadings.
- Avoid repetitive sentence patterns and generic filler.
- Do not mention AI, detectors, or that this was generated.
"""
    return prompt.strip()


def generate_article(prompt: str) -> str:
    resp = client.responses.create(
        model="gpt-5",
        input=prompt,
        temperature=0.8,
    )
    text = (resp.output_text or "").strip()
    return text


def process_one(record: dict):
    record_id = record["id"]
    fields = record.get("fields", {})

    # Mark as Drafting
    update_record(record_id, {
        STATUS_FIELD: DRAFTING_VALUE,
        LAST_ERROR_FIELD: ""
    })

    prompt = build_prompt(fields)
    article = generate_article(prompt)

    if not article or len(article) < 200:
        raise RuntimeError("Generated article text looks empty/too short.")

    # Save and mark Delivered
    update_record(record_id, {
        FINAL_TEXT_FIELD: article,
        STATUS_FIELD: DELIVERED_VALUE
    })


# -------------------------
# Main worker loop
# -------------------------
def main():
    print("=== CMS WORKER STARTED ===", flush=True)
    print("Python runtime OK", flush=True)
    print("Airtable base:", AIRTABLE_BASE_ID, flush=True)
    print("Airtable table:", AIRTABLE_TABLE, flush=True)
    print("Status field:", STATUS_FIELD, flush=True)
    print("Ready value:", READY_VALUE, flush=True)
    print("Poll seconds:", POLL_SECONDS, flush=True)
    print("Max records/cycle:", MAX_RECORDS_PER_CYCLE, flush=True)
    print("httpx version:", httpx.__version__, flush=True)
    print("openai client initialized", flush=True)

    while True:
        print("Polling Airtable for Ready records...", flush=True)

        try:
            ready = list_ready(MAX_RECORDS_PER_CYCLE)
            print("Ready records found:", len(ready), flush=True)

            for rec in ready:
                rid = rec["id"]
                try:
                    print("Processing:", rid, flush=True)
                    process_one(rec)
                    print("Delivered:", rid, flush=True)
                except Exception as e:
                    msg = str(e)
                    print("FAILED record:", rid, msg, flush=True)
                    try:
                        update_record(rid, {
                            STATUS_FIELD: FAILED_VALUE,
                            LAST_ERROR_FIELD: msg[:9000]
                        })
                    except Exception as inner:
                        print("Also failed to update Airtable error:", str(inner), flush=True)

        except Exception as e:
            print("Cycle error:", str(e), flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
