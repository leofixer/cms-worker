import os
import time
import requests
from urllib.parse import quote

import httpx
from openai import OpenAI

# ==================================================
# VERSION STAMP (VERY IMPORTANT FOR DEBUGGING)
# ==================================================
WORKER_VERSION = "v2-chatcompletions-2026-02-23"

# ==================================================
# REQUIRED ENV VARS (Render)
# ==================================================
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")  # use tblXXXXXXXX (recommended)
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
# ENV VALIDATION
# ==================================================
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
    timeout=60.0,
    follow_redirects=True,
    trust_env=False  # prevents proxy-related crash
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=http_client
)


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
    params = {
        "maxRecords": max_records,
        "filterByFormula": filter_formula
    }

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
    topic = fields.get("Topics") or fields.get("Topic") or "Write an original article on the provided topic."
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
        ],
        temperature=0.8
    )

    return (response.choices[0].message.content or "").strip()


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
        raise RuntimeError("Generated article is empty or too short.")

    update_record(record_id, {
        FINAL_TEXT_FIELD: article,
        STATUS_FIELD: DELIVERED_VALUE
    })


# ==================================================
# MAIN LOOP
# ==================================================
def main():
    print("========================================", flush=True)
    print("WORKER_VERSION:", WORKER_VERSION, flush=True)
    print("BASE:", AIRTABLE_BASE_ID, flush=True)
    print("TABLE:", AIRTABLE_TABLE, flush=True)
    print("STATUS_FIELD:", STATUS_FIELD, "READY_VALUE:", READY_VALUE, flush=True)
    print("FINAL_TEXT_FIELD:", FINAL_TEXT_FIELD, flush=True)
    print("LAST_ERROR_FIELD:", LAST_ERROR_FIELD, flush=True)
    print("POLL_SECONDS:", POLL_SECONDS, flush=True)
    print("MAX_RECORDS_PER_CYCLE:", MAX_RECORDS_PER_CYCLE, flush=True)
    print("httpx version:", httpx.__version__, flush=True)
    print("OpenAI client initialized", flush=True)
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

                    update_record(record_id, {
                        STATUS_FIELD: FAILED_VALUE,
                        LAST_ERROR_FIELD: error_msg[:9000]
                    })

        except Exception as cycle_error:
            print("Cycle error:", str(cycle_error), flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
