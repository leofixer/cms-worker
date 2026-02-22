import os
import time
import requests
from openai import OpenAI

# -------------------------
# ENV VARS (set on Render)
# -------------------------
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")  # exact table name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "120"))
MAX_RECORDS_PER_CYCLE = int(os.getenv("MAX_RECORDS_PER_CYCLE", "3"))

# Basic safety checks
missing = [k for k, v in {
    "AIRTABLE_TOKEN": AIRTABLE_TOKEN,
    "AIRTABLE_BASE_ID": AIRTABLE_BASE_ID,
    "AIRTABLE_TABLE": AIRTABLE_TABLE,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}.items() if not v]

if missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

import httpx

http_client = httpx.Client(
    timeout=60.0,
    follow_redirects=True,
    trust_env=False  # IMPORTANT: ignores proxy env vars that cause the crash
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=http_client
)

def airtable_headers():
    return {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json",
    }

def airtable_url(path=""):
    # Table name must be URL-encoded if it has spaces; requests handles this if we pass it in the URL as-is in most cases,
    # but safest is to keep your table name simple (e.g., Orders).
    return f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}{path}"

def list_ready(max_records: int):
    params = {
        "maxRecords": max_records,
        "filterByFormula": '{Status}="Ready"'
    }
    r = requests.get(airtable_url(), headers=airtable_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("records", [])

def update_record(record_id: str, fields: dict):
    r = requests.patch(
        airtable_url(f"/{record_id}"),
        headers=airtable_headers(),
        json={"fields": fields},
        timeout=30
    )
    r.raise_for_status()
    return r.json()

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

Formatting:
- Include a clear title on the first line.
- Use subheadings.
- Avoid repetitive sentence patterns and generic filler.
- Do not mention AI or detectors.
"""
    return prompt.strip()

def generate_article(prompt: str) -> str:
    resp = client.responses.create(
        model="gpt-5",
        input=prompt,
        temperature=0.8,
    )
    return (resp.output_text or "").strip()

def process_one(record: dict):
    record_id = record["id"]
    fields = record.get("fields", {})

    # Mark Drafting
    update_record(record_id, {"Status": "Drafting", "Last Error": ""})

    prompt = build_prompt(fields)
    article = generate_article(prompt)

    if not article or len(article) < 200:
        raise RuntimeError("Generated article text looks empty/too short.")

    # Save back
    update_record(record_id, {
        "Final Article Text": article,
        "Status": "Delivered"
    })

def main():
    print("Worker started. Polling Airtable...")
    while True:
        try:
            ready = list_ready(MAX_RECORDS_PER_CYCLE)
            if ready:
                print(f"Found {len(ready)} Ready records")
            for rec in ready:
                try:
                    process_one(rec)
                    print(f"Delivered: {rec['id']}")
                except Exception as e:
                    rid = rec["id"]
                    msg = str(e)
                    print(f"Failed record {rid}: {msg}")
                    try:
                        update_record(rid, {"Status": "Failed", "Last Error": msg[:9000]})
                    except Exception as inner:
                        print(f"Also failed to update Airtable error field: {inner}")

        except Exception as e:
            print(f"Cycle error: {e}")

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":

    main()
