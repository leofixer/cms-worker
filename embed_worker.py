import os
import requests
import json

AIRTABLE_API_KEY = os.environ["AIRTABLE_API_KEY"]
AIRTABLE_BASE_ID = os.environ["AIRTABLE_BASE_ID"]
STYLE_TABLE_ID = os.environ["AIRTABLE_TABLE_ID_STYLE_LIBRARY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{STYLE_TABLE_ID}"

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}

def fetch_pending_samples():
    params = {
        "filterByFormula": "{Embedding Status}='Pending'"
    }
    response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("records", [])

def create_embedding(text):
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        }
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def update_record(record_id, fields):
    url = f"{AIRTABLE_URL}/{record_id}"
    response = requests.patch(url, headers=HEADERS, json={"fields": fields})
    response.raise_for_status()

def main():
    records = fetch_pending_samples()

    if not records:
        print("No pending samples.")
        return

    for record in records:
        record_id = record["id"]
        fields = record.get("fields", {})
        text = fields.get("Chunk Text")

        if not text:
            update_record(record_id, {
                "Embedding Status": "Failed",
                "Embedding Error": "Missing Chunk Text"
            })
            continue

        try:
            embedding = create_embedding(text)

            update_record(record_id, {
                "Embedding Vector": json.dumps(embedding),
                "Embedding Status": "Embedded",
                "Embedding Error": ""
            })

            print(f"Embedded record {record_id}")

        except Exception as e:
            update_record(record_id, {
                "Embedding Status": "Failed",
                "Embedding Error": str(e)
            })

            print(f"Failed record {record_id}: {e}")

if __name__ == "__main__":
    main()
