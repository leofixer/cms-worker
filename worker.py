import os
import time
import json
import math
import requests
from urllib.parse import quote

import httpx
from openai import OpenAI

# ==================================================
# VERSION
# ==================================================
WORKER_VERSION = "v10.2-gen+ragsamples(pyfilter)+timeouts+qc-under15-needsreview-2026-02-24"

# ==================================================
# REQUIRED ENV VARS (Render)
# ==================================================
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")  # Airtable Personal Access Token
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")  # Articles table (tblXXXXXXXX recommended)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==================================================
# REQUIRED ENV VARS (for style RAG)
# ==================================================
AIRTABLE_CLIENTS_TABLE = os.getenv("AIRTABLE_CLIENTS_TABLE")  # Clients table id/name
AIRTABLE_STYLE_TABLE = os.getenv("AIRTABLE_STYLE_TABLE")      # Client Style Library table id/name

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

# record watchdog (increase to avoid false failures when generation is slow)
RECORD_TIMEOUT_SECONDS = int(os.getenv("RECORD_TIMEOUT_SECONDS", "600"))  # 10 min default

# ==================================================
# HTTP TIMEOUTS (Airtable / ZeroGPT)
# ==================================================
AIRTABLE_HTTP_TIMEOUT = int(os.getenv("AIRTABLE_HTTP_TIMEOUT", "60"))
ZEROGPT_HTTP_TIMEOUT = int(os.getenv("ZEROGPT_HTTP_TIMEOUT", "90"))

# ==================================================
# ZeroGPT (per your docs)
# ==================================================
ZEROGPT_API_KEY = os.getenv("ZEROGPT_API_KEY")
ZEROGPT_API_URL = os.getenv("ZEROGPT_API_URL", "https://api.zerogpt.com/api/detect/detectText")
ZEROGPT_THRESHOLD = float(os.getenv("ZEROGPT_THRESHOLD", "15"))  # must be < 15 to deliver

AI_SCORE_FIELD = os.getenv("AI_SCORE_FIELD", "AI Score")
AI_RAW_FIELD = os.getenv("AI_RAW_FIELD", "AI Raw Response")

# ==================================================
# Style/RAG config (optional)
# ==================================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Defaults lowered to prevent huge prompts / slow generations
TOP_STYLE_SAMPLES = int(os.getenv("TOP_STYLE_SAMPLES", "2"))          # recommend 2–3
MAX_STYLE_CHARS_EACH = int(os.getenv("MAX_STYLE_CHARS_EACH", "2500")) # trim references
MAX_STYLE_SAMPLES_FETCH = int(os.getenv("MAX_STYLE_SAMPLES_FETCH", "200"))  # fetch embedded, then filter

# Article fields (Airtable)
CLIENT_LINK_FIELD = os.getenv("CLIENT_LINK_FIELD", "Client")  # link-to-record field on Articles
BRIEF_FIELD = os.getenv("BRIEF_FIELD", "Brief")
TOPIC_FIELD_1 = os.getenv("TOPIC_FIELD_1", "Topics")
TOPIC_FIELD_2 = os.getenv("TOPIC_FIELD_2", "Topic")

# Optional debug field on Articles (create it if you want)
STYLE_SAMPLES_USED_FIELD = os.getenv("STYLE_SAMPLES_USED_FIELD", "Style Samples Used")

# Clients table fields
CLIENT_STYLE_RULES_FIELD = os.getenv("CLIENT_STYLE_RULES_FIELD", "Style Rules")
CLIENT_FORMAT_RULES_FIELD = os.getenv("CLIENT_FORMAT_RULES_FIELD", "Format Rules")
CLIENT_BANNED_PHRASES_FIELD = os.getenv("CLIENT_BANNED_PHRASES_FIELD", "Banned Phrases")

# Style library fields
STYLE_CLIENT_LINK_FIELD = os.getenv("STYLE_CLIENT_LINK_FIELD", "Client")  # link-to-record to Clients
STYLE_TEXT_FIELD = os.getenv("STYLE_TEXT_FIELD", "Chunk Text")
STYLE_VECTOR_FIELD = os.getenv("STYLE_VECTOR_FIELD", "Embedding Vector")
STYLE_STATUS_FIELD = os.getenv("STYLE_STATUS_FIELD", "Embedding Status")
STYLE_STATUS_EMBEDDED = os.getenv("STYLE_STATUS_EMBEDDED", "Embedded")
STYLE_SAMPLE_TITLE_FIELD = os.getenv("STYLE_SAMPLE_TITLE_FIELD", "Sample Title")  # primary or title

# ==================================================
# ENV CHECK
# ==================================================
def require_env():
    missing = []
    required = {
        "AIRTABLE_TOKEN": AIRTABLE_TOKEN,
        "AIRTABLE_BASE_ID": AIRTABLE_BASE_ID,
        "AIRTABLE_TABLE": AIRTABLE_TABLE,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "ZEROGPT_API_KEY": ZEROGPT_API_KEY,
        "AIRTABLE_CLIENTS_TABLE": AIRTABLE_CLIENTS_TABLE,
        "AIRTABLE_STYLE_TABLE": AIRTABLE_STYLE_TABLE,
    }
    for k, v in required.items():
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

require_env()

# ==================================================
# LOGGING HELPERS
# ==================================================
def log(msg: str):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)

# ==================================================
# OPENAI CLIENT (proxy-safe) — increased read timeout
# ==================================================
http_client = httpx.Client(
    timeout=httpx.Timeout(
        180.0,   # total (fallback)
        connect=20.0,
        read=180.0,
        write=60.0,
        pool=60.0,
    ),
    follow_redirects=True,
    trust_env=False,
)
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

# ==================================================
# AIRTABLE HELPERS
# ==================================================
def airtable_headers():
    return {"Authorization": f"Bearer {AIRTABLE_TOKEN}", "Content-Type": "application/json"}

def airtable_table_url(table_name_or_id: str):
    table = quote(table_name_or_id, safe="")
    return f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table}"

def airtable_get_table(table_name_or_id: str, params: dict):
    url = airtable_table_url(table_name_or_id)
    r = requests.get(url, headers=airtable_headers(), params=params, timeout=AIRTABLE_HTTP_TIMEOUT)
    if r.status_code != 200:
        log(f"Airtable GET URL: {r.url}")
        log(f"Airtable GET status: {r.status_code}")
        log(f"Airtable GET response: {r.text}")
        r.raise_for_status()
    return r.json()

def airtable_get(params: dict):
    return airtable_get_table(AIRTABLE_TABLE, params)

def update_record(record_id: str, fields: dict):
    url = f"{airtable_table_url(AIRTABLE_TABLE)}/{record_id}"
    r = requests.patch(url, headers=airtable_headers(), json={"fields": fields}, timeout=AIRTABLE_HTTP_TIMEOUT)
    if r.status_code != 200:
        log(f"Airtable PATCH URL: {url}")
        log(f"Airtable PATCH status: {r.status_code}")
        log(f"Airtable PATCH response: {r.text}")
        r.raise_for_status()
    return r.json()

def safe_patch_articles(record_id: str, fields: dict):
    try:
        update_record(record_id, fields)
    except Exception as e:
        log(f"Non-fatal patch error: {e}")

def safe_mark_failed(record_id: str, msg: str):
    msg = (msg or "Unknown error")[:9000]
    try:
        update_record(record_id, {STATUS_FIELD: FAILED_VALUE, LAST_ERROR_FIELD: msg})
    except Exception as e:
        log(f"Failed to mark failed: {e}")
        try:
            update_record(record_id, {STATUS_FIELD: FAILED_VALUE})
        except Exception as e2:
            log(f"Failed even minimal mark failed: {e2}")

def list_ready(max_records: int):
    formula = f'{{{STATUS_FIELD}}}="{READY_VALUE}"'
    data = airtable_get({"maxRecords": max_records, "filterByFormula": formula})
    return data.get("records", [])

# ==================================================
# RAG STYLE HELPERS
# ==================================================
def cosine_similarity(vec1, vec2) -> float:
    n = min(len(vec1), len(vec2))
    if n == 0:
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for i in range(n):
        a = float(vec1[i])
        b = float(vec2[i])
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0 or norm2 <= 0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))

def embed_text(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def get_client_record(client_record_id: str) -> dict:
    data = airtable_get_table(
        AIRTABLE_CLIENTS_TABLE,
        {"maxRecords": 1, "filterByFormula": f'RECORD_ID()="{client_record_id}"'},
    )
    recs = data.get("records", [])
    if not recs:
        raise RuntimeError(f"Client record not found: {client_record_id}")
    return recs[0]

def _normalize_banned_phrases(v):
    if not v:
        return ""
    if isinstance(v, list):
        return "\n".join([str(x) for x in v if str(x).strip()])
    return str(v).strip()

def fetch_style_samples_for_client(client_record_id: str, max_records: int = 200):
    """
    Robust approach:
    - Airtable filter only on Embedding Status = Embedded
    - Filter by linked Client record_id in Python
    """
    formula = f'{{{STYLE_STATUS_FIELD}}}="{STYLE_STATUS_EMBEDDED}"'
    data = airtable_get_table(
        AIRTABLE_STYLE_TABLE,
        {
            "maxRecords": max_records,
            "filterByFormula": formula,
            "fields[]": [STYLE_TEXT_FIELD, STYLE_VECTOR_FIELD, STYLE_SAMPLE_TITLE_FIELD, STYLE_CLIENT_LINK_FIELD],
        },
    )
    recs = data.get("records", [])
    matched = []
    for r in recs:
        f = r.get("fields", {}) or {}
        linked = f.get(STYLE_CLIENT_LINK_FIELD)
        if isinstance(linked, list) and client_record_id in linked:
            matched.append(r)
    return matched

def select_top_style_samples(style_records: list, query_embedding: list, top_k: int):
    scored = []
    for rec in style_records:
        f = rec.get("fields", {}) or {}
        vec_raw = f.get(STYLE_VECTOR_FIELD)
        text = f.get(STYLE_TEXT_FIELD) or ""
        if not vec_raw or not text:
            continue
        try:
            vec = json.loads(vec_raw) if isinstance(vec_raw, str) else vec_raw
            if not isinstance(vec, list) or not vec:
                continue
            score = cosine_similarity(query_embedding, vec)
            scored.append((score, rec))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# ==================================================
# CONTENT GENERATION (with style RAG)
# ==================================================
def build_prompt(article_fields: dict, client_fields: dict, top_samples_scored: list):
    topic = article_fields.get(TOPIC_FIELD_1) or article_fields.get(TOPIC_FIELD_2) or "Write an original article."
    word_count = article_fields.get("word count") or article_fields.get("Word Count") or 700

    # We no longer use Tone / Special Content Instructions (safe to delete those fields)
    style_rules = (client_fields.get(CLIENT_STYLE_RULES_FIELD) or "").strip()
    format_rules = (client_fields.get(CLIENT_FORMAT_RULES_FIELD) or "").strip()
    banned_phrases = _normalize_banned_phrases(client_fields.get(CLIENT_BANNED_PHRASES_FIELD))

    anchor = article_fields.get("Anchor Text")
    target_url = article_fields.get("Target URL")
    link_rule = ""
    if anchor and target_url:
        link_rule = f'Include this exact anchor text once: "{anchor}" linking to {target_url}. Make it natural.'

    refs_lines = []
    used_titles = []
    for idx, (score, rec) in enumerate(top_samples_scored, start=1):
        sf = rec.get("fields", {}) or {}
        title = sf.get(STYLE_SAMPLE_TITLE_FIELD) or rec.get("id")
        used_titles.append(f"{idx}. {title} (score={score:.4f})")

        sample_text = (sf.get(STYLE_TEXT_FIELD) or "").strip()
        if len(sample_text) > MAX_STYLE_CHARS_EACH:
            sample_text = sample_text[:MAX_STYLE_CHARS_EACH].rstrip() + "\n[...trimmed]"
        refs_lines.append(f"Sample {idx}:\n{sample_text}")

    style_refs_block = "\n\n".join(refs_lines).strip()
    used_titles_block = "\n".join(used_titles).strip()

    prompt = f"""
Write an original article.

Topic:
{topic}

Target length:
About {word_count} words.

Client style rules:
{style_rules}

Client format rules:
{format_rules}

Banned phrases (do not use any of these exact phrases):
{banned_phrases}

Link requirement:
{link_rule}

STYLE REFERENCES (for tone, cadence, and structure only — do not copy sentences verbatim):
{style_refs_block}

Formatting rules (global):
- Put a clear title on the first line.
- Use subheadings.
- Do not use bullet lists or hyphen bullets.
- Avoid repetitive filler.
- Do not mention AI or detectors.
""".strip()

    return prompt, used_titles_block

def generate_article(prompt: str) -> str:
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
    headers = {"Content-Type": "application/json", "ApiKey": ZEROGPT_API_KEY}
    payload = {"input_text": text}

    r = requests.post(ZEROGPT_API_URL, headers=headers, json=payload, timeout=ZEROGPT_HTTP_TIMEOUT)
    raw = r.text

    try:
        data = r.json()
    except Exception:
        return {"enabled": True, "error": True, "http_status": r.status_code, "raw": raw[:20000]}

    if isinstance(data, dict) and data.get("success") is False:
        return {"enabled": True, "error": True, "http_status": data.get("code", r.status_code), "raw_json": data}

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

    # Validate client link exists
    client_link = fields.get(CLIENT_LINK_FIELD)
    if not isinstance(client_link, list) or not client_link:
        raise RuntimeError(f'Missing linked Client in field "{CLIENT_LINK_FIELD}".')
    client_record_id = client_link[0]

    # set Drafting + timestamp
    drafting_patch = {
        STATUS_FIELD: DRAFTING_VALUE,
        LAST_ERROR_FIELD: "",
        DRAFTING_STARTED_AT_FIELD: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"Setting Drafting: {record_id}")
    update_record(record_id, drafting_patch)
    check_timeout("drafting_patch")

    # Debug: show config
    log(f"Client record id from Article: {client_record_id}")
    log(f"STYLE_CLIENT_LINK_FIELD: {STYLE_CLIENT_LINK_FIELD}")
    log(f"STYLE_STATUS_FIELD: {STYLE_STATUS_FIELD} = {STYLE_STATUS_EMBEDDED}")
    log(f"TOP_STYLE_SAMPLES: {TOP_STYLE_SAMPLES}, MAX_STYLE_CHARS_EACH: {MAX_STYLE_CHARS_EACH}")

    # Fetch client rules
    log("Fetching client rules...")
    client_rec = get_client_record(client_record_id)
    client_fields = client_rec.get("fields", {}) or {}
    check_timeout("client_fetch")

    # Fetch style samples
    log("Fetching embedded style samples...")
    style_recs = fetch_style_samples_for_client(client_record_id, max_records=MAX_STYLE_SAMPLES_FETCH)
    log(f"Embedded style samples matched: {len(style_recs)}")
    if not style_recs:
        raise RuntimeError(
            "No embedded style samples found for this client. "
            "Check that Style Library rows are linked to the same Client record as the Article, "
            "and that Embedding Status is exactly 'Embedded'."
        )
    check_timeout("style_fetch")

    # Embed query for retrieval (brief + topic)
    topic = fields.get(TOPIC_FIELD_1) or fields.get(TOPIC_FIELD_2) or ""
    brief = fields.get(BRIEF_FIELD) or fields.get("Instructions") or ""
    query_text = (str(brief).strip() + "\n\n" + str(topic).strip()).strip() or "article brief"
    log(f"Embedding query (len={len(query_text)} chars)...")
    query_emb = embed_text(query_text[:8000])
    check_timeout("embed_query")

    # Select top samples
    top_scored = select_top_style_samples(style_recs, query_emb, top_k=TOP_STYLE_SAMPLES)
    if not top_scored:
        raise RuntimeError("Could not score/select any style samples (missing vectors or invalid JSON).")
    check_timeout("select_top_samples")

    # Build prompt
    prompt, used_titles_block = build_prompt(fields, client_fields, top_scored)

    # Optional: write debug field if present
    safe_patch_articles(record_id, {STYLE_SAMPLES_USED_FIELD: used_titles_block})

    # Generate
    log("OpenAI generation start...")
    article = generate_article(prompt)
    log(f"OpenAI generation done (len={len(article)} chars).")
    check_timeout("openai_done")

    if not article or len(article) < 200:
        raise RuntimeError("Generated article too short.")

    # ZeroGPT
    log("ZeroGPT start...")
    qc = zerogpt_detect(article)
    log(f"ZeroGPT done (qc_error={qc.get('error')}, score={qc.get('score')}).")
    check_timeout("zerogpt_done")

    patch = {
        FINAL_TEXT_FIELD: article,
        AI_RAW_FIELD: json.dumps(qc, ensure_ascii=False),
    }

    if qc.get("error"):
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

    if float(score) >= ZEROGPT_THRESHOLD:
        patch[STATUS_FIELD] = NEEDS_REVIEW_VALUE
    else:
        patch[STATUS_FIELD] = DELIVERED_VALUE

    log(f"Writing final status: {record_id} -> {patch[STATUS_FIELD]}")
    update_record(record_id, patch)

# ==================================================
# MAIN LOOP
# ==================================================
def main():
    log("========================================")
    log(f"WORKER_VERSION: {WORKER_VERSION}")
    log(f"BASE: {AIRTABLE_BASE_ID}")
    log(f"ARTICLES TABLE: {AIRTABLE_TABLE}")
    log(f"CLIENTS TABLE: {AIRTABLE_CLIENTS_TABLE}")
    log(f"STYLE TABLE: {AIRTABLE_STYLE_TABLE}")
    log(f"STATUS_FIELD: {STATUS_FIELD} READY_VALUE: {READY_VALUE}")
    log(f"POLL_SECONDS: {POLL_SECONDS} MAX_RECORDS_PER_CYCLE: {MAX_RECORDS_PER_CYCLE}")
    log(f"EMBED_MODEL: {EMBED_MODEL} TOP_STYLE_SAMPLES: {TOP_STYLE_SAMPLES} MAX_STYLE_CHARS_EACH: {MAX_STYLE_CHARS_EACH}")
    log(f"AIRTABLE_HTTP_TIMEOUT: {AIRTABLE_HTTP_TIMEOUT} ZEROGPT_HTTP_TIMEOUT: {ZEROGPT_HTTP_TIMEOUT}")
    log(f"ZeroGPT URL: {ZEROGPT_API_URL}")
    log(f"ZeroGPT threshold (<): {ZEROGPT_THRESHOLD}")
    log("========================================")

    while True:
        try:
            log("Polling Airtable...")
            ready = list_ready(MAX_RECORDS_PER_CYCLE)
            log(f"Ready records found: {len(ready)}")

            for rec in ready:
                rid = rec["id"]
                try:
                    log(f"Processing: {rid}")
                    process_one(rec)
                except Exception as e:
                    log(f"FAILED: {rid} {e}")
                    safe_mark_failed(rid, str(e))

        except Exception as cycle_error:
            log(f"Cycle error: {cycle_error}")

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
