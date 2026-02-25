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
WORKER_VERSION = "v11.0-rag+polishpass+qc-routing+1retry+responsesapi-2026-02-25"

# ==================================================
# REQUIRED ENV VARS (Render)
# ==================================================
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")  # Airtable Personal Access Token
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")  # Articles table (name or tblXXXXXXXX)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==================================================
# REQUIRED ENV VARS (for style RAG)
# ==================================================
AIRTABLE_CLIENTS_TABLE = os.getenv("AIRTABLE_CLIENTS_TABLE")  # Clients table id/name
AIRTABLE_STYLE_TABLE = os.getenv("AIRTABLE_STYLE_TABLE")      # Client Style Library table id/name

# ==================================================
# ZeroGPT
# ==================================================
ZEROGPT_API_KEY = os.getenv("ZEROGPT_API_KEY")
ZEROGPT_API_URL = os.getenv("ZEROGPT_API_URL", "https://api.zerogpt.com/api/detect/detectText")
ZEROGPT_THRESHOLD = float(os.getenv("ZEROGPT_THRESHOLD", "15"))  # routing threshold

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
HUMAN_EDIT_VALUE = os.getenv("HUMAN_EDIT_VALUE", "Human Edit")  # new

FINAL_TEXT_FIELD = os.getenv("FINAL_TEXT_FIELD", "Final Article Text")
LAST_ERROR_FIELD = os.getenv("LAST_ERROR_FIELD", "Last Error")

# optional timestamp field (recommended)
DRAFTING_STARTED_AT_FIELD = os.getenv("DRAFTING_STARTED_AT_FIELD", "Drafting Started At")

# QC fields
AI_SCORE_FIELD = os.getenv("AI_SCORE_FIELD", "AI Score")
AI_RAW_FIELD = os.getenv("AI_RAW_FIELD", "AI Raw Response")
QC_ATTEMPTS_FIELD = os.getenv("QC_ATTEMPTS_FIELD", "QC Attempts")  # Number
QC_NOTES_FIELD = os.getenv("QC_NOTES_FIELD", "QC Notes")            # Long text
HUMAN_EDIT_REQUIRED_FIELD = os.getenv("HUMAN_EDIT_REQUIRED_FIELD", "Human Edit Required")  # Checkbox

# record watchdog (increase to avoid false failures when generation is slow)
RECORD_TIMEOUT_SECONDS = int(os.getenv("RECORD_TIMEOUT_SECONDS", "600"))  # 10 min default

# ==================================================
# HTTP TIMEOUTS (Airtable / ZeroGPT)
# ==================================================
AIRTABLE_HTTP_TIMEOUT = int(os.getenv("AIRTABLE_HTTP_TIMEOUT", "60"))
ZEROGPT_HTTP_TIMEOUT = int(os.getenv("ZEROGPT_HTTP_TIMEOUT", "90"))

# ==================================================
# Style/RAG config
# ==================================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOP_STYLE_SAMPLES = int(os.getenv("TOP_STYLE_SAMPLES", "2"))
MAX_STYLE_CHARS_EACH = int(os.getenv("MAX_STYLE_CHARS_EACH", "2500"))
MAX_STYLE_SAMPLES_FETCH = int(os.getenv("MAX_STYLE_SAMPLES_FETCH", "200"))

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
STYLE_CLIENT_LINK_FIELD = os.getenv("STYLE_CLIENT_LINK_FIELD", "Client")
STYLE_TEXT_FIELD = os.getenv("STYLE_TEXT_FIELD", "Chunk Text")
STYLE_VECTOR_FIELD = os.getenv("STYLE_VECTOR_FIELD", "Embedding Vector")
STYLE_STATUS_FIELD = os.getenv("STYLE_STATUS_FIELD", "Embedding Status")
STYLE_STATUS_EMBEDDED = os.getenv("STYLE_STATUS_EMBEDDED", "Embedded")
STYLE_SAMPLE_TITLE_FIELD = os.getenv("STYLE_SAMPLE_TITLE_FIELD", "Sample Title")

# OpenAI generation model
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5.2")

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
# OPENAI CLIENT (proxy-safe)
# ==================================================
http_client = httpx.Client(
    timeout=httpx.Timeout(
        240.0,   # total (fallback)
        connect=20.0,
        read=240.0,
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
    # best-effort patch; never crash the record
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
# PROMPTING
# ==================================================
def build_prompt(article_fields: dict, client_fields: dict, top_samples_scored: list):
    topic = article_fields.get(TOPIC_FIELD_1) or article_fields.get(TOPIC_FIELD_2) or "Write an original article."
    word_count = article_fields.get("word count") or article_fields.get("Word Count") or 700

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
- Avoid repetitive filler and generic transitions.
- Do not mention AI or detectors.
""".strip()

    return prompt, used_titles_block

# ==================================================
# OPENAI: Responses API helpers
# ==================================================
def _responses_text(resp) -> str:
    # Most SDK versions expose output_text; fallback to traversing output.
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    out = getattr(resp, "output", None) or []
    chunks = []
    for item in out:
        # item may be dict-like or object-like
        content = None
        try:
            content = item.get("content", None)
        except Exception:
            content = getattr(item, "content", None)

        if not content:
            continue
        for c in content:
            try:
                ctype = c.get("type")
                ctext = c.get("text")
            except Exception:
                ctype = getattr(c, "type", None)
                ctext = getattr(c, "text", None)
            if ctype in ("output_text", "text") and ctext:
                chunks.append(ctext)
    return "".join(chunks).strip()

def generate_article(prompt: str) -> str:
    resp = client.responses.create(
        model=GEN_MODEL,
        input=[
            {"role": "system", "content": "You are a professional content writer who follows instructions exactly."},
            {"role": "user", "content": prompt},
        ],
    )
    return _responses_text(resp)

def polish_article(original_text: str) -> str:
    # This is a quality edit pass. Not a “detector” pass.
    prompt = f"""
You are editing an article to improve natural human flow.

Hard rules:
- Do NOT change the topic or meaning.
- Keep the title and subheadings structure.
- Do NOT remove required links/anchors.
- Keep roughly the same length (do not shorten significantly).
- Do not add bullet lists.

Edit goals:
- Reduce generic phrasing and repetitive transitions.
- Vary sentence rhythm and paragraph pacing.
- Make wording feel specific, grounded, and clean.
- Keep it informative and non-promotional.

Article:
{original_text}
""".strip()

    resp = client.responses.create(
        model=GEN_MODEL,
        input=[
            {"role": "system", "content": "You are a careful editor."},
            {"role": "user", "content": prompt},
        ],
    )
    text = _responses_text(resp)
    return text if text else original_text

def targeted_rewrite(article_text: str) -> str:
    # One controlled rewrite to improve clarity/specificity if QC flags it.
    prompt = f"""
Rewrite ONLY:
- the introduction, and
- the two weakest sections (choose them yourself).

Keep:
- the title
- subheadings overall (you may lightly rephrase headings)
- the same meaning and constraints
- any required links/anchors exactly as they appear

Rules:
- No bullet lists.
- Keep similar length.
- Reduce generic phrasing. Use clearer, more concrete wording.

Article:
{article_text}
""".strip()

    resp = client.responses.create(
        model=GEN_MODEL,
        input=[
            {"role": "system", "content": "You rewrite selectively and carefully."},
            {"role": "user", "content": prompt},
        ],
    )
    text = _responses_text(resp)
    return text if text else article_text

# ==================================================
# ZeroGPT DETECTION
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
    fields = record.get("fields", {}) or {}
    start = time.time()

    def check_timeout(step: str):
        if time.time() - start > RECORD_TIMEOUT_SECONDS:
            raise TimeoutError(f"Timeout after {RECORD_TIMEOUT_SECONDS}s at step: {step}")

    # Validate client link exists
    client_link = fields.get(CLIENT_LINK_FIELD)
    if not isinstance(client_link, list) or not client_link:
        raise RuntimeError(f'Missing linked Client in field "{CLIENT_LINK_FIELD}".')
    client_record_id = client_link[0]

    # Set Drafting + timestamp
    drafting_patch = {
        STATUS_FIELD: DRAFTING_VALUE,
        LAST_ERROR_FIELD: "",
        DRAFTING_STARTED_AT_FIELD: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"Setting Drafting: {record_id}")
    update_record(record_id, drafting_patch)
    check_timeout("drafting_patch")

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
            "Check Style Library rows are linked to the same Client record as the Article, "
            "and Embedding Status is exactly 'Embedded'."
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

    # Optional debug field
    if STYLE_SAMPLES_USED_FIELD and used_titles_block:
        safe_patch_articles(record_id, {STYLE_SAMPLES_USED_FIELD: used_titles_block})

    # Draft generation
    log("OpenAI draft generation start...")
    draft = generate_article(prompt)
    log(f"OpenAI draft done (len={len(draft)} chars).")
    check_timeout("openai_draft_done")

    if not draft or len(draft) < 200:
        raise RuntimeError("Generated draft too short.")

    # Polish pass
    log("OpenAI polish pass start...")
    article = polish_article(draft)
    log(f"OpenAI polish pass done (len={len(article)} chars).")
    check_timeout("openai_polish_done")

    # QC (ZeroGPT)
    log("ZeroGPT start...")
    qc = zerogpt_detect(article)
    log(f"ZeroGPT done (qc_error={qc.get('error')}, score={qc.get('score')}).")
    check_timeout("zerogpt_done")

    # Prepare patch base
    patch = {
        FINAL_TEXT_FIELD: article,
        AI_RAW_FIELD: json.dumps(qc, ensure_ascii=False),
        LAST_ERROR_FIELD: "",
    }

    # attempts
    attempts = 0
    try:
        attempts = int(fields.get(QC_ATTEMPTS_FIELD) or 0)
    except Exception:
        attempts = 0
    patch[QC_ATTEMPTS_FIELD] = attempts + 1

    if qc.get("error"):
        patch[STATUS_FIELD] = NEEDS_REVIEW_VALUE
        patch[QC_NOTES_FIELD] = f"QC service error (status={qc.get('http_status')})."
        update_record(record_id, patch)
        return

    score = qc.get("score")
    if score is None:
        patch[STATUS_FIELD] = NEEDS_REVIEW_VALUE
        patch[QC_NOTES_FIELD] = "QC did not return a numeric score."
        update_record(record_id, patch)
        return

    patch[AI_SCORE_FIELD] = float(score)

    # Routing
    if float(score) >= ZEROGPT_THRESHOLD:
        # One controlled retry, then human edit
        if attempts < 1:
            log("Score above threshold; running one targeted rewrite...")
            rewritten = targeted_rewrite(article)

            # Re-QC immediately (same cycle) to avoid waiting another poll
            log("ZeroGPT re-check after targeted rewrite...")
            qc2 = zerogpt_detect(rewritten)
            log(f"ZeroGPT re-check done (qc_error={qc2.get('error')}, score={qc2.get('score')}).")

            patch[FINAL_TEXT_FIELD] = rewritten
            patch[AI_RAW_FIELD] = json.dumps(qc2, ensure_ascii=False)

            if qc2.get("error") or qc2.get("score") is None:
                patch[STATUS_FIELD] = NEEDS_REVIEW_VALUE
                patch[QC_NOTES_FIELD] = "Re-check failed or missing score after rewrite."
                update_record(record_id, patch)
                return

            patch[AI_SCORE_FIELD] = float(qc2["score"])

            if float(qc2["score"]) < ZEROGPT_THRESHOLD:
                patch[STATUS_FIELD] = DELIVERED_VALUE
                patch[QC_NOTES_FIELD] = "Passed after one targeted rewrite."
            else:
                patch[STATUS_FIELD] = HUMAN_EDIT_VALUE
                patch[QC_NOTES_FIELD] = "Still above threshold after one rewrite; routed to human edit."
                if HUMAN_EDIT_REQUIRED_FIELD:
                    patch[HUMAN_EDIT_REQUIRED_FIELD] = True

            update_record(record_id, patch)
            return

        patch[STATUS_FIELD] = HUMAN_EDIT_VALUE
        patch[QC_NOTES_FIELD] = "Above threshold; already attempted rewrite. Routed to human edit."
        if HUMAN_EDIT_REQUIRED_FIELD:
            patch[HUMAN_EDIT_REQUIRED_FIELD] = True
        update_record(record_id, patch)
        return

    patch[STATUS_FIELD] = DELIVERED_VALUE
    patch[QC_NOTES_FIELD] = "Passed QC."
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
    log(f"GEN_MODEL: {GEN_MODEL}")
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
