import os
import sys
import json
import requests
import psycopg2
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DB_PASS = os.environ.get("COURTLISTENER_DB_PASS")
if not DB_PASS:
    raise SystemExit("Error: Set the COURTLISTENER_DB_PASS environment variable.")

DB_NAME = "courtlistener"
DB_USER = "greg"
DB_HOST = "localhost"
DB_PORT = "5432"

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
TOP_K = 5

# --- Initialize Local Embedder ---
print("Initializing BGE-large embedder on CUDA...", file=sys.stderr)
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda").half()

def retrieve_precedents(query_text: str, top_k: int = TOP_K, jurisdiction_filter=None):
    """Embeds the query and fetches the top K nearest opinions using pgvector HNSW."""
    qvec = embed_model.encode([query_text], normalize_embeddings=True)[0].tolist()
    
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS,
        host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    
    # Optional court filtering (defaults to scope if None)
    filter_sql = ""
    params = [qvec, qvec, top_k]
    if jurisdiction_filter:
        filter_sql = "AND d.court_id = ANY(%s)"
        params = [jurisdiction_filter, qvec, qvec, top_k]
        query_sql = f"""
            SELECT
                o.id AS opinion_id,
                d.case_name,
                d.court_id,
                oc.date_filed,
                substring(o.plain_text from 1200 for 1600) AS substantive_text,
                1 - (e.embedding <=> %s::halfvec) AS similarity
            FROM opinion_embeddings e
            JOIN opinions o ON o.id = e.opinion_id
            JOIN opinion_clusters oc ON o.cluster_id = oc.id
            JOIN dockets d ON oc.docket_id = d.id
            WHERE o.plain_text IS NOT NULL
              {filter_sql}
            ORDER BY e.embedding <=> %s::halfvec
            LIMIT %s;
        """
    else:
        query_sql = """
            SELECT
                o.id AS opinion_id,
                d.case_name,
                d.court_id,
                oc.date_filed,
                substring(o.plain_text from 1200 for 1600) AS substantive_text,
                1 - (e.embedding <=> %s::halfvec) AS similarity
            FROM opinion_embeddings e
            JOIN opinions o ON o.id = e.opinion_id
            JOIN opinion_clusters oc ON o.cluster_id = oc.id
            JOIN dockets d ON oc.docket_id = d.id
            WHERE o.plain_text IS NOT NULL
            ORDER BY e.embedding <=> %s::halfvec
            LIMIT %s;
        """

    cur.execute(query_sql, params if jurisdiction_filter else (qvec, qvec, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def format_prompt(query_text: str, context_rows):
    """Constructs a prompt enforcing strict citation attribution."""
    system_msg = (
        "You are an expert legal research assistant. Analyze the user's inquiry based STRICTLY "
        "on the provided judicial opinions. Every substantive legal proposition must be followed "
        "by an inline citation in the form [Case Name (Court, Year)]. If the provided materials "
        "do not contain sufficient precedent to answer, state that explicitly. Do not invent citations."
    )
    
    context_str = ""
    for idx, row in enumerate(context_rows, 1):
        op_id, case_name, court, date, text, sim = row
        year = str(date).split("-")[0] if date else "Unknown"
        clean_snippet = text.replace("\r", " ").replace("\n", " ").strip()
        context_str += f"\n--- Source [{idx}]: {case_name} ({court}, {year}) [Opinion ID: {op_id}] ---\n"
        context_str += f"{clean_snippet}\n"

    user_msg = (
        f"Context Authorities:\n{context_str}\n"
        f"Legal Question: {query_text}\n\n"
        "Provide a structured legal synthesis addressing the question. Cite the relevant source cases."
    )
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def stream_llm_response(messages):
    """Sends the formatted prompt to llama-server and streams output token-by-token."""
    payload = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": True
    }
    
    response = requests.post(LLAMA_SERVER_URL, json=payload, stream=True)
    response.raise_for_status()
    
    print("\n=== Legal Analysis ===\n")
    for chunk in response.iter_lines():
        if chunk:
            line = chunk.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    data = json.loads(line[6:])
                    delta = data["choices"][0]["delta"].get("content")
                    if delta:
                        print(delta, end="", flush=True)
                except Exception:
                    continue
    print("\n")

def run_query(query: str, courts=None):
    print(f"\nQuerying: {query}", file=sys.stderr)
    retrieved = retrieve_precedents(query, top_k=TOP_K, jurisdiction_filter=courts)
    
    print("\n--- Retrieved Authorities ---", file=sys.stderr)
    for r in retrieved:
        print(f"[{r[5]:.3f}] {r[1]} ({r[2]}, {r[3]}) - ID: {r[0]}", file=sys.stderr)
        
    messages = format_prompt(query, retrieved)
    stream_llm_response(messages)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "What is required to establish landlord constructive notice of lead paint hazards?"
    
    run_query(user_query)