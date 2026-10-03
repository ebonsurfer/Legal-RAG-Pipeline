import os
import sys
import json
import requests
import psycopg2
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration & Environment ---
DB_PASS = os.environ.get("COURTLISTENER_DB_PASS")
if not DB_PASS:
    raise SystemExit("Error: Set COURTLISTENER_DB_PASS environment variable.")

DB_NAME = "courtlistener"
DB_USER = "greg"
DB_HOST = "localhost"
DB_PORT = "5432"

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- Load Local BGE-large Model (GPU FP16) ---
print("Loading BGE-large onto CUDA...", file=sys.stderr)
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda").half()

CURRENT_COURTS = None
TOP_K = 5

def run_query(query_text: str, top_k: int = TOP_K, courts: list = None):
    # 1. Embed query vector locally on GPU
    qvec = embed_model.encode([query_text], normalize_embeddings=True)[0].tolist()
    
    conn = psycopg2.connect(
        dbname=DB_NAME, 
        user=DB_USER, 
        password=DB_PASS, 
        host=DB_HOST, 
        port=DB_PORT
    )
    cur = conn.cursor()
    
    # 2. Stage 1: Fast Vector Search (Fetches only IDs & Metadata, avoiding text bloat)
    candidate_limit = top_k * 4 if courts else top_k * 2
    
    cur.execute("SET hnsw.iterative_scan = relaxed_order;")
    cur.execute("SET hnsw.ef_search = 100;")
    
    if courts:
        stage1_sql = """
            SELECT 
                e.opinion_id,
                oc.id AS cluster_id,
                d.case_name,
                d.court_id,
                oc.date_filed,
                (1 - (e.embedding <=> %s::halfvec)) AS similarity,
                (1 - (e.embedding <=> %s::halfvec)) * 
                    (CASE WHEN d.court_id IN ('ca2', 'ny', 'nj') THEN 1.05 ELSE 1.0 END) AS score
            FROM opinion_embeddings e
            JOIN opinions o ON o.id = e.opinion_id
            JOIN opinion_clusters oc ON o.cluster_id = oc.id
            JOIN dockets d ON oc.docket_id = d.id
            WHERE d.court_id = ANY(%s)
            ORDER BY e.embedding <=> %s::halfvec
            LIMIT %s;
        """
        cur.execute(stage1_sql, (qvec, qvec, courts, qvec, candidate_limit))
    else:
        stage1_sql = """
            SELECT 
                e.opinion_id,
                oc.id AS cluster_id,
                d.case_name,
                d.court_id,
                oc.date_filed,
                (1 - (e.embedding <=> %s::halfvec)) AS similarity,
                (1 - (e.embedding <=> %s::halfvec)) AS score
            FROM opinion_embeddings e
            JOIN opinions o ON o.id = e.opinion_id
            JOIN opinion_clusters oc ON o.cluster_id = oc.id
            JOIN dockets d ON oc.docket_id = d.id
            ORDER BY e.embedding <=> %s::halfvec
            LIMIT %s;
        """
        cur.execute(stage1_sql, (qvec, qvec, candidate_limit))
        
    candidates = cur.fetchall()
    
    if not candidates:
        cur.close()
        conn.close()
        print("\nNo matching authorities found for the given criteria.\n")
        return

    # In-memory deduplication by cluster_id (instant in Python)
    seen_clusters = set()
    deduped = []
    # Sort candidates by boosted score first
    candidates.sort(key=lambda x: x[6], reverse=True)
    
    for row in candidates:
        op_id, cluster_id, case_name, court_id, date_filed, sim, score = row
        if cluster_id not in seen_clusters:
            seen_clusters.add(cluster_id)
            deduped.append(row)
            if len(deduped) == top_k:
                break
                
    selected_ids = [r[0] for r in deduped]

    # 3. Stage 2: Hydrate substantive text for only the winning IDs (fast indexed PK lookup)
    cur.execute("""
        SELECT id, substring(plain_text from 1000 for 2500)
        FROM opinions
        WHERE id = ANY(%s);
    """, (selected_ids,))
    
    text_map = {row[0]: (row[1] or "") for row in cur.fetchall()}
    cur.close()
    conn.close()

    # 4. Display Retrieved Cases
    print("\n--- Retrieved Authorities ---", file=sys.stderr)
    context_str = ""
    for idx, row in enumerate(deduped, 1):
        op_id, cluster_id, case_name, court, date, sim, score = row
        text = text_map.get(op_id, "")
        year = str(date).split("-")[0] if date else "Unknown"
        clean_text = text.replace("\r", " ").replace("\n", " ").strip()
        print(f"[{sim:.3f}] {case_name} ({court}, {year}) - ID: {op_id}", file=sys.stderr)
        context_str += f"\n--- Source [{idx}]: {case_name} ({court}, {year}) [Opinion ID: {op_id}] ---\n{clean_text}\n"

    # 5. Build Grounded Prompt
    system_msg = (
        "You are an expert legal research assistant. Analyze the inquiry based STRICTLY "
        "on the provided opinions. Follow substantive legal propositions with inline "
        "citations in the form [Case Name (Court, Year)]. If the context is insufficient, state that clearly."
    )
    user_msg = f"Context Authorities:\n{context_str}\nLegal Question: {query_text}\n\nSynthesize the applicable law:"

    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": True
    }

    # 6. Stream from llama-server
    try:
        resp = requests.post(LLAMA_SERVER_URL, json=payload, stream=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"\nError connecting to llama-server at {LLAMA_SERVER_URL}: {e}\n")
        return

    print("\n=== Legal Analysis ===\n")
    for chunk in resp.iter_lines():
        if chunk:
            line = chunk.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                    if delta:
                        print(delta, end="", flush=True)
                except Exception:
                    continue
    print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    print("\nLegal RAG CLI Ready.")
    print("Commands:")
    print("  /courts <id1,id2>  (e.g., /courts ca2,nysd)")
    print("  /all               (clears court filter)")
    print("  /topk <int>        (adjust context count, e.g., /topk 8)")
    print("  exit / quit        (leave REPL)\n")
    
    while True:
        try:
            scope_label = "ALL" if not CURRENT_COURTS else ",".join(CURRENT_COURTS)
            prompt = f"[{scope_label} | K={TOP_K}] Query > "
            q = input(prompt).strip()
            
            if not q or q.lower() in ("exit", "quit", "q"):
                break
            
            if q.startswith("/courts "):
                parts = q.split(" ", 1)[1].strip()
                CURRENT_COURTS = [c.strip().lower() for c in parts.split(",") if c.strip()]
                print(f"Filter set to: {CURRENT_COURTS}\n")
                continue
            elif q == "/all":
                CURRENT_COURTS = None
                print("Court filter cleared (searching all).\n")
                continue
            elif q.startswith("/topk "):
                try:
                    TOP_K = int(q.split(" ", 1)[1].strip())
                    print(f"Top-K set to: {TOP_K}\n")
                except ValueError:
                    print("Invalid integer for topk.\n")
                continue
                
            run_query(q, top_k=TOP_K, courts=CURRENT_COURTS)
        except KeyboardInterrupt:
            print("\nExiting.")
            break