import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import torch

DB_NAME = "courtlistener"
DB_USER = "greg"
DB_PASS = "GregO@5512"   # <-- FIX
DB_HOST = "localhost"
DB_PORT = "5432"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 128
MAX_CHARS = 2500
TEST_LIMIT = 12800   # <-- tiny test run

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS,
                            host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()

    cur.execute("SELECT id FROM embed_scope_courts")
    court_ids = tuple(r[0] for r in cur.fetchall())
    print(f"Courts in scope: {len(court_ids)}", flush=True)

    cur.execute("""
        SELECT o.id, o.plain_text FROM opinions o
        JOIN opinion_clusters oc ON o.cluster_id = oc.id
        JOIN dockets d ON oc.docket_id = d.id
        WHERE d.court_id = ANY(%s)
          AND o.plain_text IS NOT NULL
          AND length(o.plain_text) > 100
          AND o.id NOT IN (SELECT opinion_id FROM opinion_embeddings)
        LIMIT %s
    """, (list(court_ids), TEST_LIMIT))

    rows = cur.fetchall()
    print(f"Fetched {len(rows)} rows", flush=True)

    if not rows:
        print("Nothing to embed. Aborting.", flush=True)
        return

    ids = [r[0] for r in rows]
    texts = [r[1][:MAX_CHARS] for r in rows]

    print("Encoding...", flush=True)
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=BATCH_SIZE)
    print(f"Encoded shape: {embeddings.shape}", flush=True)

    # Sanity check: all vectors should be unit length (normalized)
    import numpy as np
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norm range: {norms.min():.4f} to {norms.max():.4f}", flush=True)

    # Check that embeddings aren't all identical (model working, not degenerate)
    variance = embeddings.var(axis=0).mean()
    print(f"Mean variance across dims: {variance:.6f}", flush=True)

    data = [(int(i), emb.tolist()) for i, emb in zip(ids, embeddings)]

    execute_values(cur, """
        INSERT INTO opinion_embeddings (opinion_id, embedding)
        VALUES %s
        ON CONFLICT (opinion_id) DO UPDATE SET embedding = EXCLUDED.embedding
    """, data)
    conn.commit()

    print(f"Inserted {len(data)} rows", flush=True)

    # Verify count
    cur.execute("SELECT count(*) FROM opinion_embeddings")
    print(f"Table count now: {cur.fetchone()[0]}", flush=True)

    cur.close()
    conn.close()
    print("Test complete.", flush=True)

if __name__ == "__main__":
    main()