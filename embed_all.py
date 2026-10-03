import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import gc
import time
import torch
torch.set_num_threads(8)

from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values

DB_PASS = os.environ.get("COURTLISTENER_DB_PASS")
if not DB_PASS:
    raise SystemExit("Set COURTLISTENER_DB_PASS first")

DB_NAME = "courtlistener"
DB_USER = "greg"
DB_HOST = "localhost"
DB_PORT = "5432"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_CHARS = 2000
BATCH_SIZE = 128

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.half()

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS,
                            host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()

    cur.execute("SELECT id FROM embed_scope_courts")
    court_ids = tuple(r[0] for r in cur.fetchall())
    print(f"Courts in scope: {len(court_ids)}", flush=True)

    cur.execute("""
        SELECT count(*) FROM opinions o
        JOIN opinion_clusters oc ON o.cluster_id = oc.id
        JOIN dockets d ON oc.docket_id = d.id
        WHERE d.court_id = ANY(%s)
          AND o.plain_text IS NOT NULL
          AND length(o.plain_text) > 100
    """, (list(court_ids),))
    total = cur.fetchone()[0]
    print(f"Total scope: {total:,}", flush=True)

    processed = 0
    last_id = 0
    t_start = time.time()

    while True:
        cur.execute("""
            SELECT o.id, o.plain_text FROM opinions o
            JOIN opinion_clusters oc ON o.cluster_id = oc.id
            JOIN dockets d ON oc.docket_id = d.id
            WHERE d.court_id = ANY(%s)
              AND o.plain_text IS NOT NULL
              AND length(o.plain_text) > 100
              AND o.id > %s
            ORDER BY o.id
            LIMIT %s
        """, (list(court_ids), last_id, BATCH_SIZE))

        rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [r[1][:MAX_CHARS] for r in rows]
        last_id = ids[-1]

        with torch.no_grad():
            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        data = [(int(i), emb.tolist()) for i, emb in zip(ids, embeddings)]
        del embeddings, texts

        execute_values(cur, """
            INSERT INTO opinion_embeddings (opinion_id, embedding)
            VALUES %s
            ON CONFLICT (opinion_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """, data)
        conn.commit()

        n = len(rows)
        del data, ids, rows
        gc.collect()
        torch.cuda.empty_cache()

        processed += n
        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        eta_sec = (total - processed) / rate if rate > 0 else 0
        print(
            f"Processed {processed:,}/{total:,} "
            f"| {rate:.1f} op/sec "
            f"| ETA: {eta_sec/3600:.1f}h",
            flush=True,
        )

    cur.close()
    conn.close()
    print("Done.", flush=True)

if __name__ == "__main__":
    main()