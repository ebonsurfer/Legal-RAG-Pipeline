import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time
import torch
torch.set_num_threads(8)

from sentence_transformers import SentenceTransformer
import psycopg2

DB_PASS = "GregO@5512"

print("1. Loading BGE model...", flush=True)

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cuda"
)

print("2. Model loaded.", flush=True)
print("   CUDA:", torch.cuda.is_available(), flush=True)
print("   GPU:", torch.cuda.get_device_name(0), flush=True)
print("   Model device:", model.device, flush=True)

print("3. Connecting to PostgreSQL...", flush=True)

conn = psycopg2.connect(
    dbname="courtlistener",
    user="greg",
    password=DB_PASS,
    host="localhost",
    port="5432"
)
cur = conn.cursor()

print("4. Running opinion query...", flush=True)

cur.execute("""
    SELECT o.plain_text
    FROM opinions o
    JOIN opinion_clusters oc ON o.cluster_id = oc.id
    JOIN dockets d ON oc.docket_id = d.id
    WHERE d.court_id IN (SELECT id FROM embed_scope_courts)
      AND o.plain_text IS NOT NULL
      AND length(o.plain_text) > 100
    LIMIT 1280
""")

print("5. Query complete. Fetching rows...", flush=True)

rows = cur.fetchall()

print(f"6. Fetched {len(rows)} rows.", flush=True)

texts = [r[0][:2500] for r in rows]

print(
    f"   Avg length: {sum(len(t) for t in texts)/len(texts):.0f} chars",
    flush=True
)

print("7. Starting 128-document warmup...", flush=True)

model.encode(
    texts[:128],
    batch_size=128,
    show_progress_bar=True
)

torch.cuda.synchronize()

print("8. Warmup complete. Starting 1,280-document benchmark...", flush=True)

t0 = time.time()

model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True
)

torch.cuda.synchronize()
dt = time.time() - t0

print("9. Benchmark complete.", flush=True)
print(f"   Time: {dt:.2f} seconds")
print(f"   Throughput: {len(texts)/dt:.1f} docs/sec")

cur.close()
conn.close()