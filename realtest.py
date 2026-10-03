import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time
import torch
torch.set_num_threads(8)

from sentence_transformers import SentenceTransformer
import psycopg2

DB_PASS = os.environ.get("COURTLISTENER_DB_PASS")
if not DB_PASS:
    raise SystemExit("Set COURTLISTENER_DB_PASS environment variable first")

MODEL_NAME = "BAAI/bge-large-en-v1.5"

model = SentenceTransformer(MODEL_NAME, device="cuda")
model.half()  # FP16

conn = psycopg2.connect(
    dbname="courtlistener", user="greg", password=DB_PASS,
    host="localhost", port="5432"
)
cur = conn.cursor()

cur.execute("""
    SELECT o.plain_text FROM opinions o
    JOIN opinion_clusters oc ON o.cluster_id = oc.id
    JOIN dockets d ON oc.docket_id = d.id
    WHERE d.court_id IN (SELECT id FROM embed_scope_courts)
      AND o.plain_text IS NOT NULL
      AND length(o.plain_text) > 100
    LIMIT 1280
""")
texts = [r[0][:2500] for r in cur.fetchall()]
print(f"Fetched {len(texts)} real opinions")
print(f"Avg length: {sum(len(t) for t in texts)/len(texts):.0f} chars")

# Warmup
model.encode(texts[:128], batch_size=128, show_progress_bar=False)

t0 = time.time()
model.encode(texts, batch_size=128, show_progress_bar=False)
dt = time.time() - t0
print(f"{len(texts)/dt:.1f} docs/sec (real opinions, FP16)")

cur.close()
conn.close()