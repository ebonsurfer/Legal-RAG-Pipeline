import os
import torch
torch.set_num_threads(8)
from sentence_transformers import SentenceTransformer
import psycopg2

DB_PASS = os.environ["COURTLISTENER_DB_PASS"]

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda").half()

query = "landlord liability for lead paint exposure in rental apartments"
qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

conn = psycopg2.connect(dbname="courtlistener", user="greg", password=DB_PASS,
                        host="localhost", port="5432")
cur = conn.cursor()

cur.execute("""
    SELECT
        d.case_name,
        d.court_id,
        oc.date_filed,
        oc.precedential_status,
        substring(o.plain_text from 1500 for 700) AS snippet,
        1 - (e.embedding <=> %s::halfvec) AS similarity
    FROM opinion_embeddings e
    JOIN opinions o ON o.id = e.opinion_id
    JOIN opinion_clusters oc ON o.cluster_id = oc.id
    JOIN dockets d ON oc.docket_id = d.id
    WHERE d.court_id IN ('ny','nyappdiv','nyappterm','nysd','nyed','nynd','nywd','njd')
    ORDER BY e.embedding <=> %s::halfvec
    LIMIT 10
""", (qvec, qvec))

for row in cur.fetchall():
    case, court, date, status, snippet, sim = row
    print(f"\n--- {sim:.3f} | {case} ({court}, {date}) ---")
    print(snippet.replace("\n", " ")[:500])

cur.close()
conn.close()