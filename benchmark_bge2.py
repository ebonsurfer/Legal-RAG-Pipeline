import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time
import torch
torch.set_num_threads(8)

from sentence_transformers import SentenceTransformer
import psycopg2

DB_PASS = "GregO@5512"   # Put your PostgreSQL password here

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading BGE-large...", flush=True)

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cuda"
)

model.half()

print("Model dtype:", next(model.parameters()).dtype)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model device: {model.device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# ---------------------------------------------------------
# Get SAME 1,280 real opinions for every test
# ---------------------------------------------------------

conn = psycopg2.connect(
    dbname="courtlistener",
    user="greg",
    password=DB_PASS,
    host="localhost",
    port="5432"
)

cur = conn.cursor()

print("Fetching test opinions...", flush=True)

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

texts = [r[0][:2500] for r in cur.fetchall()]

cur.close()
conn.close()

print(f"Fetched: {len(texts)} opinions")
print(f"Average length: {sum(len(t) for t in texts)/len(texts):.0f} chars")

# ---------------------------------------------------------
# Initial GPU warmup
# ---------------------------------------------------------

print("\nWarming up GPU...", flush=True)

model.encode(
    texts[:64],
    batch_size=32,
    show_progress_bar=False
)

torch.cuda.synchronize()

# ---------------------------------------------------------
# Batch-size benchmark
# ---------------------------------------------------------

batch_sizes = [16, 32, 64, 128, 256]

print("\n" + "=" * 72)
print("BGE-LARGE BATCH SIZE BENCHMARK")
print("=" * 72)

results = []

for bs in batch_sizes:

    print(f"\nTesting batch size {bs}...", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        torch.cuda.synchronize()
        start = time.perf_counter()

        model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=False
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        docs_sec = len(texts) / elapsed
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

        results.append(
            (bs, elapsed, docs_sec, peak_vram)
        )

        print(
            f"Batch {bs:3d}: "
            f"{elapsed:7.2f} sec | "
            f"{docs_sec:6.2f} docs/sec | "
            f"{peak_vram:5.2f} GB peak VRAM"
        )

    except torch.cuda.OutOfMemoryError:

        print(f"Batch {bs:3d}: CUDA OUT OF MEMORY")

        results.append(
            (bs, None, None, None)
        )

        torch.cuda.empty_cache()

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n")
print("=" * 72)
print("FINAL RESULTS")
print("=" * 72)

print(
    f"{'Batch':>8} "
    f"{'Seconds':>12} "
    f"{'Docs/sec':>12} "
    f"{'Peak VRAM':>12}"
)

print("-" * 72)

for bs, elapsed, docs_sec, peak_vram in results:

    if elapsed is None:
        print(
            f"{bs:>8} "
            f"{'OOM':>12} "
            f"{'-':>12} "
            f"{'-':>12}"
        )
    else:
        print(
            f"{bs:>8} "
            f"{elapsed:>12.2f} "
            f"{docs_sec:>12.2f} "
            f"{peak_vram:>11.2f}G"
        )

valid = [r for r in results if r[2] is not None]

if valid:
    best = max(valid, key=lambda x: x[2])

    print("\nFASTEST:")
    print(
        f"Batch {best[0]} = "
        f"{best[2]:.2f} docs/sec "
        f"({best[1]:.2f} seconds)"
    )

print("=" * 72)