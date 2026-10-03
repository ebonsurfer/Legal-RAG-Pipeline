import os
import time
import torch
import psycopg2

from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "true"

torch.set_num_threads(8)

DB_PASS = os.environ.get("COURTLISTENER_DB_PASS")

if not DB_PASS:
    raise RuntimeError(
        "COURTLISTENER_DB_PASS environment variable is not set.\n"
        'PowerShell: $env:COURTLISTENER_DB_PASS = "your_password"'
    )

MODEL_NAME = "BAAI/bge-large-en-v1.5"

BATCH_SIZE = 32
TEST_DOCUMENTS = 1280

CHAR_LENGTHS = [
    500,
    1000,
    1500,
    2000,
    2500
]


# ============================================================
# CUDA check
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("=" * 72)
print("BGE-LARGE FP16 SEQUENCE-LENGTH BENCHMARK")
print("=" * 72)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")


# ============================================================
# Load BGE
# ============================================================

print("\nLoading BGE-large...", flush=True)

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)

# Critical optimization for RTX 3060
model.half()

print("Model loaded.")
print(f"Model device: {model.device}")
print(f"Model dtype: {next(model.parameters()).dtype}")


# ============================================================
# Connect to PostgreSQL
# ============================================================

print("\nConnecting to PostgreSQL...", flush=True)

conn = psycopg2.connect(
    dbname="courtlistener",
    user="greg",
    password=DB_PASS,
    host="localhost",
    port="5432"
)

cur = conn.cursor()


# ============================================================
# Fetch real CourtListener opinions
#
# IMPORTANT:
# Fetch more than 2500 characters here so every sequence-length
# test uses the same underlying documents.
# ============================================================

print("Fetching real opinions...", flush=True)

cur.execute("""
    SELECT o.plain_text
    FROM opinions o
    JOIN opinion_clusters oc
        ON o.cluster_id = oc.id
    JOIN dockets d
        ON oc.docket_id = d.id
    WHERE d.court_id IN (
        SELECT id
        FROM embed_scope_courts
    )
      AND o.plain_text IS NOT NULL
      AND length(o.plain_text) >= 2500
    LIMIT %s
""", (TEST_DOCUMENTS,))

rows = cur.fetchall()

cur.close()
conn.close()

texts = [row[0] for row in rows]

if not texts:
    raise RuntimeError("No opinions returned from PostgreSQL.")

print(f"Fetched: {len(texts)} opinions")

lengths = [len(t) for t in texts]

print(
    f"Opinion lengths: "
    f"min={min(lengths):,}, "
    f"avg={sum(lengths)/len(lengths):,.0f}, "
    f"max={max(lengths):,} chars"
)


# ============================================================
# Initial CUDA warmup
# ============================================================

print("\nInitial GPU warmup...", flush=True)

warmup_texts = [
    t[:1000]
    for t in texts[:64]
]

with torch.inference_mode():
    model.encode(
        warmup_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True
    )

torch.cuda.synchronize()

print("Warmup complete.")


# ============================================================
# Sequence-length benchmark
# ============================================================

print("\n" + "=" * 72)
print("INPUT LENGTH BENCHMARK")
print(f"Batch size: {BATCH_SIZE}")
print(f"Documents:  {len(texts)}")
print("Precision:  FP16")
print("=" * 72)

results = []

for char_length in CHAR_LENGTHS:

    print(
        f"\nTesting {char_length:,} characters...",
        flush=True
    )

    test_texts = [
        text[:char_length]
        for text in texts
    ]

    # Clear cached allocations between tests
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Small warmup at this particular input length
    with torch.inference_mode():
        model.encode(
            test_texts[:64],
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Timed run
    # --------------------------------------------------------

    start = time.perf_counter()

    with torch.inference_mode():
        model.encode(
            test_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    docs_per_sec = len(test_texts) / elapsed

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    results.append(
        (
            char_length,
            elapsed,
            docs_per_sec,
            peak_vram
        )
    )

    print(
        f"{char_length:5d} chars | "
        f"{elapsed:7.2f} sec | "
        f"{docs_per_sec:8.2f} docs/sec | "
        f"{peak_vram:5.2f} GB VRAM"
    )


# ============================================================
# Final results
# ============================================================

print("\n")
print("=" * 72)
print("FINAL RESULTS")
print("=" * 72)

print(
    f"{'Chars':>8} "
    f"{'Seconds':>12} "
    f"{'Docs/sec':>12} "
    f"{'Peak VRAM':>12} "
    f"{'Speedup':>10}"
)

print("-" * 72)

# Use longest sequence as baseline
baseline_rate = results[-1][2]

for char_length, elapsed, rate, vram in results:

    speedup = rate / baseline_rate

    print(
        f"{char_length:>8,} "
        f"{elapsed:>12.2f} "
        f"{rate:>12.2f} "
        f"{vram:>11.2f}G "
        f"{speedup:>9.2f}x"
    )


# ============================================================
# Fastest result
# ============================================================

best = max(
    results,
    key=lambda x: x[2]
)

print("\nFASTEST:")

print(
    f"{best[0]:,} characters = "
    f"{best[2]:.2f} docs/sec "
    f"({best[1]:.2f} seconds)"
)

print("=" * 72)