import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time
import torch
torch.set_num_threads(8)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

print(f"CPU threads: {torch.get_num_threads()}")
print(f"CUDA: {torch.cuda.is_available()}")

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
model.half()
tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

# Same size text as your real run (~2500 chars)
text = "This is a test legal opinion about contract law and various holdings. " * 35
texts = [text[:2500]] * 1280

# Warmup
print("Warming up...")
model.encode(texts[:128], batch_size=128, show_progress_bar=False)

# Time tokenization alone
print("\n--- Tokenization only ---")
t0 = time.time()
for _ in range(10):
    tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(f"1280 docs tokenized in {time.time()-t0:.2f}s")

# Time full encode (GPU)
print("\n--- Full encode ---")
t0 = time.time()
emb = model.encode(texts, batch_size=128, show_progress_bar=False)
dt = time.time() - t0
print(f"1280 docs encoded in {dt:.2f}s = {1280/dt:.1f} docs/sec")

# Time with larger batch
print("\n--- Full encode, batch 256 ---")
t0 = time.time()
emb = model.encode(texts, batch_size=256, show_progress_bar=False)
dt = time.time() - t0
print(f"1280 docs encoded in {dt:.2f}s = {1280/dt:.1f} docs/sec")