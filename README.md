# Local Legal RAG Pipeline

A fully local, privacy-focused Legal Retrieval-Augmented Generation (RAG) assistant running on local PostgreSQL, pgvector, and llama.cpp.

## Features
- **Vector Search:** BAAI/bge-large-en-v1.5 embeddings indexed with HNSW via pgvector.
- **Relational Hydration:** Metadata and text extraction over federal and state court decisions.
- **Local Inference:** Qwen2.5-14B-Instruct running on GPU via llama-server.
- **Interactive UI:** Streamlit web dashboard and command-line REPL with jurisdictional filtering.

## Environment Variables
- COURTLISTENER_DB_PASS: PostgreSQL user password.

## Startup Commands
1. Start Local LLM Server:
   .\start_server.ps1

2. Launch Web Dashboard:
   streamlit run app.py

3. Run Interactive CLI:
   python rag_interactive.py
