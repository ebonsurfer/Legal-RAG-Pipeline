import os
import sys
import json
import streamlit as st
import psycopg2
import requests
from sentence_transformers import SentenceTransformer

# --- Page Setup ---
st.set_page_config(page_title="CourtListener Legal RAG", layout="wide")

DB_PASS = os.environ.get("COURTLISTENER_DB_PASS")
if not DB_PASS:
    st.error("COURTLISTENER_DB_PASS environment variable is missing.")
    st.stop()

DB_NAME = "courtlistener"
DB_USER = "greg"
DB_HOST = "localhost"
DB_PORT = "5432"
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda").half()

embed_model = load_embedder()

# --- Sidebar Controls ---
st.sidebar.title("Research Parameters")

court_options = {
    "Second Circuit (ca2)": "ca2",
    "S.D.N.Y. (nysd)": "nysd",
    "E.D.N.Y. (nyed)": "nyed",
    "New York State Courts (ny, nyappdiv)": ["ny", "nyappdiv"],
    "New Jersey State Courts (nj, njsuperctappdiv)": ["nj", "njsuperctappdiv"],
    "Third Circuit (ca3)": "ca3",
    "D.N.J. (njd)": "njd"
}

selected_options = st.sidebar.multiselect(
    "Jurisdictional Filter",
    options=list(court_options.keys()),
    default=["Second Circuit (ca2)", "S.D.N.Y. (nysd)", "E.D.N.Y. (nyed)"]
)

active_courts = []
for sel in selected_options:
    val = court_options[sel]
    if isinstance(val, list):
        active_courts.extend(val)
    else:
        active_courts.append(val)

top_k = st.sidebar.slider("Precedents to Retrieve (Top-K)", min_value=3, max_value=15, value=8)

# --- Retrieval Function ---
def retrieve(query_text, k, courts):
    qvec = embed_model.encode([query_text], normalize_embeddings=True)[0].tolist()
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    cur.execute("SET hnsw.iterative_scan = relaxed_order;")
    cur.execute("SET hnsw.ef_search = 100;")

    candidate_limit = k * 4 if courts else k * 2

    if courts:
        sql = """
            SELECT e.opinion_id, oc.id AS cluster_id, d.case_name, d.court_id, oc.date_filed,
                   (1 - (e.embedding <=> %s::halfvec)) AS similarity,
                   (1 - (e.embedding <=> %s::halfvec)) * (CASE WHEN d.court_id IN ('ca2', 'ny', 'nj', 'ca3') THEN 1.05 ELSE 1.0 END) AS score
            FROM opinion_embeddings e
            JOIN opinions o ON o.id = e.opinion_id
            JOIN opinion_clusters oc ON o.cluster_id = oc.id
            JOIN dockets d ON oc.docket_id = d.id
            WHERE d.court_id = ANY(%s)
            ORDER BY e.embedding <=> %s::halfvec
            LIMIT %s;
        """
        cur.execute(sql, (qvec, qvec, courts, qvec, candidate_limit))
    else:
        sql = """
            SELECT e.opinion_id, oc.id AS cluster_id, d.case_name, d.court_id, oc.date_filed,
                   (1 - (e.embedding <=> %s::halfvec)) AS similarity,
                   (1 - (e.embedding <=> %s::halfvec)) AS score
            FROM opinion_embeddings e
            JOIN opinions o ON o.id = e.opinion_id
            JOIN opinion_clusters oc ON o.cluster_id = oc.id
            JOIN dockets d ON oc.docket_id = d.id
            ORDER BY e.embedding <=> %s::halfvec
            LIMIT %s;
        """
        cur.execute(sql, (qvec, qvec, candidate_limit))

    candidates = cur.fetchall()
    if not candidates:
        cur.close()
        conn.close()
        return []

    # Dedup
    seen = set()
    deduped = []
    candidates.sort(key=lambda x: x[6], reverse=True)
    for c in candidates:
        if c[1] not in seen:
            seen.add(c[1])
            deduped.append(c)
            if len(deduped) == k:
                break

    ids = [r[0] for r in deduped]
    cur.execute("SELECT id, substring(plain_text from 1000 for 2500) FROM opinions WHERE id = ANY(%s);", (ids,))
    texts = {r[0]: (r[1] or "") for r in cur.fetchall()}
    cur.close()
    conn.close()

    results = []
    for r in deduped:
        results.append({
            "opinion_id": r[0],
            "case_name": r[2],
            "court": r[3],
            "date": str(r[4]).split("-")[0] if r[4] else "Unknown",
            "sim": r[5],
            "text": texts.get(r[0], "")
        })
    return results

# --- Main Application Layout ---
st.title("⚖️ Local Legal Research Assistant")

query = st.text_input("Enter Legal Research Inquiry:", value="Under Title VII, what standard applies to establish a hostile work environment claim?")

if st.button("Run Research", type="primary") and query:
    with st.spinner("Searching case law..."):
        authorities = retrieve(query, top_k, active_courts if active_courts else None)

    if not authorities:
        st.warning("No matching opinions found for the chosen filters.")
    else:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Legal Synthesis")
            
            # Format Prompt
            context_str = ""
            for idx, a in enumerate(authorities, 1):
                clean = a['text'].replace('\r', ' ').replace('\n', ' ').strip()
                context_str += f"\n--- Source [{idx}]: {a['case_name']} ({a['court']}, {a['date']}) ---\n{clean}\n"

            messages = [
                {"role": "system", "content": "You are an expert legal research assistant. Analyze the inquiry based STRICTLY on the provided opinions. Follow substantive legal propositions with inline citations in the form [Case Name (Court, Year)]. If context is insufficient, state that clearly."},
                {"role": "user", "content": f"Context Authorities:\n{context_str}\nLegal Question: {query}\n\nSynthesize the applicable law:"}
            ]

            # Stream LLM Response
            placeholder = st.empty()
            full_response = ""
            try:
                resp = requests.post(LLAMA_SERVER_URL, json={"messages": messages, "temperature": 0.2, "max_tokens": 1024, "stream": True}, stream=True)
                for chunk in resp.iter_lines():
                    if chunk:
                        line = chunk.decode("utf-8")
                        if line.startswith("data: ") and line != "data: [DONE]":
                            delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_response += delta
                                placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Inference error: {e}")

        with col2:
            st.subheader(f"Retrieved Precedents ({len(authorities)})")
            for a in authorities:
                with st.expander(f"[{a['sim']:.3f}] {a['case_name']} ({a['court']}, {a['date']})"):
                    st.caption(f"Opinion ID: {a['opinion_id']}")
                    st.text(a['text'])