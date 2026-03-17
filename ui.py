"""
ui.py — Streamlit interface for the AI Research Assistant (RAG).

Run with:
    streamlit run ui.py
"""

import streamlit as st
from app.ingest import load_multiple_pdfs
from app.embed import create_vectorstore, vectorstore_exists
from app.retriever import get_relevant_docs
from app.generator import generate_answer

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e6e6e6; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2a2d3e; }

    /* Chat message boxes */
    .user-msg {
        background: #1e2a3a;
        border-left: 3px solid #4f8ef7;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin: 8px 0;
        color: #e6e6e6;
    }
    .ai-msg {
        background: #1a2230;
        border-left: 3px solid #2dca8c;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin: 8px 0;
        color: #e6e6e6;
    }
    .source-tag {
        display: inline-block;
        background: #2a3a4a;
        color: #7eb8f7;
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 12px;
        margin: 2px 3px;
    }
    /* Stats box */
    .stat-card {
        background: #1e2532;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .stat-num { font-size: 24px; font-weight: 700; color: #4f8ef7; }
    .stat-label { font-size: 12px; color: #8890a4; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ──────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"|"ai", "content": str}

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"chunks": 0, "files": 0, "ready": False}

if "raw_history" not in st.session_state:
    st.session_state.raw_history = []  # flat Q/A strings for the LLM prompt

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📚 Document Upload")
    st.markdown("Upload one or more PDF files to build the knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")

    process_btn = st.button(
        "⚙️ Process Documents",
        use_container_width=True,
        disabled=not uploaded_files
    )

    if process_btn and uploaded_files:
        with st.spinner("Chunking and embedding documents..."):
            try:
                chunks = load_multiple_pdfs(uploaded_files)
                create_vectorstore(chunks)

                st.session_state.doc_stats = {
                    "chunks": len(chunks),
                    "files": len(uploaded_files),
                    "ready": True
                }
                st.session_state.chat_history = []
                st.session_state.raw_history = []
                st.success(f"✅ Processed {len(chunks)} chunks from {len(uploaded_files)} file(s)!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Stats
    if st.session_state.doc_stats["ready"]:
        st.markdown("### 📊 Knowledge Base")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{st.session_state.doc_stats['files']}</div>
                <div class="stat-label">Files</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{st.session_state.doc_stats['chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>""", unsafe_allow_html=True)
        st.divider()

    # Controls
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=5)
    show_sources = st.checkbox("Show source chunks", value=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.raw_history = []
        st.rerun()

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("""
    - 🧠 LLM: Gemini Pro  
    - 🔍 Embeddings: HuggingFace  
    - 🗄️ Vector DB: FAISS  
    - 🔗 Framework: LangChain  
    - 🖥️ UI: Streamlit
    """)

# ─── Main content ─────────────────────────────────────────────────────────────

st.markdown("# 📚 AI Research Assistant")
st.markdown("Ask questions about your uploaded documents using Retrieval-Augmented Generation.")

# Check if system is ready
is_ready = st.session_state.doc_stats["ready"] or vectorstore_exists()

if not is_ready:
    st.warning("⬅️ Upload and process your PDF documents in the sidebar to get started.")

# Chat display
st.markdown("---")
st.markdown("### 💬 Conversation")

chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown(
            '<div style="color: #555; text-align: center; padding: 40px 0;">No messages yet. Ask a question below!</div>',
            unsafe_allow_html=True
        )
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg">🧑 <strong>You:</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ai-msg">🤖 <strong>Assistant:</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
                if show_sources and "sources" in msg:
                    src_html = "".join(
                        f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"]
                    )
                    st.markdown(
                        f'<div style="margin: 4px 0 12px 8px;">Sources: {src_html}</div>',
                        unsafe_allow_html=True
                    )

# ─── Query input ──────────────────────────────────────────────────────────────

st.markdown("---")

with st.form("query_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Your question",
            placeholder="Ask anything about your documents...",
            label_visibility="collapsed",
            disabled=not is_ready
        )
    with col2:
        submit = st.form_submit_button("Send →", use_container_width=True, disabled=not is_ready)

if submit and query.strip():
    with st.spinner("Retrieving context and generating answer..."):
        try:
            retrieved_docs = get_relevant_docs(query, k=top_k)
            answer = generate_answer(query, retrieved_docs, st.session_state.raw_history)

            # Collect unique sources
            sources = list({
                doc.metadata.get("source", "Unknown")
                for doc in retrieved_docs
            })

            # Update session state
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({
                "role": "ai",
                "content": answer,
                "sources": sources
            })
            st.session_state.raw_history.append(f"Q: {query}")
            st.session_state.raw_history.append(f"A: {answer}")

            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")