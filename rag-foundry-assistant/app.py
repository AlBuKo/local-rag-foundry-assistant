"""
app.py - Streamlit Web Interface for Local RAG Knowledge Assistant

Interactive web application powered by Microsoft Foundry Local SDK and SQLite.
Allows users to ingest local text documents (.txt, .md), execute vector similarity searches,
and chat with a local LLM grounded strictly on retrieved context.
"""

import os
import streamlit as st
from typing import Dict, Any

from database import DEFAULT_DB_PATH, get_indexed_files, get_chunk_count
from ingestion import ingest_documents, DEFAULT_EMBEDDING_MODEL, LocalEmbeddingProvider
from rag_engine import (
    DEFAULT_LLM_MODEL,
    SYSTEM_PROMPT,
    LocalLLMProvider,
    retrieve_top_k,
    format_context_prompt,
    get_chunk_count
)


@st.cache_resource
def get_cached_llm_provider(alias: str) -> LocalLLMProvider:
    """Keep LLM model instance cached in RAM/VRAM permanently."""
    provider = LocalLLMProvider(model_alias=alias)
    provider.initialize()
    return provider


@st.cache_resource
def get_cached_embedding_provider(alias: str) -> LocalEmbeddingProvider:
    """Keep embedding model instance cached in RAM/VRAM permanently."""
    provider = LocalEmbeddingProvider(model_alias=alias)
    provider.initialize()
    return provider


# Streamlit Page Setup
st.set_page_config(
    page_title="Local RAG Assistant - Microsoft Foundry Local",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark UI
st.markdown("""
<style>
    /* Main Theme Overrides */
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header Styling */
    .app-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 2rem;
    }
    
    .app-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    /* Chunk Score Badge */
    .score-badge {
        background-color: #0369a1;
        color: #e0f2fe;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .file-badge {
        background-color: #334155;
        color: #f8fafc;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingest_status" not in st.session_state:
        st.session_state.ingest_status = None


def render_sidebar():
    """Render sidebar controls and document ingestion panel."""
    with st.sidebar:
        st.title("⚙: Settings & Data")
        st.markdown("---")
        
        st.subheader("🤖 Local AI Models")
        llm_model = st.text_input("LLM Model Alias", value=DEFAULT_LLM_MODEL, help="Foundry Local LLM (e.g. phi-3.5-mini)")

        emb_model = st.text_input("Embedding Model Alias", value=DEFAULT_EMBEDDING_MODEL, help="Foundry Local Embedding model (e.g. qwen3-embedding-0.6b)")
        k_top = st.slider("Top-K Retrieval Chunks", min_value=1, max_value=10, value=6)

        st.markdown("---")
        st.subheader("📚 Document Knowledge Base")
        
        # Check current document status
        indexed_files = get_indexed_files(DEFAULT_DB_PATH)
        chunk_count = get_chunk_count(DEFAULT_DB_PATH)
        
        st.info(f"**Database Status:** {chunk_count} chunks indexed from {len(indexed_files)} files.")
        
        if indexed_files:
            st.markdown("**Currently Indexed Files:**")
            for f in indexed_files:
                st.markdown(f"- 📄 `{f}`")
        else:
            st.warning("No documents currently indexed in SQLite.")
            
        st.markdown("")
        # Ingestion Button
        if st.button("🚀 Process & Ingest Documents into SQLite", use_container_width=True, type="primary"):
            progress_bar = st.progress(0, text="Starting ingestion...")
            status_box = st.empty()
            
            def update_progress(msg: str, pct: float):
                progress_bar.progress(pct, text=msg)
                status_box.caption(msg)
                
            try:
                res = ingest_documents(
                    docs_dir="documents",
                    db_path=DEFAULT_DB_PATH,
                    model_alias=emb_model,
                    progress_callback=update_progress
                )
                st.session_state.ingest_status = res
                st.success(res["message"])
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                
        return llm_model, emb_model, k_top


def render_header():
    """Render main title header."""
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🤖 Local RAG Knowledge Assistant</div>
        <div class="app-subtitle">
            Offline Retrieval-Augmented Generation powered by <b>Microsoft Foundry Local SDK</b> and <b>SQLite</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    init_session_state()
    llm_model, emb_model, k_top = render_sidebar()
    render_header()
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If message contains retrieved chunks, display them in an expander
            if "retrieved_chunks" in message and message["retrieved_chunks"]:
                with st.expander("🔍 View Retrieved Context Chunks & Similarity Scores"):
                    for idx, chunk in enumerate(message["retrieved_chunks"], 1):
                        st.markdown(
                            f"**Passage {idx}** | "
                            f"<span class='file-badge'>📄 {chunk['file_name']} (Chunk {chunk['chunk_id']})</span> | "
                            f"<span class='score-badge'>Cosine Similarity: {chunk['score']:.4f}</span>",
                            unsafe_allow_html=True
                        )
                        st.text_area(
                            f"Content ({chunk['file_name']})",
                            value=chunk["content"],
                            height=120,
                            key=f"hist_{message.get('id', 0)}_{idx}"
                        )
                        st.divider()

    # User Query Input
    if prompt := st.chat_input("Ask a question about your local documents..."):
        # Append User Message
        msg_id = len(st.session_state.messages)
        st.session_state.messages.append({
            "id": msg_id,
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate Assistant Response
        with st.chat_message("assistant"):
            emb_provider = get_cached_embedding_provider(emb_model)
            llm_provider = get_cached_llm_provider(llm_model)
            
            total_chunks = get_chunk_count(DEFAULT_DB_PATH)
            if total_chunks == 0:
                answer_text = "Bu bilgi belgelerimde bulunmuyor. (Veritabanı boş, lütfen önce belgeleri işleyip SQLite'a aktarın.)"
                st.markdown(answer_text)
                retrieved_chunks = []
            else:
                top_chunks = retrieve_top_k(
                    query=prompt,
                    k=k_top,
                    db_path=DEFAULT_DB_PATH,
                    embedding_provider=emb_provider,
                    min_score=0.20
                )
                
                if not top_chunks:
                    answer_text = "Bu bilgi belgelerimde bulunmuyor."
                    st.markdown(answer_text)
                    retrieved_chunks = []
                else:
                    user_prompt = format_context_prompt(prompt, top_chunks)
                    # Live real-time token streaming
                    stream_generator = llm_provider.generate_stream(SYSTEM_PROMPT, user_prompt)
                    answer_text = st.write_stream(stream_generator)
                    retrieved_chunks = top_chunks
                
            # Expandable retrieved context chunks
            if retrieved_chunks:
                with st.expander("🔍 View Retrieved Context Chunks & Similarity Scores"):
                    for idx, chunk in enumerate(retrieved_chunks, 1):
                        st.markdown(
                            f"**Passage {idx}** | "
                            f"<span class='file-badge'>📄 {chunk['file_name']} (Chunk {chunk['chunk_id']})</span> | "
                            f"<span class='score-badge'>Cosine Similarity: {chunk['score']:.4f}</span>",
                            unsafe_allow_html=True
                        )
                        st.text_area(
                            f"Content ({chunk['file_name']})",
                            value=chunk["content"],
                            height=120,
                            key=f"curr_{msg_id}_{idx}"
                        )
                        st.divider()
                            
        # Append Assistant Response to Session History
        st.session_state.messages.append({
            "id": msg_id + 1,
            "role": "assistant",
            "content": answer_text,
            "retrieved_chunks": retrieved_chunks
        })


if __name__ == "__main__":
    main()
