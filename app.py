
import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.retriever import Retriever
from src.llm_handler import LLMHandler

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# ─── Initialize Session State ──────────────────────────────
@st.cache_resource
def load_rag_system():
    doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    embedding_manager = EmbeddingManager(collection_name="documents")
    retriever = Retriever(embedding_manager, top_k=3)
    llm = LLMHandler()
    return doc_processor, embedding_manager, retriever, llm

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []


# ─── Load System ───────────────────────────────────────────
try:
    doc_processor, embedding_manager, retriever, llm = load_rag_system()
    system_ready = True
except Exception as e:
    system_ready = False
    st.error(f"❌ System Error: {str(e)}")


# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.markdown("---")

    # Stats
    if system_ready:
        chunk_count = embedding_manager.collection.count()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", chunk_count)
        with col2:
            st.metric("Docs", len(st.session_state.uploaded_files_list))

    st.markdown("---")

    # Upload Section
    st.subheader("📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "docx"]
    )

    if uploaded_file and system_ready:
        if st.button("⬆️ Process Document"):
            with st.spinner("Processing..."):
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    chunks = doc_processor.load_document(temp_path)
                    embedding_manager.add_documents(chunks)
                    os.remove(temp_path)

                    if uploaded_file.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uploaded_file.name)

                    st.success(f"✅ {uploaded_file.name} — {len(chunks)} chunks added!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Uploaded files list
    if st.session_state.uploaded_files_list:
        st.markdown("---")
        st.subheader("📄 Loaded Documents")
        for fname in st.session_state.uploaded_files_list:
            st.markdown(f"• `{fname}`")

    st.markdown("---")

    # Clear DB
    if st.button("🗑️ Clear Database"):
        if system_ready:
            embedding_manager.clear_database()
            st.session_state.uploaded_files_list = []
            st.session_state.chat_history = []
            st.success("Database cleared!")
            st.rerun()

    st.markdown("---")
    st.markdown("**🤖 Model:** `llama3-8b-8192`")
    st.markdown("**⚡ Provider:** `Groq (Free)`")


# ─── Main Area ─────────────────────────────────────────────
st.title("💬 Ask Your Documents")
st.caption("Upload a document from the sidebar, then ask anything about it.")
st.markdown("---")

# Question Input
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Your Question",
        placeholder="e.g. What is machine learning?",
        label_visibility="collapsed"
    )
with col2:
    ask_btn = st.button("Ask →", use_container_width=True)

# Handle question
if ask_btn and query:
    if not system_ready:
        st.error("System not ready.")
    elif embedding_manager.collection.count() == 0:
        st.warning("⚠️ No documents uploaded yet! Use the sidebar to upload a file first.")
    else:
        with st.spinner("🤖 Thinking..."):
            try:
                context, sources = retriever.retrieve(query)
                if not context:
                    st.warning("No relevant information found in documents.")
                else:
                    answer = llm.generate_answer(query, context)
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer,
                        "sources": sources
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")

# Chat History
if st.session_state.chat_history:
    st.subheader("🗨️ Conversation")
    for turn in reversed(st.session_state.chat_history):
        with st.container():
            st.info(f"🧑 **You:** {turn['question']}")
            st.success(f"🤖 **Answer:** {turn['answer']}")

            if turn.get("sources"):
                with st.expander("📎 View Sources"):
                    for src in turn["sources"]:
                        st.markdown(f"**Chunk {src['chunk_number']}** | Similarity: `{src['similarity']}` | Source: `{src['source']}`")
                        st.caption(src['content'])
                        st.markdown("---")
else:
    st.info("👆 Upload a document and ask a question to get started!")
