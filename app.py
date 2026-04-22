import os
import streamlit as st
from graph import run_graph
from memory.conversation import get_chat_history, clear_history
from tools.doc_loader import load_and_preprocess

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Research Agent",
    page_icon="🤖",
    layout="wide"
)

# ---------------- Session State Init ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_1"

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "collection_name" not in st.session_state:
    st.session_state.collection_name = ""

if "has_document" not in st.session_state:
    st.session_state.has_document = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚙️ Settings")

    # Document Upload
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            text = load_and_preprocess(uploaded_file, file_type="pdf")
            collection_name = os.path.splitext(uploaded_file.name)[0]

            st.session_state.document_text = text
            st.session_state.collection_name = collection_name
            st.session_state.has_document = True

        st.success(f"✅ Document loaded: {uploaded_file.name}")

    # Document status
    if st.session_state.has_document:
        st.info(f"📄 Active doc: {st.session_state.collection_name}")
    else:
        st.warning("⚠️ No document uploaded")

    st.divider()

    # Session ID
    st.subheader("🔑 Session")
    session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id
    )
    st.session_state.session_id = session_id

    st.divider()

    # Clear History
    if st.button("🗑️ Clear Chat History"):
        clear_history(st.session_state.session_id)
        st.session_state.messages = []
        st.success("History cleared!")
        st.rerun()


# ---------------- Main UI ----------------
st.title("🤖 Personal Research Agent")
st.caption("Powered by LangGraph + Ollama + RAG + Web Search")

# Route indicator
col1, col2 = st.columns(2)
with col1:
    st.metric("Document", "✅ Loaded" if st.session_state.has_document else "❌ None")
with col2:
    st.metric("LLM", "llama3.2 (local)")

st.divider()

# ---------------- Chat History Display ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Source badge
        if msg.get("source"):
            if msg["source"] == "rag":
                st.caption("📄 Source: Document (RAG)")
            else:
                st.caption("🔍 Source: Web Search")

        # Show retrieved chunks for RAG
        if msg.get("top_docs"):
            with st.expander("📑 Retrieved Chunks"):
                for i, doc in enumerate(msg["top_docs"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)

        # Show web results
        if msg.get("raw_results"):
            with st.expander("🌐 Web Search Results"):
                for r in msg["raw_results"]:
                    st.markdown(f"**{r['title']}**")
                    st.write(r['snippet'])
                    st.caption(r['url'])


# ---------------- Chat Input ----------------
query = st.chat_input("Ask anything...")

if query:
    # User message display
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Agent run karo
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = run_graph(
                query=query,
                has_document=st.session_state.has_document,
                document_text=st.session_state.document_text,
                collection_name=st.session_state.collection_name,
                session_id=st.session_state.session_id
            )

        # Answer display
        st.write(result["answer"])

        # Source badge
        if result["source"] == "rag":
            st.caption("📄 Source: Document (RAG)")
        else:
            st.caption("🔍 Source: Web Search")

        # Query expansions (RAG)
        if result.get("expansions"):
            with st.expander("🔍 Query Expansions"):
                for q in result["expansions"]:
                    st.write(f"• {q}")

        # Retrieved chunks (RAG)
        if result.get("top_docs"):
            with st.expander("📑 Retrieved Chunks"):
                for i, doc in enumerate(result["top_docs"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)

        # Web results
        if result.get("raw_results"):
            with st.expander("🌐 Web Search Results"):
                for r in result["raw_results"]:
                    st.markdown(f"**{r['title']}**")
                    st.write(r['snippet'])
                    st.caption(r['url'])

    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "source": result.get("source"),
        "top_docs": result.get("top_docs"),
        "raw_results": result.get("raw_results")
    })