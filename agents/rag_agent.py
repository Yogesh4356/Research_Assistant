import numpy as np
import torch
from langchain_core.prompts import ChatPromptTemplate  # ← change
from langchain_core.output_parsers import StrOutputParser  # ← change
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from vectorstore.chroma_store import build_or_load_vectorstore, get_retriever


# ---------------- Device ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 RAG Agent using device: {DEVICE}")


# ---------------- LLM ----------------
from config import get_llm
llm = get_llm()


# ---------------- Cross Encoder ----------------
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=DEVICE
)


# ---------------- Query Expansion ----------------
expansion_prompt = ChatPromptTemplate.from_template(
    "Generate 3 alternative queries that mean the same as: {query}. "
    "Return only the queries, one per line, no numbering."
)

expansion_chain = expansion_prompt | llm | StrOutputParser()


def expand_query(query: str) -> list[str]:
    """Original query + 3 expanded variants."""
    expansion = expansion_chain.invoke({"query": query})
    expanded = [q.strip("-• ") for q in expansion.split("\n") if q.strip()]
    return [query] + expanded[:3]  # original + max 3 expansions


# ---------------- Multi Query Retrieval ----------------
def multi_query_retrieve(query: str, retriever) -> tuple[list, list]:
    """Expanded queries se docs retrieve karo."""
    queries = expand_query(query)
    all_docs = []

    for q in queries:
        docs = retriever.invoke(q)  
        all_docs.extend(docs)

    # Deduplicate
    unique_docs = {doc.page_content: doc for doc in all_docs}
    return list(unique_docs.values()), queries


# ---------------- RRF Reranking ----------------
def rrf_rerank(query: str, docs: list, top_k: int = 5, k: int = 60) -> list:
    """Cross-Encoder + BM25 → RRF Fusion."""
    if not docs:
        return []

    # Cross-Encoder scores
    pairs = [(query, d.page_content) for d in docs]
    dense_scores = cross_encoder.predict(pairs)
    dense_ranking = np.argsort(-dense_scores)
    dense_ranks = {idx: rank for rank, idx in enumerate(dense_ranking)}

    # BM25 scores
    tokenized_corpus = [d.page_content.split() for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    sparse_scores = bm25.get_scores(query.split())
    sparse_ranking = np.argsort(-sparse_scores)
    sparse_ranks = {idx: rank for rank, idx in enumerate(sparse_ranking)}

    # RRF Fusion
    fused_scores = {}
    for idx in range(len(docs)):
        rank_dense = dense_ranks.get(idx, len(docs))
        rank_sparse = sparse_ranks.get(idx, len(docs))
        fused_scores[idx] = (1 / (k + rank_dense)) + (1 / (k + rank_sparse))

    reranked_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [docs[i] for i in reranked_indices[:top_k]]


# ---------------- Memory ----------------
_session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


# ---------------- QA Chain ----------------
qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer only from the context above. If not found, say "I don't know".
""")

qa_chain = qa_prompt | llm | StrOutputParser()

qa_with_history = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
    output_messages_key="answer",
)


# ---------------- Main RAG Function ----------------
def run_rag(
    query: str,
    text: str,
    collection_name: str,
    session_id: str = "default",
    top_k: int = 5
) -> dict:
    """
    Main function jo LangGraph node call karega.
    Returns: answer, expansions, top_docs
    """
    # Vectorstore + retriever
    vectorstore = build_or_load_vectorstore(text, collection_name)
    retriever = get_retriever(vectorstore, k=top_k)

    # Retrieve + Rerank
    candidates, expansions = multi_query_retrieve(query, retriever)
    top_docs = rrf_rerank(query, candidates, top_k=top_k)

    # Context
    context = "\n\n".join(d.page_content for d in top_docs)

    # Answer generate 
    answer = qa_with_history.invoke(
        {"query": query, "context": context},
        config={"configurable": {"session_id": session_id}}
    )

    return {
        "answer": answer,
        "expansions": expansions,
        "top_docs": top_docs,
        "source": "rag"
    }