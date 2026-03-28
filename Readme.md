# 🤖 Personal Research Agent

A production-inspired multi-agent AI assistant built with LangGraph, Ollama, and RAG — runs fully locally with zero API costs.

## 🎯 What It Does

Ask anything — the agent automatically decides whether to search the web or query your uploaded document, then returns a well-reasoned answer with sources.

**Example:**
- *"What is LangGraph?"* → Web Search Agent
- *"What does the document say about embeddings?"* → RAG Agent

---

## 🏗️ Architecture
```
User Query
    ↓
[Router Agent] — decides: RAG or Web Search?
    ↓                          ↓
[RAG Agent]          [Web Search Agent]
  - Query Expansion      - DuckDuckGo Search
  - BM25 + CrossEncoder  - LLM Summarization
  - RRF Reranking        - Source Attribution
    ↓                          ↓
         [Final Answer + Sources]
                ↓
        [SQLite Memory]
```

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Agent Orchestration | LangGraph |
| LLM (Local) | Ollama — llama3.2 |
| Embeddings | Ollama — nomic-embed-text |
| Vector Database | ChromaDB |
| Hybrid Reranking | CrossEncoder + BM25 + RRF Fusion |
| Web Search | DuckDuckGo (ddgs) |
| Persistent Memory | SQLite |
| UI | Streamlit |
| Cost | $0 — fully local & free |

---

## 🚀 Features

- **Smart Routing** — LLM-based router auto-selects RAG or Web Search
- **Hybrid RAG Pipeline** — Query expansion + BM25 + CrossEncoder + RRF reranking
- **Persistent Memory** — Conversation history saved in SQLite across sessions
- **Local LLM** — No API keys, no cost, full privacy via Ollama
- **Web Search** — Real-time answers via DuckDuckGo
- **GPU Support** — Auto-detects CUDA for CrossEncoder inference

---

## 📁 Project Structure
```
research_agent/
├── agents/
│   ├── router.py          # Query routing — RAG vs Web Search
│   ├── rag_agent.py       # RAG pipeline with hybrid reranking
│   └── search_agent.py    # Web search + LLM summarization
├── tools/
│   ├── web_search.py      # DuckDuckGo search tool
│   └── doc_loader.py      # PDF/text loader + preprocessing
├── memory/
│   └── conversation.py    # SQLite persistent memory
├── vectorstore/
│   └── chroma_store.py    # ChromaDB setup + retriever
├── graph.py               # LangGraph workflow
├── app.py                 # Streamlit UI
└── requirements.txt
```

---

## 🛠️ Setup & Run

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed

### 1. Clone the repo
```bash
git clone https://github.com/your-username/research_agent.git
cd research_agent
```

### 2. Create conda environment
```bash
conda create -n research_agent python=3.11
conda activate research_agent
pip install -r requirements.txt
```

### 3. Pull Ollama models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements
```
langchain
langchain-community
langchain-ollama
langgraph
chromadb
sentence-transformers
rank-bm25
ddgs
PyPDF2
streamlit
numpy
torch
```

---

## 🗺️ Roadmap

- [x] Multi-agent LangGraph workflow
- [x] Hybrid RAG (BM25 + CrossEncoder + RRF)
- [x] Web Search Agent
- [x] Persistent SQLite Memory
- [x] Streamlit UI
- [ ] FastAPI backend
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Monitoring & logging

---

## 👨‍💻 Author

**Yogeshwar Prasad Lohiya**
[LinkedIn](#) | [GitHub](#)