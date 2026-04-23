# 🤖 Personal Research Agent

A production-grade multi-agent AI assistant built with **LangGraph**, **Ollama**, and **hybrid RAG** — runs fully locally with **zero API costs**. Includes Streamlit UI, FastAPI backend, Docker containerization, and Kubernetes deployment ready.

## 🎯 What It Does

Ask anything — the agent automatically decides whether to search the web or query your uploaded document, then returns a well-reasoned answer with sources. Full conversation history persists across sessions.

**Examples:**
- *"What is LangGraph?"* → Web Search Agent (fetches latest info from DuckDuckGo)
- *"What does the document say about embeddings?"* → RAG Agent (queries uploaded PDF)
- *"Compare this with the latest research"* → Hybrid approach (RAG + Web Search)

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   Multi-Interface Support                │
│           Streamlit UI  +  FastAPI Backend               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
         ┌───────────────────────┐
         │  Router Agent         │
         │  (Decides: RAG/Web?)  │
         └───────────┬───────────┘
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
    ┌──────────────┐     ┌──────────────┐
    │  RAG Agent   │     │Web Search    │
    ├──────────────┤     ├──────────────┤
    │• Query Exp.  │     │• DuckDuckGo  │
    │• BM25 Search │     │• DuckDuckGo  │
    │• CrossEnc.   │     │  Search API  │
    │• RRF Rerank  │     │• Summarize   │
    │• ChromaDB    │     │• Attribute   │
    └──────┬───────┘     └──────┬───────┘
           │                     │
           └──────────┬──────────┘
                      ↓
         ┌─────────────────────────┐
         │  Final Answer + Sources │
         └────────────┬────────────┘
                      ↓
         ┌─────────────────────────┐
         │  SQLite Memory (Persist)│
         └─────────────────────────┘
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM Orchestration** | LangGraph (agent workflow) |
| **LLM (Local)** | Ollama — llama3.2 |
| **Embeddings** | Ollama — nomic-embed-text |
| **Vector Database** | ChromaDB |
| **Hybrid RAG** | BM25 + CrossEncoder + RRF Fusion |
| **Web Search** | DuckDuckGo (ddgs) |
| **Frontend UI** | Streamlit |
| **Backend API** | FastAPI + Uvicorn |
| **Session Memory** | SQLite + SQLAlchemy |
| **Document Processing** | PyPDF2 + NLP preprocessing |
| **ML Inference** | PyTorch (CUDA auto-detection) |
| **Containerization** | Docker + Docker Compose |
| **Orchestration** | Kubernetes (k8s) |
| **Cost** | $0 — 100% local, offline-capable |

---

## 🚀 Features

✅ **Smart Query Routing** — LLM-based router auto-selects RAG or Web Search based on intent  
✅ **Hybrid RAG Pipeline** — Query expansion + BM25 + CrossEncoder reranking + RRF fusion  
✅ **Persistent Conversations** — SQLite stores full chat history across sessions  
✅ **Local-First Privacy** — No cloud APIs, no data leaks — everything runs on your machine  
✅ **Web Search Integration** — Real-time answers via DuckDuckGo when needed  
✅ **GPU Acceleration** — Auto-detects CUDA for CrossEncoder reranking (CPU fallback available)  
✅ **Dual Interface** — Streamlit web UI + FastAPI REST API  
✅ **Document Upload** — PDF upload with automatic text extraction & chunking  
✅ **Production-Ready** — Dockerized, Kubernetes-deployable, fully configurable  

---

## 📁 Project Structure

```
research_agent/
├── agents/
│   ├── router.py              # LLM-based query router (RAG vs Web Search)
│   ├── rag_agent.py           # Hybrid RAG with query expansion + BM25 + CrossEncoder
│   └── search_agent.py        # Web search with DuckDuckGo + LLM summarization
├── tools/
│   ├── web_search.py          # DuckDuckGo search wrapper + formatting
│   └── doc_loader.py          # PDF/text loader, tokenization, preprocessing
├── memory/
│   └── conversation.py        # SQLite persistence layer (chat history)
├── vectorstore/
│   └── chroma_store.py        # ChromaDB initialization + retriever setup
├── k8s/
│   ├── deployment.yaml        # K8s Deployment manifest
│   └── service.yaml           # K8s Service manifest
├── graph.py                   # LangGraph state machine & workflow
├── app.py                     # Streamlit UI (main entry point)
├── api.py                     # FastAPI backend with /upload, /query, /history routes
├── Dockerfile                 # Docker image for API service
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (Ollama host, etc.)
└── Readme.md                  # This file
```

---

## 🛠️ Setup & Run

### Prerequisites
- **Python 3.11+**
- **Ollama** installed and running ([Download here](https://ollama.com))
  - Models required: `llama3.2` (LLM) + `nomic-embed-text` (embeddings)

### Quick Start (Streamlit UI)

```bash
# 1. Clone repo
git clone https://github.com/your-username/research_agent.git
cd research_agent

# 2. Create Python environment
conda create -n research_agent python=3.11
conda activate research_agent

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull Ollama models (first time only)
ollama pull llama3.2
ollama pull nomic-embed-text

# 5. Start Ollama service (in another terminal)
ollama serve

# 6. Run Streamlit app
streamlit run app.py
```

The Streamlit UI will open at `http://localhost:8501`

### FastAPI Backend Only

```bash
# With Ollama running (see step 5 above):
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Docker Deployment

```bash
# Build image
docker build -t research-agent:latest .

# Run with Ollama (must be accessible)
docker run -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  research-agent:latest
```

### Kubernetes Deployment

```bash
# Apply k8s manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods -l app=research-agent
kubectl port-forward svc/research-agent-service 8000:80
```

---

## 📦 Dependencies

All packages listed in [requirements.txt](requirements.txt):

- `langchain==0.3.27` — LLM framework
- `langchain-community==0.3.27` — Community extensions
- `langchain-ollama` — Ollama integration
- `langgraph` — Agent orchestration
- `chromadb` — Vector database
- `sentence-transformers` — CrossEncoder for reranking
- `rank-bm25` — BM25 retrieval
- `ddgs` — DuckDuckGo search
- `PyPDF2` — PDF extraction
- `streamlit` — Web UI
- `fastapi` + `uvicorn` — REST API
- `pydantic` + `sqlalchemy` — Data validation & ORM
- `torch` + `numpy` — ML compute



---

## � API Endpoints (FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check / welcome message |
| `GET` | `/health` | API status |
| `POST` | `/upload` | Upload PDF document |
| `POST` | `/query` | Send query (RAG or Web Search) |
| `GET` | `/history/{session_id}` | Retrieve chat history |
| `DELETE` | `/history/{session_id}` | Clear chat history |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

**Example Query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LangGraph?",
    "session_id": "user_123",
    "collection_name": "my_document"
  }'
```

---

## 🧠 How It Works

### 1️⃣ Query Input
User submits query via Streamlit UI or FastAPI endpoint.

### 2️⃣ Router Agent
- Decides: Should we use RAG (document) or Web Search?
- Decision logic: Is there a document? Does the query mention it?
- Routes to appropriate agent.

### 3️⃣ RAG Agent (If document uploaded)
- **Query Expansion**: Generate 3 alternative phrasings of the question
- **Multi-Query Retrieval**: Retrieve docs using all 4 queries (original + expansions)
- **BM25 Ranking**: Score documents by keyword relevance
- **CrossEncoder Reranking**: Fine-grained relevance scoring (GPU accelerated)
- **RRF Fusion**: Combine BM25 + CrossEncoder scores for final ranking
- **LLM Synthesis**: Generate answer from top-k documents

### 4️⃣ Web Search Agent (If no document or for general knowledge)
- **DuckDuckGo Search**: Fetch top search results
- **Result Formatting**: Parse & structure search results
- **LLM Summarization**: Generate concise answer with source attribution

### 5️⃣ Memory Persistence
- All Q&A pairs saved to SQLite database
- Indexed by session_id for multi-user support
- Retrievable for context in follow-up queries

---

## ⚙️ Configuration

Create `.env` file in project root:
```env
# Ollama connection
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# RAG settings
MAX_DOCS=10
CHUNK_SIZE=1024
CHUNK_OVERLAP=100

# Web search
MAX_SEARCH_RESULTS=5

# SQLite
DATABASE_URL=sqlite:///memory/chat_history.db
```

---

## 🗺️ Roadmap

| Status | Feature | Details |
|--------|---------|---------|
| ✅ | Multi-agent LangGraph | Router, RAG, Web Search agents |
| ✅ | Hybrid RAG | BM25 + CrossEncoder + RRF |
| ✅ | Web Search | DuckDuckGo integration |
| ✅ | Persistent Memory | SQLite conversation history |
| ✅ | Streamlit UI | Full-featured web interface |
| ✅ | FastAPI Backend | Production REST API |
| ✅ | Docker | Containerized deployment |
| ✅ | Kubernetes | k8s manifests included |
| 🔄 | Metrics & Monitoring | Prometheus/Grafana integration |
| 🔄 | Advanced Reranking | Cohere Rerank API (optional) |
| 🔄 | Multi-Document Support | Handle multiple documents in RAG |
| 🔄 | Long-Context LLMs | Support for context-window > 4K |
| 🔄 | Chat Memory Context | Use conversation history in RAG |
| 🔄 | Admin Dashboard | User management, analytics |

---

## � License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Yogeshwar Prasad Lohiya**

Built with ❤️ using LangGraph, Ollama, and ChromaDB

---

## ⭐ Support

If you find this project useful, please consider giving it a star on GitHub!

For issues, questions, or feature requests, open an [issue on GitHub](https://github.com/your-username/research_agent/issues).

---

## 📖 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Models](https://ollama.ai/library)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)