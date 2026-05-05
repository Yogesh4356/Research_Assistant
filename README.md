# 🤖 Personal Research Agent

A production-grade multi-agent AI assistant built with LangGraph, Groq, and RAG — featuring dynamic query planning, parallel/sequential execution, and smart routing.

## 🎯 What It Does

Ask anything — the agent automatically plans, decomposes your query into sub-queries, routes them to the right agents, and synthesizes a final answer.

**Examples:**
- *"Hi"* → Chitchat Agent
- *"What is LangGraph?"* → Web Search Agent
- *"What does the document say?"* → RAG Agent
- *"NIRF ranking of college in my resume?"* → RAG + Web Search (parallel)
- *"Tell me about elections"* → AskBack ("Which state? Which year?")

---

## 🏗️ Architecture

```
User Query
    ↓
[Planner Agent] — decompose query, detect dependencies
    ↓                          ↓
Clarification needed?      Sub-queries ready
    ↓                          ↓
[AskBack] → END         [Executor] — parallel/sequential
                               ↓
                         [Observer] — sufficient?
                          ↙              ↘
                    NO → re-plan      YES → [Synthesizer]
                    (cycle, max 3x)           ↓
                                           Final Answer
```

---

## 🤖 Agents

| Agent | Role |
|-------|------|
| Planner | Query decomposition, dependency detection, clarification |
| Executor | Runs RAG / Web Search / Chitchat agents |
| Observer | Checks if collected info is sufficient |
| Synthesizer | Combines results into final answer |
| RAG Agent | Hybrid retrieval from uploaded document |
| Web Search Agent | DuckDuckGo search + LLM summarization |
| Chitchat Agent | Casual conversation |

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Agent Orchestration | LangGraph |
| LLM (Reasoning) | Groq — llama-3.3-70b-versatile |
| Embeddings (Local) | Ollama — nomic-embed-text |
| Vector Database | ChromaDB |
| Hybrid Reranking | CrossEncoder + BM25 + RRF Fusion |
| Web Search | DuckDuckGo (ddgs) |
| Persistent Memory | SQLite |
| REST API | FastAPI |
| UI | Streamlit |
| Containerization | Docker |
| Orchestration | Kubernetes (Minikube) |

---

## 🚀 Features

- **Dynamic Query Planning** — Planner decomposes complex queries into sub-queries
- **Parallel/Sequential Execution** — Independent sub-queries run in parallel, dependent ones sequentially
- **AskBack Clarification** — Agent asks user for missing info before proceeding
- **Hybrid RAG Pipeline** — Query expansion + BM25 + CrossEncoder + RRF reranking
- **Smart Routing** — RAG, Web Search, or Chitchat based on query type
- **Persistent Memory** — SQLite conversation history across sessions
- **Zero Local LLM Cost** — Groq free tier for reasoning, Ollama for embeddings

---

## 📁 Project Structure

```
research_agent/
├── agents/
│   ├── planner.py         # Query decomposition + routing
│   ├── rag_agent.py       # Hybrid RAG pipeline
│   ├── search_agent.py    # Web search + summarization
│   ├── chitchat_agent.py  # Casual conversation
│   ├── observer.py        # Result sufficiency check
│   └── synthesizer.py     # Final answer generation
├── tools/
│   ├── web_search.py      # DuckDuckGo tool
│   └── doc_loader.py      # PDF/text loader
├── memory/
│   └── conversation.py    # SQLite persistent memory
├── vectorstore/
│   └── chroma_store.py    # ChromaDB setup
├── k8s/
│   ├── deployment.yaml    # Kubernetes deployment
│   └── service.yaml       # Kubernetes service
├── graph.py               # LangGraph cyclic workflow
├── api.py                 # FastAPI REST API
├── config.py              # LLM config (Groq)
├── app.py                 # Streamlit UI
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

---

## 🛠️ Setup & Run

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed
- Groq API key (free at [groq.com](https://groq.com))

### 1. Clone the repo
```bash
git clone https://github.com/Yogesh4356/Research_Assistant.git
cd Research_Assistant
```

### 2. Create conda environment
```bash
conda create -n research_agent python=3.11
conda activate research_agent
pip install -r requirements.txt
```

### 3. Pull Ollama models
```bash
ollama pull nomic-embed-text
```

### 4. Set environment variables
```bash
# .env file 
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run options

**Streamlit UI:**
```bash
streamlit run app.py
```

**FastAPI:**
```bash
python api.py
# http://localhost:8000/docs
```

**Docker:**
```bash
docker build -t research-agent .
docker run -p 8000:8000 --env-file .env research-agent
```

**Kubernetes (Minikube):**
```bash
minikube start
kubectl create secret generic groq-secret --from-literal=api-key=your_key
minikube image load research-agent:latest
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
minikube service research-agent-service --url
```

---

## 📦 Requirements

```
fastapi
uvicorn
langchain==0.3.27
langchain-community==0.3.27
langchain-ollama
langchain-core==0.3.72
langchain-groq
langgraph
chromadb
sentence-transformers
rank-bm25
ddgs
PyPDF2
streamlit
numpy
pydantic
sqlalchemy
python-multipart
python-dotenv
```

---

## 🗺️ Roadmap

- [x] Multi-agent LangGraph cyclic workflow
- [x] Dynamic query decomposition + dependency handling
- [x] Parallel/Sequential execution
- [x] AskBack clarification
- [x] Hybrid RAG (BM25 + CrossEncoder + RRF)
- [x] Web Search Agent
- [x] Chitchat Agent
- [x] Persistent SQLite Memory
- [x] Groq LLM integration (llama-3.3-70b)
- [x] FastAPI REST API
- [x] Docker containerization
- [x] Kubernetes deployment
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Monitoring & logging
- [ ] Cloud deployment (AWS/GCP)

---

## 👨‍💻 Author

**Yogeshwar Prasad Lohiya**

[LinkedIn](https://www.linkedin.com/in/yogeshwarlohiya) | [GitHub](https://github.com/Yogesh4356)