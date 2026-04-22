from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

from graph import run_graph
from tools.doc_loader import load_and_preprocess
from memory.conversation import get_chat_history, clear_history


# ---------------- App ----------------
app = FastAPI(
    title="Research Agent API",
    description="Multi-agent RAG + Web Search powered by LangGraph & Ollama",
    version="1.0.0"
)

# ---------------- In-memory doc store ----------------
doc_store = {}


# ---------------- Schemas ----------------
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    collection_name: str = "default"


class ClearRequest(BaseModel):
    session_id: str


# ---------------- Routes ----------------

@app.get("/")
def root():
    return {"message": "Research Agent API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """PDF upload karo — text extract + store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    try:
        text = load_and_preprocess(file.file, file_type="pdf")
        collection_name = file.filename.replace(".pdf", "")
        doc_store[collection_name] = text

        return {
            "message": "Document uploaded successfully",
            "collection_name": collection_name,
            "chars": len(text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(request: QueryRequest):
    """Query karo — agent automatically route karega."""
    has_document = request.collection_name in doc_store
    document_text = doc_store.get(request.collection_name, "")

    try:
        result = run_graph(
            query=request.query,
            has_document=has_document,
            document_text=document_text,
            collection_name=request.collection_name,
            session_id=request.session_id
        )

        return {
            "answer": result["answer"],
            "source": result["source"],
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
def history(session_id: str):
    """Session ki chat history fetch karo."""
    messages = get_chat_history(session_id)
    return {
        "session_id": session_id,
        "messages": [
            {"role": m.type, "content": m.content}
            for m in messages
        ]
    }


@app.delete("/history")
def delete_history(request: ClearRequest):
    """Session history clear karo."""
    clear_history(request.session_id)
    return {"message": f"History cleared for session: {request.session_id}"}


# ---------------- Run ----------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)