import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------- Embeddings ----------------
def get_embeddings():
    """Nomic embeddings via Ollama."""
    return OllamaEmbeddings(model="nomic-embed-text")


# ---------------- Text Splitter ----------------
def get_text_splitter(chunk_size: int = 500, chunk_overlap: int = 50):
    """Recursive text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


# ---------------- Build or Load Vectorstore ----------------
def build_or_load_vectorstore(
    text: str,
    collection_name: str,
    persist_dir: str = "db"
) -> Chroma:
    """
    If the collection already exists, load it,
    otherwise create a new one.
    """
    embeddings = get_embeddings()
    collection_path = os.path.join(persist_dir, collection_name)

    if os.path.exists(collection_path):
        print(f"✅ Loading existing collection: {collection_name}")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
    else:
        print(f"🆕 Creating new collection: {collection_name}")
        splitter = get_text_splitter()
        chunks = splitter.split_text(text)

        vectorstore = Chroma.from_texts(
            chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        print(f"✅ Created collection with {len(chunks)} chunks")

    return vectorstore


# ---------------- Get Retriever ----------------
def get_retriever(vectorstore: Chroma, k: int = 5):
    """Create a retriever from the vectorstore."""
    return vectorstore.as_retriever(search_kwargs={"k": k})