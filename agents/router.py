from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate  # ← change
from langchain_core.output_parsers import StrOutputParser


# ---------------- LLM ----------------
llm = OllamaLLM(model="llama3.2")


# ---------------- Router Prompt ----------------
router_prompt = ChatPromptTemplate.from_template("""
You are a query router. Your job is to decide where to route the user's question.

You have two options:
1. "rag" - Use this when:
   - User asks about an uploaded document
   - Question is about specific file content
   - User says "in the document", "from the pdf", "based on the file"

2. "web_search" - Use this when:
   - User asks about current events or latest news
   - Question needs real-time or recent information
   - Question is general knowledge not related to any document
   - User asks "what is", "how does", "latest", "recent"

User Question: {query}
Document uploaded: {has_document}

Respond with ONLY one word: either "rag" or "web_search"
""")

router_chain = router_prompt | llm | StrOutputParser()


# ---------------- Router Function ----------------
def route_query(query: str, has_document: bool = False) -> str:
    """
    Query ko route karo — 'rag' ya 'web_search' return karega.
    """
    result = router_chain.invoke({
        "query": query,
        "has_document": "Yes" if has_document else "No"
    })

    # Clean output — sirf rag ya web_search chahiye
    result = result.strip().lower()

    # Safety check — agar LLM kuch aur bol de
    if "rag" in result:
        return "rag"
    elif "web" in result or "search" in result:
        return "web_search"
    else:
        # Default — agar document hai toh rag, warna web
        return "rag" if has_document else "web_search"