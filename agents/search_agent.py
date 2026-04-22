from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate  # ← change
from langchain_core.output_parsers import StrOutputParser  # ← change

from tools.web_search import search_and_format


# ---------------- LLM ----------------
llm = OllamaLLM(model="llama3.2")


# ---------------- Prompt ----------------
search_prompt = ChatPromptTemplate.from_template("""
You are a helpful research assistant.
Based on the following web search results, answer the user's question clearly and concisely.

Search Results:
{search_results}

User Question: {query}

Instructions:
- Answer only from the search results above
- If results don't contain the answer, say "I couldn't find relevant information"
- Be concise and to the point
- Mention sources where relevant
""")

search_chain = search_prompt | llm | StrOutputParser()


# ---------------- Main Search Function ----------------
def run_web_search(query: str, max_results: int = 5) -> dict:
    """
    Main function jo LangGraph node call karega.
    Returns: answer, raw results
    """
    # Search karo
    formatted_results, raw_results = search_and_format(query, max_results=max_results)

    # LLM se answer banao
    answer = search_chain.invoke({
        "query": query,
        "search_results": formatted_results
    })

    return {
        "answer": answer,
        "raw_results": raw_results,
        "source": "web_search"
    }