from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from memory.conversation import add_user_message, add_ai_message

# ---------------- LLM ----------------
from config import get_llm
llm = get_llm()

synthesizer_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the user's question using the collected information.

Original User Query: {original_query}

Collected Information:
{results}

Instructions:
- Give a direct, clear answer
- Use information exactly as provided — do not say "Source 1" or "Source 2"
- Do not mention agent names like "rag" or "web_search"
- If answer is from document, just say the answer directly
- Be concise
""")

synthesizer_chain = synthesizer_prompt | llm | StrOutputParser()


def run_synthesizer(
    original_query: str,
    results: list[dict],
    session_id: str = "default"
) -> dict:

    # Results readable format mein
    results_text = ""
    for i, r in enumerate(results, 1):
        agent = r.get("agent", "unknown")
        answer = r.get("answer", "")
        results_text += f"\nSource {i} (from {agent}):\n{answer}\n"

    final_answer = synthesizer_chain.invoke({
        "original_query": original_query,
        "results": results_text
    })

    # Memory mein save karo
    add_user_message(session_id, original_query)
    add_ai_message(session_id, final_answer)

    return {
        "answer": final_answer,
        "source": "synthesizer"
    }