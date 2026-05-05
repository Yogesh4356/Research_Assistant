from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from memory.conversation import add_user_message, add_ai_message


from config import get_llm
llm = get_llm()


chitchat_prompt = ChatPromptTemplate.from_template("""
You are a friendly and helpful AI assistant.
Respond naturally and conversationally.

Conversation History:
{history}

User: {query}
Assistant:
""")

chitchat_chain = chitchat_prompt | llm | StrOutputParser()


def run_chitchat(query: str, session_id: str = "default") -> dict:
    from memory.conversation import get_chat_history

    messages = get_chat_history(session_id)
    history = "\n".join([
        f"{m.type.upper()}: {m.content}"
        for m in messages[-6:] 
    ])

    answer = chitchat_chain.invoke({
        "query": query,
        "history": history
    })

    add_user_message(session_id, query)
    add_ai_message(session_id, answer)

    return {
        "answer": answer,
        "source": "chitchat"
    }