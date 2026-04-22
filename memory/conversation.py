from langchain_community.chat_message_histories import SQLChatMessageHistory


# ---------------- Config ----------------
DB_PATH = "sqlite:///memory/chat_history.db"


# ---------------- Session History ----------------
def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """SQLite se session history fetch karo."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_PATH
    )


def add_user_message(session_id: str, message: str):
    """User message save karo."""
    history = get_session_history(session_id)
    history.add_user_message(message)


def add_ai_message(session_id: str, message: str):
    """AI message save karo."""
    history = get_session_history(session_id)
    history.add_ai_message(message)


def get_chat_history(session_id: str) -> list:
    """Full chat history return karo."""
    history = get_session_history(session_id)
    return history.messages


def clear_history(session_id: str):
    """Session history clear karo."""
    history = get_session_history(session_id)
    history.clear()
    print(f"✅ History cleared for session: {session_id}")
