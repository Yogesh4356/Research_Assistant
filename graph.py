from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.router import route_query
from agents.rag_agent import run_rag
from agents.search_agent import run_web_search
from memory.conversation import add_user_message, add_ai_message


# ---------------- State Definition ----------------
class AgentState(TypedDict):
    query: str
    has_document: bool
    document_text: str
    collection_name: str
    session_id: str
    route: Optional[str]
    answer: Optional[str]
    expansions: Optional[list]
    top_docs: Optional[list]
    raw_results: Optional[list]
    source: Optional[str]


# ---------------- Node 1: Router ----------------
def router_node(state: AgentState) -> AgentState:
    """Query ko route karo — rag ya web_search."""
    print(f"🧭 Routing query: {state['query']}")

    route = route_query(
        query=state["query"],
        has_document=state["has_document"]
    )

    print(f"➡️ Route decided: {route}")
    return {**state, "route": route}


# ---------------- Node 2: RAG Node ----------------
def rag_node(state: AgentState) -> AgentState:
    """RAG agent se answer lo."""
    print(f"📄 Running RAG agent...")

    result = run_rag(
        query=state["query"],
        text=state["document_text"],
        collection_name=state["collection_name"],
        session_id=state["session_id"]
    )

    # Memory mein save karo
    add_user_message(state["session_id"], state["query"])
    add_ai_message(state["session_id"], result["answer"])

    return {
        **state,
        "answer": result["answer"],
        "expansions": result["expansions"],
        "top_docs": result["top_docs"],
        "source": "rag"
    }


# ---------------- Node 3: Web Search Node ----------------
def web_search_node(state: AgentState) -> AgentState:
    """Web search agent se answer lo."""
    print(f"🔍 Running Web Search agent...")

    result = run_web_search(query=state["query"])

    # Memory mein save karo
    add_user_message(state["session_id"], state["query"])
    add_ai_message(state["session_id"], result["answer"])

    return {
        **state,
        "answer": result["answer"],
        "raw_results": result["raw_results"],
        "source": "web_search"
    }


# ---------------- Routing Logic ----------------
def decide_route(state: AgentState) -> str:
    """Router ka decision LangGraph ko batao."""
    return state["route"]


# ---------------- Build Graph ----------------
def build_graph() -> StateGraph:
    """LangGraph workflow banao."""

    workflow = StateGraph(AgentState)

    # Nodes add karo
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("web_search", web_search_node)

    # Entry point
    workflow.set_entry_point("router")

    # Conditional edges — router decide karega
    workflow.add_conditional_edges(
        "router",
        decide_route,
        {
            "rag": "rag",
            "web_search": "web_search"
        }
    )

    # Dono nodes END pe jaate hain
    workflow.add_edge("rag", END)
    workflow.add_edge("web_search", END)

    return workflow.compile()


# ---------------- Run Graph ----------------
def run_graph(
    query: str,
    has_document: bool = False,
    document_text: str = "",
    collection_name: str = "default",
    session_id: str = "default"
) -> dict:
    """
    Main function — Streamlit yahi call karega.
    """
    graph = build_graph()

    initial_state = AgentState(
        query=query,
        has_document=has_document,
        document_text=document_text,
        collection_name=collection_name,
        session_id=session_id,
        route=None,
        answer=None,
        expansions=None,
        top_docs=None,
        raw_results=None,
        source=None
    )

    result = graph.invoke(initial_state)
    return result