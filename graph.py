import asyncio
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.planner import run_planner
from agents.rag_agent import run_rag
from agents.search_agent import run_web_search
from agents.chitchat_agent import run_chitchat
from agents.observer import run_observer
from agents.synthesizer import run_synthesizer


# ---------------- State ----------------
class AgentState(TypedDict):
    query: str
    has_document: bool
    document_text: str
    collection_name: str
    session_id: str
    plan: Optional[dict]
    results: Optional[list]
    needs_clarification: bool
    clarification_question: Optional[str]
    is_sufficient: bool
    answer: Optional[str]
    source: Optional[str]
    iteration: int 


# ---------------- Node 1: Planner ----------------
def planner_node(state: AgentState) -> AgentState:
    print(f"🧠 Planning query: {state['query']}")

    plan = run_planner(
        query=state["query"],
        has_document=state["has_document"]
    )

    print(f"📋 Plan: {plan}")

    return {
        **state,
        "plan": plan,
        "needs_clarification": plan.get("needs_clarification", False),
        "clarification_question": plan.get("clarification_question", ""),
        "results": []
    }


# ---------------- Node 2: Executor ----------------
def executor_node(state: AgentState) -> AgentState:
    print(f"⚡ Executing sub-queries... (iteration {state['iteration']})")

    plan = state["plan"]
    sub_queries = plan.get("sub_queries", [])
    existing_results = state.get("results") or []
    new_results = []

    results_by_id = {r["id"]: r for r in existing_results if "id" in r}

    for sq in sub_queries:
        sq_id = sq["id"]
        sq_query = sq["query"]
        sq_agent = sq["agent"]
        depends_on = sq.get("depends_on", [])

        if sq_id in results_by_id:
            continue

        if depends_on:
            for dep_id in depends_on:
                if dep_id in results_by_id:
                    dep_answer = results_by_id[dep_id]["answer"]
                    sq_query = f"{sq_query} (context: {dep_answer})"

        print(f"  → Executing sub-query {sq_id} via {sq_agent}: {sq_query}")

        if sq_agent == "rag":
            result = run_rag(
                query=sq_query,
                text=state["document_text"],
                collection_name=state["collection_name"],
                session_id=state["session_id"]
            )
        elif sq_agent == "web_search":
            result = run_web_search(query=sq_query)
        elif sq_agent == "chitchat":
            result = run_chitchat(
                query=sq_query,
                session_id=state["session_id"]
            )
        else:
            result = run_web_search(query=sq_query)

        new_results.append({
            "id": sq_id,
            "query": sq_query,
            "agent": sq_agent,
            "answer": result["answer"]
        })
        results_by_id[sq_id] = new_results[-1]

    all_results = existing_results + new_results

    return {
        **state,
        "results": all_results,
        "iteration": state["iteration"] + 1 
    }


# ---------------- Node 3: Observer ----------------
def observer_node(state: AgentState) -> AgentState:
    print(f"👁️ Observing results...")

    observation = run_observer(
        original_query=state["query"],
        results=state["results"]
    )

    print(f"📊 Observation: {observation}")

    if not observation["is_sufficient"] and observation["additional_queries"]:
        updated_plan = {
            **state["plan"],
            "sub_queries": observation["additional_queries"]
        }
        return {
            **state,
            "is_sufficient": False,
            "plan": updated_plan
        }

    return {**state, "is_sufficient": True}


# ---------------- Node 4: Synthesizer ----------------
def synthesizer_node(state: AgentState) -> AgentState:
    print(f"✍️ Synthesizing final answer...")

    result = run_synthesizer(
        original_query=state["query"],
        results=state["results"],
        session_id=state["session_id"]
    )

    return {
        **state,
        "answer": result["answer"],
        "source": "synthesizer"
    }


# ---------------- Node 5: Ask Back ----------------
def askback_node(state: AgentState) -> AgentState:
    print(f"❓ Asking clarification...")
    return {
        **state,
        "answer": state["clarification_question"],
        "source": "askback"
    }


# ---------------- Routing Functions ----------------
def after_planner(state: AgentState) -> str:
    if state["needs_clarification"]:
        return "askback"
    return "executor"


def after_observer(state: AgentState) -> str:
    # Max 3 iterations — infinite loop prevent
    if state["iteration"] >= 3:
        return "synthesizer"
    if state["is_sufficient"]:
        return "synthesizer"
    return "executor"


# ---------------- Build Graph ----------------
def build_graph():
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("observer", observer_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("askback", askback_node)

    # Entry
    workflow.set_entry_point("planner")

    # Edges
    workflow.add_conditional_edges(
        "planner",
        after_planner,
        {
            "askback": "askback",
            "executor": "executor"
        }
    )

    workflow.add_edge("executor", "observer")

    workflow.add_conditional_edges(
        "observer",
        after_observer,
        {
            "executor": "executor", 
            "synthesizer": "synthesizer"
        }
    )

    workflow.add_edge("synthesizer", END)
    workflow.add_edge("askback", END)

    return workflow.compile()


# ---------------- Run ----------------
def run_graph(
    query: str,
    has_document: bool = False,
    document_text: str = "",
    collection_name: str = "default",
    session_id: str = "default"
) -> dict:

    graph = build_graph()

    initial_state = AgentState(
        query=query,
        has_document=has_document,
        document_text=document_text,
        collection_name=collection_name,
        session_id=session_id,
        plan=None,
        results=[],
        needs_clarification=False,
        clarification_question=None,
        is_sufficient=False,
        answer=None,
        source=None,
        iteration=0
    )

    return graph.invoke(
        initial_state,
        config={"recursion_limit": 10}
    )