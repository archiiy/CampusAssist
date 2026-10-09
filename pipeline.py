"""
Phase 6 — Pipeline Graph.

Combines the Router, Desk (RAG), Reflection, and Governance agents into
one compiled LangGraph: query -> route -> retrieve+draft -> reflect ->
govern -> final answer.
"""

from langgraph.graph import StateGraph, END
from state import CampusAssistState

from router_agent import router_agent, route_from_router
from desk_agent import desk_agent
from reflection_agent import reflection_agent
from governance_agent import governance_agent


def build_pipeline_graph():
    graph = StateGraph(CampusAssistState)

    graph.add_node("router", router_agent)
    graph.add_node("desk", desk_agent)
    graph.add_node("reflection", reflection_agent)
    graph.add_node("governance", governance_agent)

    graph.set_entry_point("router")
    # route_from_router returns the desk name, but every desk shares the
    # same desk_agent function, so map all five labels to that one node
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {d: "desk" for d in ["academic", "hostel", "examcell", "placement", "fee"]},
    )
    graph.add_edge("desk", "reflection")
    graph.add_edge("reflection", "governance")
    graph.add_edge("governance", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_pipeline_graph()

    test_queries = [
        "When is the last date to pay hostel fees this semester?",
        "How many days do I have to apply for revaluation?",
        "What CGPA do I need to be eligible for placement drives?",
        "Can I bring a pet to the hostel?",   # not in the docs -> should escalate
        "Do you offer scholarships for sports achievements?",  # not in docs -> should escalate
    ]

    for q in test_queries:
        result = app.invoke({"raw_query": q})
        print(f"Query:      {q}")
        print(f"Desk:       {result['desk']}")
        print(f"Grounded:   {result['is_grounded']}  (confidence: {result['confidence']})")
        print(f"Escalated:  {result['escalated']}")
        print(f"Answer:     {result['final_answer']}")
        print()
