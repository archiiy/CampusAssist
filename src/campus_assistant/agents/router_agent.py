"""
Router Agent.

An LLM node reads the free-text query and picks one of five desks. A
conditional edge then sends execution to that desk's RAG agent.
"""

import logging

from campus_assistant.state import CampusAssistState
from campus_assistant.llm.factory import get_llm
from campus_assistant.prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)

VALID_DESKS = {"academic", "hostel", "examcell", "placement", "fee"}


def router_agent(state: CampusAssistState, mock: bool | None = None) -> CampusAssistState:
    llm = get_llm(mock=mock)
    prompt = ROUTER_PROMPT.format(query=state["raw_query"])
    raw = llm.invoke(prompt)
    label = (raw.content if hasattr(raw, "content") else str(raw)).strip().lower()

    if label not in VALID_DESKS:
        logger.warning(
            "router returned invalid label %r for query %r — falling back to academic",
            label, state["raw_query"],
        )
        return {"desk": "academic", "router_fallback": True}

    return {"desk": label, "router_fallback": False}


def route_from_router(state: CampusAssistState) -> str:
    return state.get("desk", "academic")


if __name__ == "__main__":
    test_queries = [
        "When is the last date to pay hostel fees this semester?",
        "How do I apply for revaluation of my answer script?",
        "What CGPA do I need to be eligible for placement drives?",
        "Can I get a refund if I withdraw from the course?",
        "What is the minimum attendance required for exams?",
    ]
    for q in test_queries:
        result = router_agent({"raw_query": q})
        print(f"[{result['desk']:10s}] {q}")
