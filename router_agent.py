"""
Phase 3 — Router Agent.

Same pattern as the trip concierge's classifier: an LLM node reads the
free-text query and picks one of five desks. A conditional edge then
sends execution to that desk's RAG agent.
"""

from state import CampusAssistState
from llm import get_llm

VALID_DESKS = {"academic", "hostel", "examcell", "placement", "fee"}

ROUTER_PROMPT = """Classify the student query into exactly one department.
Departments: academic, hostel, examcell, placement, fee.

- academic: syllabus, attendance, courses, credits, timetable, faculty advisors
- hostel: room allotment, mess, hostel fees, warden issues
- examcell: hall tickets, exams, revaluation, arrears, CGPA
- placement: internships, campus drives, resumes, eligibility for placements
- fee: tuition fee payment, scholarships, refunds

Respond with ONLY the single department word — nothing else.

Student query: "{query}"

Department:"""


def router_agent(state: CampusAssistState) -> CampusAssistState:
    llm = get_llm()
    prompt = ROUTER_PROMPT.format(query=state["raw_query"])
    raw = llm.invoke(prompt)
    label = (raw.content if hasattr(raw, "content") else str(raw)).strip().lower()

    if label not in VALID_DESKS:
        label = "academic"   # safe fallback

    return {"desk": label}


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
