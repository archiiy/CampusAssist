"""
Phase 5 — Governance & Escalation Agent.

Pure rule-based logic (like your trip concierge's Recommendation Agent
thresholds) — NOT an LLM call. This is deliberate: the decision of
whether to trust an AI-generated answer shouldn't itself be made by
another unaccountable AI call. A simple, auditable rule is exactly
what "governance" means here.
"""

from state import CampusAssistState

ESCALATION_CONTACTS = {
    "academic": "Academic Office / Class Advisor",
    "hostel": "Hostel Warden's Office",
    "examcell": "Examination Cell",
    "placement": "Placement Cell",
    "fee": "Accounts Office",
}


def governance_agent(state: CampusAssistState) -> CampusAssistState:
    is_grounded = state.get("is_grounded", False)
    confidence = state.get("confidence", "low")
    desk = state.get("desk", "academic")

    if is_grounded and confidence == "high":
        return {
            "escalated": False,
            "final_answer": state.get("draft_answer", ""),
        }

    contact = ESCALATION_CONTACTS.get(desk, "the relevant department")
    return {
        "escalated": True,
        "escalation_contact": contact,
        "final_answer": (
            "I'm not fully confident this answer is accurate based on the "
            f"documents available to me. Please confirm with the {contact} "
            "to be sure."
        ),
    }


if __name__ == "__main__":
    test_cases = [
        {"is_grounded": True, "confidence": "high", "desk": "hostel", "draft_answer": "Hostel fees are due by 30th September."},
        {"is_grounded": False, "confidence": "low", "desk": "examcell", "draft_answer": "I don't have enough information."},
    ]
    for case in test_cases:
        result = governance_agent(case)
        print(f"escalated={result['escalated']}")
        print(f"  -> {result['final_answer']}\n")
