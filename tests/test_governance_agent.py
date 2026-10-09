import pytest

from campus_assistant.agents.governance_agent import governance_agent, ESCALATION_CONTACTS


@pytest.mark.parametrize("desk", list(ESCALATION_CONTACTS.keys()))
def test_grounded_high_confidence_passes_through_unescalated(desk):
    state = {"is_grounded": True, "confidence": "high", "desk": desk, "draft_answer": "The answer."}
    result = governance_agent(state)
    assert result["escalated"] is False
    assert result["final_answer"] == "The answer."


@pytest.mark.parametrize("is_grounded,confidence", [
    (False, "low"), (False, "high"), (True, "low"),
])
def test_anything_short_of_grounded_and_high_escalates(is_grounded, confidence):
    state = {"is_grounded": is_grounded, "confidence": confidence, "desk": "hostel", "draft_answer": "Maybe correct."}
    result = governance_agent(state)
    assert result["escalated"] is True
    assert result["escalation_contact"] == "Hostel Warden's Office"
    assert "Maybe correct." not in result["final_answer"]


def test_escalation_contact_matches_desk():
    for desk, contact in ESCALATION_CONTACTS.items():
        result = governance_agent({"is_grounded": False, "confidence": "low", "desk": desk})
        assert result["escalation_contact"] == contact


def test_unknown_desk_falls_back_to_generic_department():
    result = governance_agent({"is_grounded": False, "confidence": "low", "desk": "unknown-desk"})
    assert result["escalation_contact"] == "the relevant department"


def test_draft_answer_not_discarded_on_escalation():
    state = {"is_grounded": False, "confidence": "low", "desk": "fee", "draft_answer": "Possibly correct draft."}
    result = governance_agent(state)
    assert result["escalated"] is True
    # draft_answer isn't part of the returned patch, but the caller's
    # original state dict (merged by LangGraph) still has it available
    assert state["draft_answer"] == "Possibly correct draft."
