"""
Phase 4 — Reflection Agent.

This is the piece your trip concierge never had: a second LLM pass that
checks the desk agent's own answer against the retrieved context before
it's shown to anyone. This is the "self-improving agent" pattern
(Reflection) your course covers — the system critiques its own output
instead of trusting it blindly.
"""

from state import CampusAssistState
from llm import get_llm

REFLECTION_PROMPT = """You are checking whether an answer is fully supported by the context it was
based on. Do not use outside knowledge — judge strictly by whether the
context actually contains what the answer claims.

ANSWER:
{answer}

CONTEXT:
{context}

Respond in exactly this format:
GROUNDED: yes|no
CONFIDENCE: high|low
REASON: <one short sentence>"""


def reflection_agent(state: CampusAssistState) -> CampusAssistState:
    answer = state.get("draft_answer", "")
    context = "\n\n".join(c["text"] for c in state.get("retrieved_chunks", []))

    llm = get_llm()
    prompt = REFLECTION_PROMPT.format(answer=answer, context=context)
    raw = llm.invoke(prompt)
    text = (raw.content if hasattr(raw, "content") else str(raw)).strip()

    grounded = "grounded: yes" in text.lower()
    confidence = "high" if "confidence: high" in text.lower() else "low"
    reason_line = next((l for l in text.split("\n") if l.lower().startswith("reason:")), "REASON: unspecified")
    reason = reason_line.split(":", 1)[1].strip() if ":" in reason_line else reason_line

    return {
        "is_grounded": grounded,
        "confidence": confidence,
        "reflection_notes": reason,
    }


if __name__ == "__main__":
    test_cases = [
        {
            "draft_answer": "Based on the available records: Hostel fees for the odd semester must be paid on or before 30th September.",
            "retrieved_chunks": [{"text": "Hostel fees for the odd semester must be paid on or before 30th September 2026."}],
        },
        {
            "draft_answer": "I don't have enough information in the available records to answer this confidently.",
            "retrieved_chunks": [{"text": "Mess Timings: Breakfast 7:30-9:00 AM."}],
        },
    ]
    for case in test_cases:
        result = reflection_agent(case)
        print(f"Answer:  {case['draft_answer']}")
        print(f"  -> grounded={result['is_grounded']} confidence={result['confidence']} reason={result['reflection_notes']}\n")
