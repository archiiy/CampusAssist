"""
Reflection Agent.

A second LLM pass that checks the desk agent's own answer against the
retrieved context before it's shown to anyone — the system critiques
its own output instead of trusting it blindly.
"""

import logging
import re

from campus_assistant.state import CampusAssistState
from campus_assistant.llm.factory import get_llm
from campus_assistant.prompts import REFLECTION_PROMPT

logger = logging.getLogger(__name__)


def parse_reflection_output(text: str) -> dict:
    """Pure parsing function, independently testable without an LLM call.

    Looks for GROUNDED:/CONFIDENCE:/REASON: lines independently via regex
    rather than one brittle exact-substring check. When a field is missing
    or doesn't match the expected format at all (as opposed to genuinely
    saying "no"/"low"), that is flagged as a parse failure rather than
    silently resolving to the same "not grounded" result a real negative
    verdict would produce — the two cases are not the same and shouldn't
    look identical in logs/reports.
    """
    grounded_match = re.search(r"GROUNDED:\s*(yes|no)", text, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(high|low)", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)

    parse_failed = grounded_match is None or confidence_match is None

    if parse_failed:
        logger.warning("reflection output did not match expected format: %r", text[:200])
        return {
            "is_grounded": False,
            "confidence": "low",
            "reflection_notes": f"PARSE_FAILURE: could not parse reflection output: {text[:100]!r}",
        }

    return {
        "is_grounded": grounded_match.group(1).lower() == "yes",
        "confidence": confidence_match.group(1).lower(),
        "reflection_notes": reason_match.group(1).strip() if reason_match else "unspecified",
    }


def reflection_agent(state: CampusAssistState, mock: bool | None = None) -> CampusAssistState:
    answer = state.get("draft_answer", "")
    context = "\n\n".join(c["text"] for c in state.get("retrieved_chunks", []))

    llm = get_llm(mock=mock)
    prompt = REFLECTION_PROMPT.format(answer=answer, context=context)
    raw = llm.invoke(prompt)
    text = (raw.content if hasattr(raw, "content") else str(raw)).strip()

    return parse_reflection_output(text)


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
