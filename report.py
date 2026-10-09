"""
Phase 8 — Session Report.

Same pattern as the trip concierge: collect the full interaction into
one structured object and write it to JSON.
"""

import json
import os
from datetime import datetime, timezone


def build_report(state: dict, followup_qna: list) -> dict:
    return {
        "session_id": state.get("session_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": {
            "raw_query": state.get("raw_query"),
            "routed_desk": state.get("desk"),
        },
        "retrieval": {
            "chunks_used": [
                {"source": c.get("source"), "score": c.get("score")}
                for c in state.get("retrieved_chunks", [])
            ],
        },
        "reflection": {
            "is_grounded": state.get("is_grounded"),
            "confidence": state.get("confidence"),
            "notes": state.get("reflection_notes"),
        },
        "governance": {
            "escalated": state.get("escalated"),
            "escalation_contact": state.get("escalation_contact"),
        },
        "final_answer": state.get("final_answer"),
        "followup_conversation": followup_qna,
    }


def save_report(report: dict, out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"session_report_{report['session_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
