"""
Session Report.

Collects the full interaction into one structured object and writes it
to JSON.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from campus_assistant.config import settings


def build_report(state: dict, followup_qna: list) -> dict:
    return {
        "session_id": state.get("session_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": {
            "raw_query": state.get("raw_query"),
            "routed_desk": state.get("desk"),
            "router_fallback": state.get("router_fallback", False),
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


def save_report(report: dict, out_dir=None) -> str:
    out_dir = settings.reports_dir if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"session_report_{report['session_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return str(path)
