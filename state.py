"""
Phase 1 — Shared state schema.

Same philosophy as your trip concierge: one TypedDict threaded through
every node (router, desk agents, reflection, governance, follow-up,
report) instead of each stage inventing its own shape.
"""

from typing import TypedDict, List, Dict, Any, Optional


class CampusAssistState(TypedDict, total=False):
    # --- Intake & routing ---
    raw_query: str                # the student's free-text question
    desk: str                     # "academic" | "hostel" | "examcell" | "placement" | "fee"
    session_id: str               # fixed thread id for the checkpointer

    # --- RAG retrieval + draft answer ---
    retrieved_chunks: List[Dict[str, Any]]   # [{"text": ..., "source": ..., "score": ...}, ...]
    draft_answer: str             # the desk agent's first-pass answer

    # --- Reflection ---
    is_grounded: bool             # did the reflection check pass?
    reflection_notes: str         # why it passed/failed
    confidence: str               # "high" | "low"

    # --- Governance & escalation ---
    escalated: bool
    final_answer: str             # what actually gets shown to the student
    escalation_contact: str       # which office/desk to contact if escalated

    # --- Follow-up conversation ---
    conversation: List[Dict[str, str]]

    # --- Report ---
    report: Dict[str, Any]
