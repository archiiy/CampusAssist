import json

from campus_assistant.report import build_report, save_report


def _fake_state():
    return {
        "session_id": "abc123",
        "raw_query": "When are hostel fees due?",
        "desk": "hostel",
        "router_fallback": False,
        "retrieved_chunks": [{"text": "...", "source": "hostel_circular_2026.txt", "score": 0.91}],
        "is_grounded": True,
        "confidence": "high",
        "reflection_notes": "matches context",
        "escalated": False,
        "escalation_contact": None,
        "final_answer": "Hostel fees are due by 30th September.",
    }


def test_build_report_shape():
    report = build_report(_fake_state(), followup_qna=[{"question": "q", "answer": "a"}])
    assert report["session_id"] == "abc123"
    assert report["query"]["raw_query"] == "When are hostel fees due?"
    assert report["query"]["routed_desk"] == "hostel"
    assert report["query"]["router_fallback"] is False
    assert report["retrieval"]["chunks_used"] == [{"source": "hostel_circular_2026.txt", "score": 0.91}]
    assert report["reflection"]["is_grounded"] is True
    assert report["governance"]["escalated"] is False
    assert report["final_answer"] == "Hostel fees are due by 30th September."
    assert report["followup_conversation"] == [{"question": "q", "answer": "a"}]


def test_save_report_round_trips_through_tmp_path(tmp_path):
    report = build_report(_fake_state(), followup_qna=[])
    path = save_report(report, out_dir=tmp_path)

    loaded = json.loads((tmp_path / f"session_report_abc123.json").read_text(encoding="utf-8"))
    assert loaded["session_id"] == "abc123"
    assert str(tmp_path) in path
