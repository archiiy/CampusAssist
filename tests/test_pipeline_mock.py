from campus_assistant.pipeline import build_pipeline_graph


def test_in_scope_query_is_answered_and_grounded():
    app = build_pipeline_graph(mock=True)
    result = app.invoke({"raw_query": "When is the last date to pay hostel fees this semester?"})
    assert result["desk"] == "hostel"
    assert result["final_answer"]
    assert result["escalated"] is False


def test_out_of_scope_query_escalates():
    app = build_pipeline_graph(mock=True)
    result = app.invoke({"raw_query": "Can I bring a pet to the hostel?"})
    assert result["escalated"] is True
    assert result["escalation_contact"]
