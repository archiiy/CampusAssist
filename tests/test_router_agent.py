from campus_assistant.agents.router_agent import router_agent, route_from_router, VALID_DESKS


class _StubLLM:
    def __init__(self, response_text):
        self.response_text = response_text

    def invoke(self, prompt):
        class _Msg:
            content = self.response_text
        return _Msg()


def test_valid_classification_passes_through(monkeypatch):
    monkeypatch.setattr(
        "campus_assistant.agents.router_agent.get_llm",
        lambda mock=None: _StubLLM("hostel"),
    )
    result = router_agent({"raw_query": "when is hostel fee due"})
    assert result["desk"] == "hostel"
    assert result["router_fallback"] is False


def test_invalid_classification_falls_back_to_academic_and_flags_it(monkeypatch, caplog):
    monkeypatch.setattr(
        "campus_assistant.agents.router_agent.get_llm",
        lambda mock=None: _StubLLM("not-a-real-desk"),
    )
    result = router_agent({"raw_query": "some ambiguous query"})
    assert result["desk"] == "academic"
    assert result["router_fallback"] is True


def test_route_from_router_reads_desk_field():
    assert route_from_router({"desk": "fee"}) == "fee"
    assert route_from_router({}) == "academic"


def test_valid_desks_constant_matches_expected_set():
    assert VALID_DESKS == {"academic", "hostel", "examcell", "placement", "fee"}
