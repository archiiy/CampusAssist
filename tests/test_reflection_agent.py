from campus_assistant.agents.reflection_agent import parse_reflection_output


def test_well_formed_grounded_high():
    text = "GROUNDED: yes\nCONFIDENCE: high\nREASON: matches the context exactly"
    result = parse_reflection_output(text)
    assert result == {
        "is_grounded": True,
        "confidence": "high",
        "reflection_notes": "matches the context exactly",
    }


def test_well_formed_not_grounded_low():
    text = "GROUNDED: no\nCONFIDENCE: low\nREASON: context does not mention this"
    result = parse_reflection_output(text)
    assert result["is_grounded"] is False
    assert result["confidence"] == "low"


def test_case_and_whitespace_variations_still_parse():
    text = "  grounded:   YES  \nconfidence:HIGH\nreason:   fine  "
    result = parse_reflection_output(text)
    assert result["is_grounded"] is True
    assert result["confidence"] == "high"


def test_missing_grounded_line_is_flagged_as_parse_failure_not_silent_negative():
    text = "CONFIDENCE: high\nREASON: the model forgot the GROUNDED line"
    result = parse_reflection_output(text)
    assert result["is_grounded"] is False
    assert result["confidence"] == "low"
    assert "PARSE_FAILURE" in result["reflection_notes"]


def test_completely_malformed_output_is_flagged_as_parse_failure():
    text = "I think this answer looks fine to me."
    result = parse_reflection_output(text)
    assert "PARSE_FAILURE" in result["reflection_notes"]
