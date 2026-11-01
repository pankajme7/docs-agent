from grounded_docs.llm_client import parse_grounded_answer


def test_parse_valid_grounded_answer():
    result = parse_grounded_answer('{"answer":"Pod labels match the selector [S1]","sources":["S1"],"insufficient_context":false}')
    assert result.answer.startswith("Pod labels")
    assert result.sources == ["S1"]
    assert result.insufficient_context is False


def test_invalid_json_is_not_silently_treated_as_valid():
    result = parse_grounded_answer("not json")
    assert result.insufficient_context is True
    assert result.sources == []
