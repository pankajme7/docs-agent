from grounded_docs.metrics import calculate_cost, lexical_fact_coverage, retrieval_source_hit


def test_cost_calculation():
    assert calculate_cost(1_000_000, 500_000, 2.0, 8.0) == 6.0


def test_cost_is_unknown_when_prices_are_not_configured():
    assert calculate_cost(100, 20, None, 2.0) is None


def test_lexical_coverage_is_bounded():
    score = lexical_fact_coverage("A Service selector matches Pod labels", ["A Service selector matches labels on Pods"])
    assert score is not None and 0 <= score <= 1


def test_retrieval_source_hit():
    assert retrieval_source_hit(["a", "b"], ["b"]) is True
    assert retrieval_source_hit(["a"], ["b"]) is False
    assert retrieval_source_hit([], []) is None
