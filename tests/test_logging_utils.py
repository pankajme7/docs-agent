import logging

from grounded_docs.logging_utils import log_payload


def test_log_payload_prints_strings_without_json_quoting(caplog):
    with caplog.at_level(logging.INFO, logger="grounded_docs.logging_utils"):
        log_payload(logging.getLogger("grounded_docs.logging_utils"), "LLM response", "Answer text")

    assert "LLM response:\nAnswer text" in caplog.text
    assert '"Answer text"' not in caplog.text


def test_log_payload_formats_structured_data_as_json(caplog):
    with caplog.at_level(logging.INFO, logger="grounded_docs.logging_utils"):
        log_payload(logging.getLogger("grounded_docs.logging_utils"), "RAG request", {"top_k": 4})

    assert 'RAG request:\n{\n  "top_k": 4\n}' in caplog.text
