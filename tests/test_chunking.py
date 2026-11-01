from grounded_docs.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ") == []


def test_chunking_respects_maximum_word_window():
    text = "\n\n".join(" ".join([f"word{i}" for i in range(start, start + 40)]) for start in (0, 40, 80))
    chunks = chunk_text(text, chunk_words=50, overlap_words=10)
    assert len(chunks) >= 2
    assert all(len(chunk.split()) <= 50 for chunk in chunks)


def test_invalid_overlap_is_rejected():
    try:
        chunk_text("one two", chunk_words=10, overlap_words=10)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
