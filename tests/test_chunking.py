from campus_assistant.rag.chunking import chunk_text


def test_empty_string_produces_no_chunks():
    assert chunk_text("", source="x.txt") == []


def test_short_string_produces_one_chunk():
    chunks = chunk_text("Short text.", source="x.txt", chunk_size=500, chunk_overlap=80)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text."
    assert chunks[0]["source"] == "x.txt"


def test_long_string_produces_multiple_nonempty_chunks():
    text = "".join("Sentence number %d provides some filler content. " % i for i in range(60))
    chunks = chunk_text(text, source="long.txt", chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    for c in chunks:
        assert c["text"].strip() != ""
        assert c["source"] == "long.txt"


def test_overlap_actually_overlaps():
    text = "".join("Sentence number %d provides some filler content. " % i for i in range(60))
    chunks = chunk_text(text, source="long.txt", chunk_size=200, chunk_overlap=40)

    # with nonzero overlap, each chunk after the first should share at
    # least a short run of text with the chunk immediately before it
    for prev, curr in zip(chunks, chunks[1:]):
        prev_words = prev["text"].split()
        curr_words = curr["text"].split()
        shared = set(prev_words[-6:]) & set(curr_words[:6])
        assert shared, f"expected overlap between chunk tail {prev_words[-6:]} and next head {curr_words[:6]}"
