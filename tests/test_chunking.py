from app.services.chunking import build_chunks, split_text


def test_short_text_is_one_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_long_text_overlaps_and_terminates():
    text = "word " * 1000
    chunks = split_text(text, 200, 30)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_keep_page_numbers_and_global_index():
    out = build_chunks([(1, "a " * 50), (3, "b " * 50)], 40, 5)
    assert [c["chunk_index"] for c in out] == list(range(len(out)))
    assert {c["page_start"] for c in out} == {1, 3}