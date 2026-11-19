from app.services.chunking import build_chunks, split_text


def test_empty_and_whitespace_text_gives_no_chunks():
    assert split_text("", 100, 10) == []
    assert split_text("   \n\n  ", 100, 10) == []
    assert build_chunks([(1, "")], 100, 10) == []


def test_chunks_never_cross_pages_and_are_within_size():
    pages = [(1, "alpha " * 400), (2, "beta " * 400)]
    chunks = build_chunks(pages, 300, 40)
    assert all(len(c["content"]) <= 300 for c in chunks)
    assert all(c["page_start"] == c["page_end"] for c in chunks)
    assert "beta" not in "".join(c["content"] for c in chunks if c["page_start"] == 1)


def test_overlap_repeats_context_between_chunks():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = split_text(text, 100, 30)
    assert len(chunks) > 2
    assert set(chunks[0].split()[-2:]) & set(chunks[1].split())


def test_text_without_spaces_still_terminates():
    chunks = split_text("x" * 5000, 1200, 150)
    assert len(chunks) >= 4


def test_token_count_is_positive():
    assert all(c["token_count"] >= 1 for c in build_chunks([(1, "hello world")], 100, 10))
