import re


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = _normalise(text)
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # prefer to cut at whitespace
            cut = max(text.rfind("\n", start + size // 2, end),
                      text.rfind(" ", start + size // 2, end))
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(pages: list[tuple[int, str]], size: int, overlap: int) -> list[dict]:
    """Chunks never cross a page, so every chunk has an exact page number."""
    out, index = [], 0
    for page_number, text in pages:
        for piece in split_text(text, size, overlap):
            out.append({
                "chunk_index": index,
                "content": piece,
                "page_start": page_number,
                "page_end": page_number,
                "token_count": max(1, len(piece) // 4),  # rough estimate
            })
            index += 1
    return out