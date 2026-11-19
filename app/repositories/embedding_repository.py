from app.db.client import get_db

BATCH_SIZE = 50


def save_embeddings(
    chunks: list[dict],
    embeddings: list[list[float]]
) -> None:

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")

    db = get_db()

    rows = []

    for chunk, vector in zip(chunks, embeddings):
        # pgvector accepts its textual vector representation.
        vector_text = "[" + ",".join(
            str(float(value)) for value in vector
        ) + "]"

        rows.append({
            **chunk,
            "embedding": vector_text,
        })

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        db.table("document_chunks").insert(batch).execute()