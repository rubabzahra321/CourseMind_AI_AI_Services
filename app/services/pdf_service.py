import fitz  # PyMuPDF
from app.core.errors import PermanentJobError


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    """Returns [(page_number, text)] with page numbers starting at 1."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PermanentJobError("The PDF could not be opened. It may be corrupted.") from exc
    with doc:
        if doc.needs_pass:
            raise PermanentJobError("The PDF is password-protected.")
        # Postgres text columns reject NUL characters, so strip them.
        return [(i + 1, page.get_text("text").replace("\x00", "").strip())
                for i, page in enumerate(doc)]