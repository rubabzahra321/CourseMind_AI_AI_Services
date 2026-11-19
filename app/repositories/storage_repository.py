from app.core.config import settings
from app.db.client import get_db


def download(path: str) -> bytes:
    return get_db().storage.from_(settings.storage_bucket).download(path)