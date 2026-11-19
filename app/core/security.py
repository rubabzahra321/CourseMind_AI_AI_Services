import hmac
from fastapi import Header, HTTPException
from app.core.config import settings


def require_internal_key(x_internal_key: str = Header(default="")) -> None:
    expected = settings.ai_internal_key
    # A blank AI_INTERNAL_KEY (misconfiguration) must lock the service, not open it.
    if not expected or not hmac.compare_digest(
        x_internal_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
