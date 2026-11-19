import hmac
from fastapi import Header, HTTPException
from app.core.config import settings


def require_internal_key(x_internal_key: str = Header(default="")) -> None:
    if not hmac.compare_digest(x_internal_key, settings.ai_internal_key):
        raise HTTPException(status_code=401, detail="Unauthorized")