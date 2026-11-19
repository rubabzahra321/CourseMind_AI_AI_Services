from fastapi import APIRouter, Depends
from app.core.security import require_internal_key
from app.db.client import get_db

public = APIRouter()
internal = APIRouter(prefix="/internal", dependencies=[Depends(require_internal_key)])


@public.get("/health")
def health():
    return {"status": "ok"}


@internal.get("/db-check")
def db_check():
    get_db().table("courses").select("id").limit(1).execute()
    return {"database": "reachable"}