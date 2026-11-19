from functools import lru_cache
from supabase import create_client, Client
from app.core.config import settings


@lru_cache
def get_db() -> Client:
    # Service role bypasses RLS: ALWAYS filter by user_id and course_id.
    return create_client(settings.supabase_url, settings.supabase_service_role_key)