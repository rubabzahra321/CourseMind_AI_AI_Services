
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    supabase_url: str
    supabase_service_role_key: str
    ai_internal_key: str

    storage_bucket: str = "course-documents"

    worker_poll_seconds: float = 3.0
    worker_stale_minutes: int = 10

    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150
    min_page_chars: int = 20

    # Gemini
    gemini_api_key: str
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dimensions: int = 768
    gemini_chat_model: str = "gemini-2.5-flash"

    # RAG
    rag_top_k: int = 6
    rag_min_similarity: float = 0.35

    # Evidence-strength thresholds
    evidence_partial_similarity: float = 0.35
    evidence_strong_similarity: float = 0.65

    # MCQ
    mcq_max_questions: int = 20
    mcq_context_max_chars: int = 18000

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
