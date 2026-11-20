
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

    # MCQ
    mcq_max_questions: int = 20
    mcq_context_max_chars: int = 18000

    # Phase 3: paper analytics
    min_papers_for_frequency: int = 3
    often_examined_frequency_percentage: float = 50.0

    # Phase 3: lecture coverage
    enough_material_chunk_count: int = 3
    material_similarity_threshold: float = 0.55

    # Phase 3: priority defaults
    priority_frequency_weight: float = 0.35
    priority_marks_weight: float = 0.25
    priority_syllabus_weight: float = 0.20
    priority_mastery_weight: float = 0.20

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )


settings = Settings()
