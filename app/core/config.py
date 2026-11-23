
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


    # Analytics - Priority Weights
    priority_frequency_weight: float = 0.40
    priority_marks_weight: float = 0.25
    priority_syllabus_weight: float = 0.20
    priority_mastery_weight: float = 0.15

    # Analytics - Data Sufficiency
    min_papers_for_frequency: int = 2

    # Analytics - Material Mapping
    material_similarity_threshold: float = 0.55
    enough_material_chunk_count: int = 3

    # Analytics - Gap Detection
    often_examined_frequency_percentage: float = 50.0

    # Machine learning - topic classifier
    ml_random_seed: int = 42                    # repeatable results
    ml_cv_max_folds: int = 5                    # cross-validation folds
    ml_min_examples_per_topic: int = 2          # topics below this are left out of training
    ml_min_topics: int = 2                      # need at least this many topics to train
    ml_min_question_samples_for_selection: int = 10   # below this, pick the model on all samples
    ml_min_confidence: float = 0.50             # below this a suggestion needs review
    ml_max_classify_questions: int = 200

    # Machine learning - difficulty model
    ml_difficulty_min_samples: int = 20         # questions needed to train
    ml_difficulty_min_per_class: int = 3        # each difficulty level needs this many
    ml_difficulty_min_attempts: int = 3         # attempts before observed results replace Gemini's label
    ml_difficulty_easy_max_wrong: float = 0.30  # wrong-answer rate below this = easy
    ml_difficulty_hard_min_wrong: float = 0.60  # wrong-answer rate above this = hard
    ml_difficulty_ensemble_size: int = 100      # trees in the Random Forest and rounds in Gradient Boosting

    # Machine learning - performance model (pooled across students)
    ml_performance_min_history: int = 2         # earlier attempts needed before an attempt becomes a training example
    ml_performance_min_samples: int = 150       # training examples needed overall
    ml_performance_min_test: int = 30           # examples needed in the later (test) period
    ml_performance_test_fraction: float = 0.30  # the latest 30% of attempts are the test period
    ml_performance_ensemble_size: int = 100     # boosting rounds
    ml_performance_ewma_alpha: float = 0.30     # weight of the newest attempt in the moving average
    ml_performance_trend_window: int = 10       # attempts used for the trend
    ml_performance_trend_min_attempts: int = 5  # attempts needed before a trend is reported
    ml_performance_trend_threshold: float = 0.03   # slope per attempt that counts as improving / declining

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
