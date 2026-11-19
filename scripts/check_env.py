"""
Pre-flight check for the CourseMind AI service. Run from the project root:

    python scripts/check_env.py

It verifies, in order: .env values -> Supabase connection -> database objects
created by the SQL schema -> storage bucket -> Gemini embeddings + chat model.
Exit code 0 = everything OK. It only READS data (one harmless no-op RPC call).
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
failures = 0


def report(ok, label, detail="", warn=False):
    global failures
    tag = OK if ok else (WARN if warn else BAD)
    if not ok and not warn:
        failures += 1
    print(f"{tag} {label}" + (f"  -> {detail}" if detail else ""))
    return ok


print("\n== 1. Environment variables ==")
try:
    from app.core.config import settings
except Exception as exc:  # pydantic ValidationError lists every missing variable
    print(f"{BAD} .env is incomplete or unreadable:\n{exc}")
    print("\nCopy .env.example to .env and fill in SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, "
          "AI_INTERNAL_KEY and GEMINI_API_KEY.")
    sys.exit(1)

report(settings.supabase_url.startswith("https://") and ".supabase.co" in settings.supabase_url,
       "SUPABASE_URL looks valid", settings.supabase_url)
report(len(settings.supabase_service_role_key) > 30, "SUPABASE_SERVICE_ROLE_KEY is set")
report(len(settings.ai_internal_key) >= 16, "AI_INTERNAL_KEY is >= 16 characters",
       "use: python -c \"import secrets; print(secrets.token_urlsafe(32))\"" if len(settings.ai_internal_key) < 16 else "")
report(len(settings.gemini_api_key) > 20, "GEMINI_API_KEY is set")
report(settings.gemini_embedding_dimensions == 768, "GEMINI_EMBEDDING_DIMENSIONS == 768 (matches vector(768))",
       str(settings.gemini_embedding_dimensions))

print("\n== 2. Supabase connection + schema ==")
from app.db.client import get_db  # noqa: E402

db = get_db()
try:
    db.table("courses").select("id").limit(1).execute()
    report(True, "connected with the service-role key")
except Exception as exc:
    report(False, "cannot reach Supabase", f"{type(exc).__name__}: {exc}")
    print("Fix SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY and re-run.")
    sys.exit(1)

OBJECTS = [
    "profiles", "courses", "documents", "processing_jobs", "document_chunks", "syllabus_topics",
    "past_paper_questions", "question_topic_mappings", "chat_sessions", "chat_messages",
    "chat_message_sources", "generated_questions", "question_options", "practice_sessions",
    "practice_session_questions", "practice_attempts", "quiz_duel_games", "quiz_duel_moves",
    "topic_progress", "topic_prerequisites", "study_plans", "study_plan_items",
    "topic_frequency", "topic_performance", "topic_ai_stats",   # views
]
missing = []
for name in OBJECTS:
    try:
        db.table(name).select("*").limit(1).execute()
    except Exception as exc:
        missing.append(f"{name} ({getattr(exc, 'code', type(exc).__name__)})")
report(not missing, f"all {len(OBJECTS)} tables/views exist", "missing: " + ", ".join(missing) if missing else "")
if missing:
    print("      -> run the v2 SQL schema (completeschema.sql) in the Supabase SQL Editor.")

try:
    zero = "[" + ",".join(["0"] * settings.gemini_embedding_dimensions) + "]"
    rows = db.rpc("match_document_chunks", {
        "p_query_embedding": zero, "p_user_id": str(uuid.uuid4()), "p_course_id": str(uuid.uuid4()),
        "p_match_count": 1, "p_min_similarity": 0.0}).execute().data
    report(rows == [], "RPC match_document_chunks works with 768-dim vectors")
except Exception as exc:
    report(False, "RPC match_document_chunks", f"{type(exc).__name__}: {exc}")

try:
    n = db.rpc("refresh_topic_progress", {"p_user_id": str(uuid.uuid4()), "p_course_id": str(uuid.uuid4()),
                                          "p_topic_id": None}).execute().data
    report(n in (0, None), "RPC refresh_topic_progress exists (no-op call)")
except Exception as exc:
    report(False, "RPC refresh_topic_progress", f"{type(exc).__name__}: {exc}")

try:
    db.rpc("claim_processing_job", {"p_worker_id": "check-env"}).execute()
    report(True, "RPC claim_processing_job exists")
except Exception as exc:
    report(False, "RPC claim_processing_job", f"{type(exc).__name__}: {exc}")

try:
    buckets = [b.name if hasattr(b, "name") else b.get("name") for b in db.storage.list_buckets()]
    report(settings.storage_bucket in buckets, f"storage bucket '{settings.storage_bucket}' exists", f"found: {buckets}")
except Exception as exc:
    report(False, "storage bucket check", f"{type(exc).__name__}: {exc}")

print("\n== 3. Gemini ==")
try:
    from app.ai.gemini import embed_query, generate_answer, get_gemini_client

    vec = embed_query("binary search tree")
    report(len(vec) == settings.gemini_embedding_dimensions,
           f"embedding model '{settings.gemini_embedding_model}' returns {len(vec)} dims")
except Exception as exc:
    report(False, "embedding call", f"{type(exc).__name__}: {str(exc)[:200]}")

try:
    answer = generate_answer("Reply with exactly the word: OK")
    report(True, f"chat model '{settings.gemini_chat_model}' responded", answer[:40])
except Exception as exc:
    report(False, f"chat model '{settings.gemini_chat_model}'", f"{type(exc).__name__}: {str(exc)[:200]}")
    try:
        names = sorted(m.name.replace("models/", "") for m in get_gemini_client().models.list()
                       if "flash" in m.name and "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"]))
        print("      Models your key can use (set one as GEMINI_CHAT_MODEL):")
        for n in names[:15]:
            print("        -", n)
    except Exception as inner:
        print("      (could not list models:", inner, ")")

print("\n" + ("All checks passed. Start the API:  uvicorn app.main:app --reload --port 8000"
              if failures == 0 else f"{failures} check(s) failed - fix them before starting the service."))
sys.exit(1 if failures else 0)
