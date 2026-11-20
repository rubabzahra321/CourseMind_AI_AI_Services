from fastapi import FastAPI

from app.api import (
    health,
    stats,
    chat,
    agents,
    quiz_duel,
    practice,
    practice_formats,
)


app = FastAPI(title="CourseMind AI Service")

app.include_router(health.public)
app.include_router(health.internal)
app.include_router(stats.router)
app.include_router(chat.router)
app.include_router(practice.router)
app.include_router(practice_formats.router)
app.include_router(agents.router)
app.include_router(quiz_duel.router)