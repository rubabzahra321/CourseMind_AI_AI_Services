
from fastapi import FastAPI

from app.api import (
    health,
    stats,
    chat,
    agents,
    quiz_duel,
    practice,
    analytics,
    phase4,
)


app = FastAPI(
    title="CourseMind AI Service"
)


app.include_router(health.public)
app.include_router(health.internal)

app.include_router(stats.router)
app.include_router(chat.router)
app.include_router(practice.router)
app.include_router(agents.router)
app.include_router(quiz_duel.router)
app.include_router(analytics.router)

app.include_router(phase4.router)
