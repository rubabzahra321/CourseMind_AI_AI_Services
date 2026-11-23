"""
SYNTHETIC practice histories for tests and the demo.

Each simulated student has an ability, each topic a difficulty; knowledge grows
with practice, fades when the student stays away, and harder questions are
answered correctly less often. The generator is seeded, so it is repeatable.

This is NOT real student data and says nothing about how well the model will
work on your students.
"""

import math
import random
from datetime import datetime, timedelta, timezone

START = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def simulate(n_students=12, topics_per_student=3, attempts_per_topic=40, seed=7):
    """
    Returns a list of attempts, each:
      {"student": i, "topic": j, "t": datetime, "correct": 0/1,
       "rt": seconds, "difficulty": 0/1/2}
    """

    rng = random.Random(seed)
    attempts = []

    for student in range(n_students):
        ability = rng.gauss(0, 0.8)

        for topic in range(topics_per_student):
            topic_hardness = rng.gauss(0, 0.6)
            knowledge = -0.6
            now = START + timedelta(days=rng.randint(0, 10))
            last = None
            count = 0

            while count < attempts_per_topic:
                # practise in short sessions separated by gaps of 1-12 days
                gap_days = rng.choice([1, 1, 2, 3, 5, 8, 12])
                now += timedelta(days=gap_days)

                if last is not None:
                    knowledge -= 0.035 * gap_days                  # forgetting

                for _ in range(rng.randint(3, 5)):
                    if count >= attempts_per_topic:
                        break

                    difficulty = rng.choice([0, 1, 1, 2])
                    logit = (ability + knowledge - topic_hardness
                             + 0.5 - 0.6 * difficulty)
                    correct = int(rng.random() < _sigmoid(logit))

                    now += timedelta(minutes=rng.randint(1, 4))
                    attempts.append({
                        "student": student, "topic": topic, "t": now,
                        "correct": correct,
                        "rt": max(3, int(rng.gauss(14 + 6 * difficulty - 4 * correct, 4))),
                        "difficulty": difficulty,
                    })
                    knowledge += 0.07                               # learning
                    last = now
                    count += 1

    return attempts


def to_histories(attempts):
    """Group into one history per (student, topic), as the model expects."""

    groups = {}

    for a in attempts:
        groups.setdefault((a["student"], a["topic"]), []).append({
            "t": a["t"].timestamp(), "correct": a["correct"],
            "rt": a["rt"], "difficulty": a["difficulty"],
        })

    return list(groups.values())


LEVELS = ["easy", "medium", "hard"]


def seed_into_db(db, attempts, ids):
    """
    Write simulated attempts as generated_questions + practice_attempts rows.
    ids: {"user": {student_index: user_id}, "topic": {(student, topic): topic_id},
          "course": {student_index: course_id}, "new_id": function}
    """

    new_id = ids["new_id"]

    for a in attempts:
        qid = new_id()
        user, course = ids["user"][a["student"]], ids["course"][a["student"]]

        db.seed("generated_questions", {
            "id": qid, "question_text": f"Q {qid}", "difficulty": LEVELS[a["difficulty"]],
            "status": "active", "topic_id": ids["topic"][(a["student"], a["topic"])],
            "user_id": user, "course_id": course})
        db.seed("practice_attempts", {
            "session_id": new_id(), "question_id": qid, "user_id": user,
            "is_correct": bool(a["correct"]), "response_time_seconds": a["rt"],
            "answered_at": a["t"].isoformat()})
