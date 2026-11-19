
from datetime import date, timedelta


def simple_reflex_agent(topic_stats, failed_documents=None):
    """
    Reacts only to the current situation using fixed rules.
    """
    alerts = []

    for topic in topic_stats:
        attempts = int(topic.get("attempted_questions") or 0)
        accuracy = topic.get("accuracy_percentage")

        if attempts >= 3 and accuracy is not None and float(accuracy) < 50:
            alerts.append({
                "type": "weak_topic",
                "topic_id": topic["topic_id"],
                "topic_title": topic["topic_title"],
                "message": "Weak topic. Practise now.",
                "priority": "high",
            })

    for document in failed_documents or []:
        alerts.append({
            "type": "failed_document",
            "document_id": document["id"],
            "message": f"Document processing failed: {document.get('title', 'Untitled')}. Retry processing.",
            "priority": "medium",
        })

    return alerts


def model_based_reflex_agent(topic_stats):
    """
    Uses stored topic progress and practice history.
    """
    reminders = []
    today = date.today()

    for topic in topic_stats:
        last_practised = topic.get("last_practised_at")
        attempts = int(topic.get("attempted_questions") or 0)

        if not last_practised:
            reminders.append({
                "topic_id": topic["topic_id"],
                "topic_title": topic["topic_title"],
                "message": "This topic has not been practised yet.",
                "reason": "no_practice_history",
            })
            continue

        try:
            last_date = date.fromisoformat(str(last_practised)[:10])
        except (TypeError, ValueError):
            continue

        days = (today - last_date).days

        if days >= 7:
            reminders.append({
                "topic_id": topic["topic_id"],
                "topic_title": topic["topic_title"],
                "message": f"You have not practised this topic for {days} days.",
                "reason": "inactive_topic",
                "days_since_practice": days,
                "attempts": attempts,
            })

    return reminders


def calculate_utility(topic):
    """
    Utility score:
    40% past-paper frequency
    40% weakness
    20% exam urgency

    If past-paper frequency data does not exist, the frequency
    component is ignored and the remaining weights are normalized.
    """
    accuracy = topic.get("accuracy_percentage")
    weakness = 1.0 if accuracy is None else (
        1.0 - max(0.0, min(100.0, float(accuracy))) / 100.0
    )

    frequency = float(topic.get("paper_frequency_percentage") or 0) / 100
    frequency = max(0.0, min(1.0, frequency))

    days_to_exam = topic.get("days_to_exam")
    if days_to_exam is None:
        urgency = 0.0
    elif int(days_to_exam) <= 0:
        urgency = 1.0
    else:
        # Urgency rises as the exam date approaches.
        urgency = max(0.0, min(1.0, 1.0 - int(days_to_exam) / 60))

    has_frequency = bool(topic.get("course_has_frequency_data"))

    if has_frequency:
        score = (
            0.40 * frequency
            + 0.40 * weakness
            + 0.20 * urgency
        )
    else:
        score = (
            0.65 * weakness
            + 0.35 * urgency
        )

    return round(score * 100, 2)


def utility_based_agent(topic_stats):
    """
    Recommends the topic with the highest utility score.
    """
    candidates = []

    for topic in topic_stats:
        if not topic.get("is_leaf", True):
            continue

        item = dict(topic)
        item["utility_score"] = calculate_utility(topic)
        candidates.append(item)

    candidates.sort(
        key=lambda item: item["utility_score"],
        reverse=True
    )

    if not candidates:
        return None

    best = candidates[0]
    reasons = []

    if best.get("accuracy_percentage") is None:
        reasons.append("This topic has not been practised yet.")
    elif float(best["accuracy_percentage"]) < 50:
        reasons.append("Your accuracy is below 50%.")
    elif float(best["accuracy_percentage"]) < 80:
        reasons.append("Your accuracy can be improved.")

    if best.get("course_has_frequency_data"):
        if float(best.get("paper_frequency_percentage") or 0) >= 50:
            reasons.append("This topic appears frequently in reviewed past papers.")
    else:
        reasons.append("Past-paper topic mapping is not available yet.")

    if best.get("days_to_exam") is not None:
        if int(best["days_to_exam"]) <= 7:
            reasons.append("The exam is within seven days.")

    return {
        "topic_id": best["topic_id"],
        "topic_title": best["topic_title"],
        "utility_score": best["utility_score"],
        "reasons": reasons,
    }


def goal_based_agent(topic_stats, exam_date):
    """
    Builds a simple daily study schedule before the exam.

    Topics with higher utility are scheduled earlier.
    """
    if not exam_date:
        raise ValueError("Set an exam date before generating a study plan.")

    exam_day = date.fromisoformat(str(exam_date)[:10])
    today = date.today()

    if exam_day <= today:
        raise ValueError("Exam date must be in the future.")

    topics = []

    for topic in topic_stats:
        if not topic.get("is_leaf", True):
            continue

        item = dict(topic)
        item["utility_score"] = calculate_utility(topic)
        topics.append(item)

    topics.sort(
        key=lambda item: item["utility_score"],
        reverse=True
    )

    if not topics:
        return []

    available_days = (exam_day - today).days
    schedule = []

    for index, topic in enumerate(topics):
        study_day = today + timedelta(
            days=min(index, available_days - 1)
        )

        schedule.append({
            "topic_id": topic["topic_id"],
            "topic_title": topic["topic_title"],
            "planned_date": study_day.isoformat(),
            "position": index + 1,
            "utility_score": topic["utility_score"],
            "reason": (
                f"Priority score: {topic['utility_score']}. "
                "Scheduled according to topic weakness, frequency and exam urgency."
            ),
        })

    return schedule
