
from datetime import date, timedelta

from app.search.algorithms import (
    bfs_all_prerequisites,
    dfs_topic_subtree,
)


def simple_reflex_agent(topic_stats, failed_documents=None):
    """
    Reacts to the current situation using fixed rules.
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
            "message": (
                f"Document processing failed: "
                f"{document.get('title', 'Untitled')}. Retry processing."
            ),
            "priority": "medium",
        })

    return alerts


def model_based_reflex_agent(topic_progress):
    """
    Uses the topic_progress table as the agent's stored memory.

    Each item should contain:
        topic_id
        topic_title
        attempts
        last_practised_at
    """
    reminders = []
    today = date.today()

    for topic in topic_progress:
        last_practised = topic.get("last_practised_at")
        attempts = int(topic.get("attempts") or 0)

        if not last_practised:
            reminders.append({
                "topic_id": topic["topic_id"],
                "topic_title": topic["topic_title"],
                "message": "This topic has not been practised yet.",
                "reason": "no_practice_history",
                "attempts": attempts,
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
                "message": (
                    f"You have not practised this topic for {days} days."
                ),
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

    If frequency data is unavailable, the remaining weights
    are normalized.
    """
    accuracy = topic.get("accuracy_percentage")

    weakness = 1.0 if accuracy is None else (
        1.0 - max(0.0, min(100.0, float(accuracy))) / 100.0
    )

    frequency = float(
        topic.get("paper_frequency_percentage") or 0
    ) / 100

    frequency = max(0.0, min(1.0, frequency))

    days_to_exam = topic.get("days_to_exam")

    if days_to_exam is None:
        urgency = 0.0
    elif int(days_to_exam) <= 0:
        urgency = 1.0
    else:
        urgency = max(
            0.0,
            min(1.0, 1.0 - int(days_to_exam) / 60)
        )

    has_frequency = bool(
        topic.get("course_has_frequency_data")
    )

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


def schedule_revision_tasks(
    topic_stats,
    topic_schedule,
    exam_date,
    hours_per_day=2,
):
    """
    Converts the Phase 5 ordered topic schedule into
    review, solve, and revisit-mistakes tasks.

    Preserves the topic order supplied by the BFS/DFS
    goal-based agent.
    """

    if not exam_date:
        raise ValueError("Set an exam date first.")

    hours_per_day = float(hours_per_day or 2)

    if hours_per_day < 0.5 or hours_per_day > 12:
        raise ValueError("Daily hours must be between 0.5 and 12.")

    today = date.today()
    exam_day = date.fromisoformat(str(exam_date)[:10])

    if exam_day <= today:
        raise ValueError("Exam date must be in the future.")

    daily_minutes = int(hours_per_day * 60)
    available_days = (exam_day - today).days

    stats_by_id = {
        str(topic["topic_id"]): topic
        for topic in topic_stats
    }

    tasks = []

    # Preserve the topic order produced by Phase 5.
    for ordered_topic in topic_schedule:
        topic_id = str(ordered_topic["topic_id"])
        topic = stats_by_id.get(topic_id)

        if not topic:
            continue

        utility_score = ordered_topic.get(
            "utility_score",
            calculate_utility(topic),
        )

        attempted = int(topic.get("attempted_questions") or 0)
        correct = int(topic.get("correct_answers") or 0)
        incorrect = max(0, attempted - correct)

        common = {
            "topic_id": topic_id,
            "topic_title": topic.get("topic_title", "Untitled topic"),
            "utility_score": utility_score,
        }

        tasks.append({
            **common,
            "task_type": "review",
            "estimated_minutes": 20,
            "reason": "Review the topic and its important concepts.",
        })

        tasks.append({
            **common,
            "task_type": "solve",
            "estimated_minutes": 30,
            "reason": "Solve practice questions for this topic.",
        })

        if incorrect > 0:
            tasks.append({
                **common,
                "task_type": "revisit_mistakes",
                "estimated_minutes": 15,
                "reason": f"Revisit {incorrect} incorrect answers.",
            })

    scheduled = []
    unscheduled = []

    day_index = 0
    used_minutes = 0
    position = 1

    for task in tasks:
        minutes = task["estimated_minutes"]

        if used_minutes + minutes > daily_minutes:
            day_index += 1
            used_minutes = 0

        if day_index >= available_days:
            unscheduled.append(task)
            continue

        planned_date = today + timedelta(days=day_index)

        scheduled.append({
            **task,
            "planned_date": planned_date.isoformat(),
            "position": position,
        })

        used_minutes += minutes
        position += 1

    return {
        "items": scheduled,
        "daily_minutes": daily_minutes,
        "available_days": available_days,
        "total_tasks": len(tasks),
        "scheduled_tasks": len(scheduled),
        "unscheduled_tasks": unscheduled,
    }


def utility_based_agent(topic_stats):
    """
    Recommends the leaf topic with the highest utility score.
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
            reasons.append(
                "This topic appears frequently in reviewed past papers."
            )
    else:
        reasons.append(
            "Past-paper topic mapping is not available yet."
        )

    if best.get("days_to_exam") is not None:
        if int(best["days_to_exam"]) <= 7:
            reasons.append("The exam is within seven days.")

    return {
        "topic_id": best["topic_id"],
        "topic_title": best["topic_title"],
        "utility_score": best["utility_score"],
        "reasons": reasons,
    }


def _build_dfs_study_order(
    topic_stats,
    syllabus_topics,
    prerequisites,
):
    """
    Build a study order using both algorithms.

    1. Rank syllabus branches by the highest utility of their leaf topics.
    2. Use DFS to visit every topic in each branch.
    3. Use BFS prerequisite ordering before adding each topic.
    4. Avoid scheduling the same topic more than once.
    """
    topic_map = {
        str(topic["id"]): topic
        for topic in syllabus_topics
    }

    stats_map = {
        str(topic["topic_id"]): topic
        for topic in topic_stats
    }

    children = {}

    for topic in syllabus_topics:
        parent_id = topic.get("parent_topic_id")
        children.setdefault(
            str(parent_id) if parent_id else None,
            []
        ).append(topic)

    # Find leaf topics and calculate their utility.
    leaf_scores = {}

    for topic in topic_stats:
        topic_id = str(topic["topic_id"])

        if topic.get("is_leaf", True):
            leaf_scores[topic_id] = calculate_utility(topic)

    # Group leaves by their top-level syllabus branch.
    branch_scores = {}

    for leaf_id, score in leaf_scores.items():
        if leaf_id not in topic_map:
            continue

        current_id = leaf_id

        while True:
            current_topic = topic_map[current_id]
            parent_id = current_topic.get("parent_topic_id")

            if not parent_id:
                break

            parent_id = str(parent_id)

            if parent_id not in topic_map:
                break

            current_id = parent_id

        branch_scores[current_id] = max(
            branch_scores.get(current_id, 0),
            score,
        )

    # Higher-priority branches are visited first.
    ordered_roots = sorted(
        branch_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    final_order = []
    already_added = set()

    for root_id, branch_score in ordered_roots:
        # DFS visits the complete branch.
        branch_topics = dfs_topic_subtree(
            root_id,
            syllabus_topics,
        )

        for branch_item in branch_topics:
            topic_id = str(branch_item["topic_id"])

            # BFS places every prerequisite before this topic.
            prerequisite_order = bfs_all_prerequisites(
                topic_id,
                prerequisites,
            )

            for prerequisite_id in prerequisite_order:
                prerequisite_id = str(prerequisite_id)

                if prerequisite_id in already_added:
                    continue

                if prerequisite_id not in topic_map:
                    continue

                topic = topic_map[prerequisite_id]
                stat = stats_map.get(prerequisite_id, {})

                utility_score = (
                    calculate_utility(stat)
                    if stat
                    else branch_score
                )

                final_order.append({
                    "topic_id": prerequisite_id,
                    "topic_title": topic["title"],
                    "utility_score": utility_score,
                    "dfs_depth": branch_item["depth"],
                    "ordering_reason": (
                        "BFS prerequisite ordering and DFS syllabus coverage."
                    ),
                })

                already_added.add(prerequisite_id)

    return final_order


def goal_based_agent(
    topic_stats,
    exam_date,
    hours_per_day=2,
):
    """
    Builds a study plan using the student's daily time budget.

    Tasks are scheduled according to:
    - Topic utility score
    - Daily available study minutes
    - Exam date

    Task types:
    - review
    - solve
    - revisit_mistakes
    """

    if not exam_date:
        raise ValueError(
            "Set an exam date before generating a study plan."
        )

    if hours_per_day is None:
        hours_per_day = 2

    hours_per_day = float(hours_per_day)

    if hours_per_day < 0.5 or hours_per_day > 12:
        raise ValueError(
            "Daily study time must be between 0.5 and 12 hours."
        )

    exam_day = date.fromisoformat(str(exam_date)[:10])
    today = date.today()

    if exam_day <= today:
        raise ValueError("Exam date must be in the future.")

    daily_minutes = int(hours_per_day * 60)
    available_days = (exam_day - today).days

    # -------------------------------------------------
    # 1. Select leaf topics and calculate priority.
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 2. Create study tasks for every selected topic.
    # -------------------------------------------------

    tasks = []

    for topic in topics:
        topic_id = topic["topic_id"]
        topic_title = topic.get("topic_title", "Untitled topic")
        utility_score = topic["utility_score"]

        attempted = int(topic.get("attempted_questions") or 0)
        correct = int(topic.get("correct_answers") or 0)

        incorrect = max(0, attempted - correct)

        # Review task.
        tasks.append({
            "topic_id": topic_id,
            "topic_title": topic_title,
            "task_type": "review",
            "estimated_minutes": 20,
            "utility_score": utility_score,
            "reason": (
                f"Review this topic. Priority score: {utility_score}."
            ),
        })

        # Practice/solve task.
        tasks.append({
            "topic_id": topic_id,
            "topic_title": topic_title,
            "task_type": "solve",
            "estimated_minutes": 30,
            "utility_score": utility_score,
            "reason": (
                "Solve practice questions to check understanding."
            ),
        })

        # Only create a mistake-revision task when mistakes exist.
        if incorrect > 0:
            tasks.append({
                "topic_id": topic_id,
                "topic_title": topic_title,
                "task_type": "revisit_mistakes",
                "estimated_minutes": 15,
                "utility_score": utility_score,
                "reason": (
                    f"Revisit mistakes from {incorrect} incorrect answers."
                ),
            })

    # -------------------------------------------------
    # 3. Schedule tasks without exceeding daily hours.
    # -------------------------------------------------

    schedule = []
    unscheduled_tasks = []

    current_day = 0
    minutes_used_today = 0
    position = 1

    for task in tasks:

        task_minutes = task["estimated_minutes"]

        # Move to the next day if this task does not fit.
        if minutes_used_today + task_minutes > daily_minutes:
            current_day += 1
            minutes_used_today = 0

        # Do not schedule tasks on or after exam day.
        if current_day >= available_days:
            unscheduled_tasks.append(task)
            continue

        planned_day = today + timedelta(days=current_day)

        schedule.append({
            "topic_id": task["topic_id"],
            "topic_title": task["topic_title"],
            "task_type": task["task_type"],
            "estimated_minutes": task_minutes,
            "planned_date": planned_day.isoformat(),
            "position": position,
            "utility_score": task["utility_score"],
            "reason": task["reason"],
        })

        minutes_used_today += task_minutes
        position += 1

    return {
        "items": schedule,
        "daily_minutes": daily_minutes,
        "available_days": available_days,
        "total_tasks": len(tasks),
        "scheduled_tasks": len(schedule),
        "unscheduled_tasks": unscheduled_tasks,
    }
