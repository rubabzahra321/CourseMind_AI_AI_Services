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
    syllabus_topics=None,
    prerequisites=None,
):
    """
    Builds a study schedule before the exam.

    When syllabus_topics and prerequisites are supplied:
      - DFS covers the entire syllabus branch.
      - BFS puts prerequisites before dependent topics.

    Without those arguments, the original utility-based leaf ordering
    is retained for backward compatibility.
    """
    if not exam_date:
        raise ValueError(
            "Set an exam date before generating a study plan."
        )

    exam_day = date.fromisoformat(str(exam_date)[:10])
    today = date.today()

    if exam_day <= today:
        raise ValueError("Exam date must be in the future.")

    if syllabus_topics:
        ordered_topics = _build_dfs_study_order(
            topic_stats=topic_stats,
            syllabus_topics=syllabus_topics,
            prerequisites=prerequisites or [],
        )
    else:
        topics = []

        for topic in topic_stats:
            if not topic.get("is_leaf", True):
                continue

            item = dict(topic)
            item["utility_score"] = calculate_utility(topic)
            topics.append(item)

        topics.sort(
            key=lambda item: item["utility_score"],
            reverse=True,
        )

        ordered_topics = [
            {
                "topic_id": item["topic_id"],
                "topic_title": item["topic_title"],
                "utility_score": item["utility_score"],
                "ordering_reason": (
                    "Ordered by utility score because no syllabus tree "
                    "was supplied."
                ),
                "dfs_depth": 0,
            }
            for item in topics
        ]

    if not ordered_topics:
        return []

    available_days = (exam_day - today).days
    schedule = []

    for index, topic in enumerate(ordered_topics):
        study_day = today + timedelta(
            days=min(index, available_days - 1)
        )

        schedule.append({
            "topic_id": topic["topic_id"],
            "topic_title": topic["topic_title"],
            "planned_date": study_day.isoformat(),
            "position": index + 1,
            "utility_score": topic["utility_score"],
            "dfs_depth": topic["dfs_depth"],
            "reason": (
                f"Priority score: {topic['utility_score']}. "
                f"{topic['ordering_reason']}"
            ),
        })

    return schedule