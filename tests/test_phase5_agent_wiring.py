
from datetime import date, timedelta

import pytest

from app.search.algorithms import (
    bfs_all_prerequisites,
    dfs_topic_subtree,
)

from app.agents.stage6_agents import goal_based_agent


def test_bfs_returns_all_prerequisites_before_target():
    prerequisites = [
        {
            "topic_id": "C",
            "prerequisite_topic_id": "A",
        },
        {
            "topic_id": "C",
            "prerequisite_topic_id": "B",
        },
        {
            "topic_id": "B",
            "prerequisite_topic_id": "D",
        },
    ]

    result = bfs_all_prerequisites("C", prerequisites)

    assert set(result) == {"A", "B", "C", "D"}

    # Every prerequisite must appear before the topic that needs it.
    positions = {
        topic_id: index
        for index, topic_id in enumerate(result)
    }

    assert positions["A"] < positions["C"]
    assert positions["B"] < positions["C"]
    assert positions["D"] < positions["B"]


def test_bfs_detects_prerequisite_cycles():
    prerequisites = [
        {
            "topic_id": "A",
            "prerequisite_topic_id": "B",
        },
        {
            "topic_id": "B",
            "prerequisite_topic_id": "A",
        },
    ]

    with pytest.raises(ValueError, match="cycle"):
        bfs_all_prerequisites("A", prerequisites)


def test_dfs_returns_root_and_all_descendants():
    topics = [
        {
            "id": "unit",
            "parent_topic_id": None,
            "title": "Unit",
        },
        {
            "id": "topic_a",
            "parent_topic_id": "unit",
            "title": "Topic A",
        },
        {
            "id": "subtopic_a",
            "parent_topic_id": "topic_a",
            "title": "Subtopic A",
        },
        {
            "id": "topic_b",
            "parent_topic_id": "unit",
            "title": "Topic B",
        },
    ]

    result = dfs_topic_subtree("unit", topics)

    assert [
        item["topic_id"] for item in result
    ] == [
        "unit",
        "topic_a",
        "subtopic_a",
        "topic_b",
    ]

    assert [
        item["depth"] for item in result
    ] == [0, 1, 2, 1]


def test_goal_agent_creates_schedule_from_topic_stats():
    today = date.today()
    exam_date = (today + timedelta(days=10)).isoformat()

    topic_stats = [
        {
            "topic_id": "topic_a",
            "topic_title": "Topic A",
            "is_leaf": True,
            "attempted_questions": 5,
            "correct_answers": 1,
            "accuracy_percentage": 20,
            "paper_frequency_percentage": 60,
            "course_has_frequency_data": True,
            "days_to_exam": 10,
        },
        {
            "topic_id": "topic_b",
            "topic_title": "Topic B",
            "is_leaf": True,
            "attempted_questions": 5,
            "correct_answers": 4,
            "accuracy_percentage": 80,
            "paper_frequency_percentage": 10,
            "course_has_frequency_data": True,
            "days_to_exam": 10,
        },
    ]

    result = goal_based_agent(
        topic_stats=topic_stats,
        exam_date=exam_date,
    )

    assert isinstance(result, dict)
    assert result["total_tasks"] > 0
    assert result["scheduled_tasks"] > 0

    items = result["items"]

    assert items[0]["topic_id"] == "topic_a"
    assert all(item["planned_date"] < exam_date for item in items)
    assert all(item["reason"] for item in items)
