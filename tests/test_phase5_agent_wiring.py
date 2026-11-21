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


def test_goal_agent_uses_dfs_branch_and_bfs_prerequisites():
    today = date.today()
    exam_date = (today + timedelta(days=10)).isoformat()

    syllabus_topics = [
        {
            "id": "unit",
            "title": "Unit 1",
            "parent_topic_id": None,
        },
        {
            "id": "topic_a",
            "title": "Topic A",
            "parent_topic_id": "unit",
        },
        {
            "id": "subtopic_a",
            "title": "Subtopic A",
            "parent_topic_id": "topic_a",
        },
        {
            "id": "topic_b",
            "title": "Topic B",
            "parent_topic_id": "unit",
        },
    ]

    topic_stats = [
        {
            "topic_id": "subtopic_a",
            "topic_title": "Subtopic A",
            "is_leaf": True,
            "accuracy_percentage": 20,
            "paper_frequency_percentage": 60,
            "course_has_frequency_data": True,
            "days_to_exam": 10,
        },
        {
            "topic_id": "topic_b",
            "topic_title": "Topic B",
            "is_leaf": True,
            "accuracy_percentage": 90,
            "paper_frequency_percentage": 10,
            "course_has_frequency_data": True,
            "days_to_exam": 10,
        },
    ]

    prerequisites = [
        {
            "topic_id": "subtopic_a",
            "prerequisite_topic_id": "topic_b",
        },
    ]

    result = goal_based_agent(
        topic_stats=topic_stats,
        exam_date=exam_date,
        syllabus_topics=syllabus_topics,
        prerequisites=prerequisites,
    )

    ids = [item["topic_id"] for item in result]

    # DFS must cover the complete branch.
    assert set(ids) == {
        "unit",
        "topic_a",
        "subtopic_a",
        "topic_b",
    }

    # BFS must place Topic B before Subtopic A.
    assert ids.index("topic_b") < ids.index("subtopic_a")

    # Every item must have a schedule date and a reason.
    assert all(item["planned_date"] < exam_date for item in result)
    assert all(item["reason"] for item in result)