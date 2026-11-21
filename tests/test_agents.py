from datetime import date, timedelta

import pytest

from app.agents.stage6_agents import (
    calculate_utility, goal_based_agent, model_based_reflex_agent,
    simple_reflex_agent, utility_based_agent,
)


def topic(**kw):
    base = {"topic_id": "t", "topic_title": "T", "is_leaf": True, "attempted_questions": 0,
            "accuracy_percentage": None, "paper_frequency_percentage": 0,
            "course_has_frequency_data": False, "days_to_exam": None, "last_practised_at": None}
    return {**base, **kw}


# ------------------------------------------------------------ utility function
def test_utility_unassessed_no_frequency_no_exam():
    # weakness=1 -> 0.65 * 1
    assert calculate_utility(topic()) == 65.0


def test_utility_with_frequency_data_exact_value():
    t = topic(accuracy_percentage=20, paper_frequency_percentage=100,
              course_has_frequency_data=True, days_to_exam=30)
    # 0.4*1.0 + 0.4*0.8 + 0.2*0.5
    assert calculate_utility(t) == 82.0


def test_utility_exam_today_or_past_is_max_urgency():
    for d in (0, -5):
        assert calculate_utility(topic(accuracy_percentage=100, days_to_exam=d)) == 35.0  # 0.35*1


def test_utility_far_exam_has_zero_urgency():
    assert calculate_utility(topic(accuracy_percentage=100, days_to_exam=365)) == 0.0


def test_utility_weak_topic_scores_higher_than_strong():
    weak = calculate_utility(topic(accuracy_percentage=10))
    strong = calculate_utility(topic(accuracy_percentage=95))
    assert weak > strong


def test_utility_clamps_out_of_range_inputs():
    t = topic(accuracy_percentage=150, paper_frequency_percentage=900, course_has_frequency_data=True)
    assert 0 <= calculate_utility(t) <= 100


# ------------------------------------------------------------ simple reflex agent
def test_simple_reflex_flags_weak_topic_only_with_enough_attempts():
    weak = topic(topic_id="w", attempted_questions=3, accuracy_percentage=40)
    too_few = topic(topic_id="f", attempted_questions=2, accuracy_percentage=0)
    ok = topic(topic_id="o", attempted_questions=10, accuracy_percentage=50)
    alerts = simple_reflex_agent([weak, too_few, ok])
    assert [a["topic_id"] for a in alerts] == ["w"]
    assert alerts[0]["priority"] == "high"


def test_simple_reflex_flags_failed_documents():
    alerts = simple_reflex_agent([], [{"id": "d1", "title": "Notes.pdf"}])
    assert alerts[0]["type"] == "failed_document" and "Notes.pdf" in alerts[0]["message"]


# ------------------------------------------------------------ model-based reflex agent
def test_model_based_reminders():
    today = date.today()
    never = topic(topic_id="never")
    fresh = topic(topic_id="fresh", last_practised_at=today.isoformat() + "T10:00:00+00:00")
    stale = topic(topic_id="stale", last_practised_at=(today - timedelta(days=9)).isoformat() + "T10:00:00+00:00")
    garbage = topic(topic_id="bad", last_practised_at="not-a-date")
    out = {r["topic_id"]: r for r in model_based_reflex_agent([never, fresh, stale, garbage])}
    assert set(out) == {"never", "stale"}
    assert out["never"]["reason"] == "no_practice_history"
    assert out["stale"]["days_since_practice"] == 9


# ------------------------------------------------------------ utility-based agent
def test_utility_agent_recommends_highest_utility_leaf_and_skips_parents():
    parent = topic(topic_id="unit", is_leaf=False, accuracy_percentage=0)
    easy = topic(topic_id="easy", accuracy_percentage=95)
    hard = topic(topic_id="hard", accuracy_percentage=20)
    best = utility_based_agent([parent, easy, hard])
    assert best["topic_id"] == "hard"
    assert "Your accuracy is below 50%." in best["reasons"]
    assert "Past-paper topic mapping is not available yet." in best["reasons"]


def test_utility_agent_returns_none_when_nothing_to_recommend():
    assert utility_based_agent([]) is None
    assert utility_based_agent([topic(is_leaf=False)]) is None



# ------------------------------------------------------------ goal-based agent
def test_goal_agent_requires_future_exam_date():
    with pytest.raises(ValueError):
        goal_based_agent([topic()], None)

    with pytest.raises(ValueError):
        goal_based_agent([topic()], date.today().isoformat())

    with pytest.raises(ValueError):
        goal_based_agent(
            [topic()],
            (date.today() - timedelta(days=1)).isoformat()
        )


def test_goal_agent_schedules_highest_utility_topics_first():
    topics = [
        topic(topic_id="a", accuracy_percentage=90),
        topic(topic_id="b", accuracy_percentage=10),
        topic(topic_id="c", accuracy_percentage=50),
    ]

    exam = (date.today() + timedelta(days=10)).isoformat()
    plan = goal_based_agent(topics, exam)

    items = plan["items"]

    assert [item["topic_id"] for item in items] == [
        "b", "b", "c", "c", "a", "a"
    ]

    assert [item["task_type"] for item in items] == [
        "review", "solve",
        "review", "solve",
        "review", "solve",
    ]

    assert [item["position"] for item in items] == [
        1, 2, 3, 4, 5, 6
    ]

    today = date.today()

    assert [item["planned_date"] for item in items] == [
        today.isoformat(),
        today.isoformat(),
        today.isoformat(),
        today.isoformat(),
        today.isoformat(),
        (today + timedelta(days=1)).isoformat(),
    ]


def test_goal_agent_never_schedules_on_or_after_exam_day():
    topics = [
        topic(topic_id=str(i), accuracy_percentage=i)
        for i in range(10)
    ]

    exam_day = date.today() + timedelta(days=3)

    plan = goal_based_agent(topics, exam_day.isoformat())

    assert plan["total_tasks"] == 20
    assert plan["scheduled_tasks"] + len(plan["unscheduled_tasks"]) == 20

    assert all(
        date.fromisoformat(item["planned_date"]) < exam_day
        for item in plan["items"]
    )


def test_goal_agent_skips_parent_topics_and_handles_empty():
    exam = (date.today() + timedelta(days=5)).isoformat()

    parent_plan = goal_based_agent(
        [topic(is_leaf=False)],
        exam
    )

    assert parent_plan["items"] == []
    assert parent_plan["total_tasks"] == 0

    empty_plan = goal_based_agent([], exam)

    assert empty_plan["items"] == []
    assert empty_plan["total_tasks"] == 0
