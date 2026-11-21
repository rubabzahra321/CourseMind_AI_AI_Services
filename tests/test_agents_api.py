from datetime import date, timedelta

import pytest

from tests.conftest import COURSE, OTHER_COURSE, OTHER_USER, USER, new_id

AGENTS = f"/internal/courses/{COURSE}/agents"
T_WEAK, T_STRONG, T_NEW, T_UNIT = new_id(), new_id(), new_id(), new_id()


def stats_row(topic_id, title, **kw):
    base = {"topic_id": topic_id, "course_id": COURSE, "user_id": USER, "topic_title": title,
            "parent_topic_id": None, "is_leaf": True, "attempted_questions": 0, "correct_answers": 0,
            "accuracy_percentage": None, "performance_label": "Not Yet Assessed",
            "paper_frequency_percentage": 0, "papers_with_topic": 0, "total_ready_papers": 0,
            "course_has_frequency_data": False, "last_practised_at": None,
            "days_to_exam": None, "exam_date": None}
    return {**base, **kw}


@pytest.fixture()
def stats(db, world):
    old = (date.today() - timedelta(days=12)).isoformat() + "T09:00:00+00:00"
    db.seed("topic_ai_stats",
            stats_row(T_WEAK, "Weak topic", attempted_questions=5, correct_answers=1, accuracy_percentage=20,
                      paper_frequency_percentage=80, course_has_frequency_data=True, days_to_exam=10,
                      last_practised_at=old),
            stats_row(T_STRONG, "Strong topic", attempted_questions=10, correct_answers=9, accuracy_percentage=90,
                      paper_frequency_percentage=10, course_has_frequency_data=True, days_to_exam=10,
                      last_practised_at=date.today().isoformat() + "T08:00:00+00:00"),
            stats_row(T_NEW, "Unpractised topic", paper_frequency_percentage=30, course_has_frequency_data=True,
                      days_to_exam=10),
            stats_row(T_UNIT, "Unit (parent)", is_leaf=False),
            # another user's data in the same table must never leak
            stats_row(new_id(), "Not mine", user_id=OTHER_USER, course_id=OTHER_COURSE, accuracy_percentage=0))
    # Agents read topics from syllabus_topics and memory from topic_progress.
    db.seed("syllabus_topics",
            *[{"id": tid, "course_id": COURSE, "user_id": USER, "title": title, "position": pos}
              for pos, (tid, title) in enumerate(
                  ((T_WEAK, "Weak topic"), (T_NEW, "Unpractised topic"),
                   (T_STRONG, "Strong topic"), (T_UNIT, "Unit (parent)")), start=1)])
    db.seed("topic_progress",
            {"topic_id": T_WEAK, "course_id": COURSE, "user_id": USER, "attempts": 5, "correct_answers": 1,
             "accuracy_percentage": 20, "last_practised_at": old, "status": "Weak"},
            {"topic_id": T_STRONG, "course_id": COURSE, "user_id": USER, "attempts": 10, "correct_answers": 9,
             "accuracy_percentage": 90, "last_practised_at": date.today().isoformat() + "T08:00:00+00:00",
             "status": "Strong"})


def post(api, path, **body):
    return api.post(f"{AGENTS}{path}", json={"user_id": USER, **body})


# ================================================================== dashboard
def test_dashboard_combines_all_four_agents(api, stats, db):
    db.seed("documents", {"user_id": USER, "course_id": COURSE, "title": "Broken.pdf", "status": "failed"})
    body = post(api, "/dashboard").json()

    alerts = body["study_alerts"]
    assert {a["type"] for a in alerts} == {"weak_topic", "failed_document"}
    assert next(a for a in alerts if a["type"] == "weak_topic")["topic_id"] == T_WEAK

    reminders = {r["topic_id"]: r["reason"] for r in body["topic_reminders"]}
    assert reminders[T_WEAK] == "inactive_topic" and reminders[T_NEW] == "no_practice_history"
    assert T_STRONG not in reminders

    assert body["recommended_next_action"]["topic_id"] == T_WEAK
    order = [t["topic_id"] for t in body["quick_revision_order"]]
    assert order == [T_WEAK, T_NEW, T_STRONG] and T_UNIT not in order


def test_dashboard_with_no_stats_is_empty_not_an_error(api, world):
    body = post(api, "/dashboard").json()
    assert body["study_alerts"] == [] and body["recommended_next_action"] is None
    assert body["quick_revision_order"] == [] and body["model_memory_source"] == "topic_progress"
    # the model-based agent still reminds about the one unpractised syllabus topic
    assert [r["reason"] for r in body["topic_reminders"]] == ["no_practice_history"]


def test_dashboard_never_includes_other_users_data(api, stats):
    assert "Not mine" not in post(api, "/dashboard").text


# ================================================================== BFS / DFS endpoints
def test_prerequisite_path_endpoint_uses_bfs(api, db, world):
    a, b, c = new_id(), new_id(), new_id()
    for tid, title in ((a, "Arrays"), (b, "Pointers"), (c, "Linked Lists")):
        db.seed("syllabus_topics", {"id": tid, "course_id": COURSE, "user_id": USER, "title": title})
    db.seed("topic_prerequisites", {"course_id": COURSE, "user_id": USER, "topic_id": c, "prerequisite_topic_id": b},
            {"course_id": COURSE, "user_id": USER, "topic_id": b, "prerequisite_topic_id": a})
    body = post(api, f"/prerequisites/{c}").json()
    assert body["algorithm"] == "Breadth-First Search"
    assert [(s["topic_title"], s["position"]) for s in body["path"]] == [("Arrays", 1), ("Pointers", 2), ("Linked Lists", 3)]


def test_prerequisite_path_for_unknown_topic_is_404(api, world):
    assert post(api, f"/prerequisites/{new_id()}").status_code == 404


def test_prerequisite_path_cannot_read_other_users_topic(api, db, world):
    theirs = db.seed("syllabus_topics", {"course_id": OTHER_COURSE, "user_id": OTHER_USER, "title": "x"})
    assert post(api, f"/prerequisites/{theirs['id']}").status_code == 404


def test_subtree_endpoint_uses_dfs(api, db, world):
    unit, t1, t1a = new_id(), new_id(), new_id()
    db.seed("syllabus_topics",
            {"id": unit, "course_id": COURSE, "user_id": USER, "title": "Unit 1", "position": 1},
            {"id": t1, "course_id": COURSE, "user_id": USER, "title": "Topic 1", "parent_topic_id": unit, "position": 2},
            {"id": t1a, "course_id": COURSE, "user_id": USER, "title": "Sub 1a", "parent_topic_id": t1, "position": 3})
    body = post(api, f"/syllabus/{unit}/subtree").json()
    assert body["algorithm"] == "Depth-First Search"
    assert [(t["topic_title"], t["depth"]) for t in body["topics"]] == [("Unit 1", 0), ("Topic 1", 1), ("Sub 1a", 2)]


def test_subtree_for_unknown_topic_is_404(api, world):
    assert post(api, f"/syllabus/{new_id()}/subtree").status_code == 404


# ================================================================== study plan
def test_generate_plan_schedules_topics_before_the_exam(api, db, stats):
    r = post(api, "/study-plan/generate")
    assert r.status_code == 200
    body = r.json()
    items = sorted(body["items"], key=lambda i: i["position"])

    # Each topic now gets several task types (task 13.4); topics keep the utility order.
    first_seen = list(dict.fromkeys(i["topic_id"] for i in items))
    assert first_seen == [T_WEAK, T_NEW, T_STRONG]
    assert {i["task_type"] for i in items} <= {"review", "solve", "revisit_mistakes"}
    types = lambda tid: {i["task_type"] for i in items if i["topic_id"] == tid}
    assert types(T_WEAK) == {"review", "solve", "revisit_mistakes"}
    assert "revisit_mistakes" not in types(T_NEW)      # nothing answered yet -> no mistakes to revisit

    assert body["plan"]["status"] == "active"
    assert body["summary"]["total_tasks"] == len(items) == body["summary"]["scheduled_tasks"]
    assert body["feasibility"]["is_feasible"] is True and body["feasibility"]["warning"] is None
    exam = date.today() + timedelta(days=30)
    assert all(date.fromisoformat(i["planned_date"]) < exam for i in items)
    assert all(i["user_id"] == USER and i["course_id"] == COURSE and i["is_done"] is False for i in items)


def test_regenerating_archives_the_previous_plan(api, db, stats):
    first = post(api, "/study-plan/generate").json()["plan"]["id"]
    second = post(api, "/study-plan/generate").json()["plan"]["id"]
    assert first != second
    status = {p["id"]: p["status"] for p in db.rows("study_plans")}
    assert status == {first: "archived", second: "active"}


def test_generate_plan_requires_an_exam_date(api, db, stats):
    db.tables["courses"][0]["exam_date"] = None
    r = post(api, "/study-plan/generate")
    assert r.status_code == 400 and "exam date" in r.json()["detail"].lower()
    assert db.rows("study_plans") == []


def test_generate_plan_rejects_past_exam_and_keeps_existing_plan(api, db, stats):
    post(api, "/study-plan/generate")
    db.tables["courses"][0]["exam_date"] = (date.today() - timedelta(days=1)).isoformat()
    assert post(api, "/study-plan/generate").status_code == 400
    assert [p["status"] for p in db.rows("study_plans")] == ["active"]   # old plan was not archived


def test_generate_plan_for_unknown_or_foreign_course(api, db, world):
    other = f"/internal/courses/{OTHER_COURSE}/agents/study-plan/generate"
    assert api.post(other, json={"user_id": USER}).status_code == 400
    assert api.post(f"/internal/courses/{new_id()}/agents/study-plan/generate", json={"user_id": USER}).status_code == 400


def test_current_plan_is_null_until_one_exists_then_returned(api, db, stats):
    assert post(api, "/study-plan/current").json() is None
    post(api, "/study-plan/generate")
    body = post(api, "/study-plan/current").json()
    assert body["plan"]["status"] == "active" and len(body["items"]) == 8
    assert [i["position"] for i in body["items"]] == list(range(1, 9))


def test_complete_item_marks_done_and_stamps_time(api, db, stats):
    item = post(api, "/study-plan/generate").json()["items"][0]
    r = post(api, "/study-plan/complete", item_id=item["id"])
    assert r.status_code == 200 and r.json()["is_done"] is True and r.json()["done_at"]
    assert db.rows("study_plan_items", id=item["id"])[0]["is_done"] is True


def test_complete_unknown_item_is_404_and_foreign_item_untouched(api, db, stats):
    assert post(api, "/study-plan/complete", item_id=new_id()).status_code == 404
    theirs = db.seed("study_plan_items", {"user_id": OTHER_USER, "course_id": COURSE, "plan_id": new_id(),
                                          "topic_id": new_id(), "reason": "x", "planned_date": "2030-01-01"})
    assert post(api, "/study-plan/complete", item_id=theirs["id"]).status_code == 404
    assert db.rows("study_plan_items", id=theirs["id"])[0]["is_done"] is False
def test_model_based_agent_reads_topic_progress_memory(api, db, stats):
    body = post(api, "/dashboard").json()

    reminders = {
        item["topic_id"]: item
        for item in body["topic_reminders"]
    }

    assert reminders[T_WEAK]["reason"] == "inactive_topic"
    assert reminders[T_WEAK]["attempts"] == 5

    # A topic without a topic_progress row is treated as unpractised.
    assert reminders[T_NEW]["reason"] == "no_practice_history"

    assert body["model_memory_source"] == "topic_progress"