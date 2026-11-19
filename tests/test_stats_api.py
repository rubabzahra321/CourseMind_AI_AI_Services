from tests.conftest import COURSE, OTHER_COURSE, OTHER_USER, USER, new_id
from tests.test_agents_api import stats_row


def test_topic_stats_requires_user_id(api):
    assert api.get(f"/internal/courses/{COURSE}/topic-stats").status_code == 422


def test_topic_stats_requires_internal_key(anon_api):
    assert anon_api.get(f"/internal/courses/{COURSE}/topic-stats", params={"user_id": USER}).status_code == 401


def test_topic_stats_flags_missing_frequency_data(api, db, world):
    db.seed("topic_ai_stats", stats_row(new_id(), "A"))
    body = api.get(f"/internal/courses/{COURSE}/topic-stats", params={"user_id": USER}).json()
    assert body["course_has_frequency_data"] is False and len(body["topics"]) == 1


def test_topic_stats_reports_frequency_data_when_any_topic_has_it(api, db, world):
    db.seed("topic_ai_stats", stats_row(new_id(), "A"), stats_row(new_id(), "B", course_has_frequency_data=True))
    body = api.get(f"/internal/courses/{COURSE}/topic-stats", params={"user_id": USER}).json()
    assert body["course_has_frequency_data"] is True and len(body["topics"]) == 2


def test_topic_stats_is_scoped_to_user_and_course(api, db, world):
    db.seed("topic_ai_stats", stats_row(new_id(), "Mine"),
            stats_row(new_id(), "Other course", course_id=OTHER_COURSE),
            stats_row(new_id(), "Other user", user_id=OTHER_USER))
    titles = [t["topic_title"] for t in
              api.get(f"/internal/courses/{COURSE}/topic-stats", params={"user_id": USER}).json()["topics"]]
    assert titles == ["Mine"]


def test_topic_stats_empty_course(api, world):
    assert api.get(f"/internal/courses/{COURSE}/topic-stats", params={"user_id": USER}).json() == \
        {"course_has_frequency_data": False, "topics": []}
