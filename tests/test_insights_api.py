"""K-03 paper insights and S-01 syllabus-to-exam knowledge map."""
import pytest

from app.core.config import settings
from tests.conftest import COURSE, OTHER_COURSE, OTHER_USER, USER, new_id

BASE = f"/internal/courses/{COURSE}"
PARAMS = {"user_id": USER}

T_TREES, T_GRAPHS, T_SORT, T_EMPTY = (new_id() for _ in range(4))
P23, P24, P_PENDING, P_OTHER = (new_id() for _ in range(4))


def topic(tid, title, pos, parent=None, user=USER, course=COURSE):
    return {"id": tid, "title": title, "position": pos, "parent_topic_id": parent,
            "user_id": user, "course_id": course}


def paper(pid, year, status="ready", user=USER, course=COURSE, title=None):
    return {"id": pid, "title": title or f"Paper {year}", "document_type": "past_paper",
            "status": status, "academic_year": year, "user_id": user, "course_id": course}


def question(did, number, marks, year, review="reviewed", user=USER, course=COURSE):
    qid = new_id()
    return qid, {"id": qid, "document_id": did, "question_number": number, "question_text": f"Q{number}?",
                 "marks": marks, "paper_year": year, "page_number": 2, "review_status": review,
                 "user_id": user, "course_id": course}


def mapping(qid, tid, user=USER):
    return {"past_paper_question_id": qid, "topic_id": tid, "user_id": user}


@pytest.fixture
def exam(db, world):
    """Trees are examined in both papers, graphs in one, sorting never."""
    db.seed("syllabus_topics",
            topic(T_TREES, "Trees", 2), topic(T_GRAPHS, "Graphs", 3),
            topic(T_SORT, "Sorting", 4), topic(T_EMPTY, "BST", 5, parent=T_TREES))
    db.seed("documents", paper(P23, 2023), paper(P24, 2024),
            paper(P_PENDING, 2022, status="processing"))

    q1, r1 = question(P23, "1", 10, 2023)
    q2, r2 = question(P24, "1a", 6, 2024)
    q3, r3 = question(P24, "2", None, 2024)          # no marks
    q4, r4 = question(P23, "3", 20, 2023, review="pending")    # not reviewed
    q5, r5 = question(P_PENDING, "1", 8, 2022)       # paper not ready
    db.seed("past_paper_questions", r1, r2, r3, r4, r5)
    db.seed("question_topic_mappings",
            mapping(q1, T_TREES), mapping(q2, T_TREES), mapping(q3, T_GRAPHS),
            mapping(q4, T_SORT), mapping(q5, T_SORT))
    return SimpleNamespace(q=[q1, q2, q3, q4, q5])


from types import SimpleNamespace  # noqa: E402


def by_topic(body):
    return {t["topic_id"]: t for t in body["topics"]}


# ================================================================ K-03
class TestPaperInsights:
    def get(self, api):
        r = api.get(f"{BASE}/paper-insights", params=PARAMS)
        assert r.status_code == 200
        return r.json()

    def test_requires_internal_key_and_user(self, anon_api, api):
        assert anon_api.get(f"{BASE}/paper-insights", params=PARAMS).status_code == 401
        assert api.get(f"{BASE}/paper-insights").status_code == 422

    def test_shows_frequency_and_marks_when_enough_papers(self, api, exam):
        body = self.get(api)
        trees = by_topic(body)[T_TREES]

        assert body["reliable"] is True and body["warning"] is None
        assert body["total_ready_papers"] == 2                      # the 'processing' paper is ignored
        assert trees["reliable"] is True
        assert trees["papers_with_topic"] == 2 and trees["question_count"] == 2
        assert trees["frequency_percentage"] == 100.0
        assert trees["years"] == [2023, 2024]
        assert trees["marks"] == {"total": 16.0, "average": 8.0, "highest": 10.0,
                                  "questions_with_marks": 2, "questions_without_marks": 0}

    def test_partial_frequency_and_missing_marks_are_reported_honestly(self, api, exam):
        graphs = by_topic(self.get(api))[T_GRAPHS]

        assert graphs["frequency_percentage"] == 50.0
        assert graphs["marks"]["total"] is None and graphs["marks"]["average"] is None
        assert graphs["marks"]["questions_without_marks"] == 1

    def test_often_examined_uses_the_configured_threshold(self, api, exam):
        topics = by_topic(self.get(api))

        assert topics[T_TREES]["often_examined"] is True            # 100% >= threshold
        assert topics[T_GRAPHS]["often_examined"] == (50.0 >= settings.often_examined_frequency_percentage)
        assert topics[T_EMPTY]["often_examined"] is False

    def test_pending_and_unready_questions_do_not_count(self, api, exam):
        sorting = by_topic(self.get(api))[T_SORT]

        assert sorting["question_count"] == 0 and sorting["frequency_percentage"] == 0.0

    def test_topics_are_ranked_by_frequency(self, api, exam):
        order = [t["topic_id"] for t in self.get(api)["topics"]]

        assert order[0] == T_TREES and order[1] == T_GRAPHS

    def test_withholds_percentages_and_marks_when_too_few_papers(self, api, db, exam, monkeypatch):
        monkeypatch.setattr(settings, "min_papers_for_frequency", 3)
        body = self.get(api)

        assert body["reliable"] is False and body["minimum_papers_required"] == 3
        assert "Only 2 ready past paper(s)" in body["warning"]
        trees = by_topic(body)[T_TREES]
        assert trees["reliable"] is False
        assert trees["frequency_percentage"] is None and trees["often_examined"] is None
        assert trees["marks"] is None
        # plain facts are still shown
        assert trees["question_count"] == 2 and trees["papers_with_topic"] == 2

    def test_no_papers_at_all(self, api, world):
        body = self.get(api)

        assert body["reliable"] is False and body["total_ready_papers"] == 0
        assert "No past papers" in body["warning"]

    def test_other_users_and_courses_are_excluded(self, api, db, exam):
        qid, row = question(P_OTHER, "9", 50, 2024, user=OTHER_USER, course=OTHER_COURSE)
        db.seed("documents", paper(P_OTHER, 2024, user=OTHER_USER, course=OTHER_COURSE))
        db.seed("past_paper_questions", row)
        db.seed("question_topic_mappings", mapping(qid, T_TREES, user=OTHER_USER))

        body = self.get(api)

        assert body["total_ready_papers"] == 2
        assert by_topic(body)[T_TREES]["question_count"] == 2


# ================================================================ S-01
@pytest.fixture
def linked(db, exam):
    """Add passages, practice questions and mastery on top of the exam data."""
    doc = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    db.seed("documents", {"id": doc, "title": "Lecture 3", "document_type": "lecture_material",
                          "status": "ready", "user_id": USER, "course_id": COURSE})
    chunks = [new_id() for _ in range(3)]
    for i, cid in enumerate(chunks):
        db.seed("document_chunks", {"id": cid, "document_id": doc, "user_id": USER, "course_id": COURSE,
                                    "content": f"Passage {i} " + "x" * 400, "page_start": i + 1, "page_end": i + 1})
    db.seed("topic_chunk_links",
            *[{"topic_id": T_TREES, "chunk_id": c, "document_id": doc, "course_id": COURSE, "user_id": USER,
               "similarity": 0.5 + i / 10, "method": "ai" if i else "manual"} for i, c in enumerate(chunks)],
            {"topic_id": T_GRAPHS, "chunk_id": chunks[0], "document_id": doc, "course_id": COURSE,
             "user_id": USER, "similarity": 0.4, "method": "ai"})

    qs = [new_id(), new_id(), new_id()]
    db.seed("generated_questions",
            {"id": qs[0], "topic_id": T_TREES, "question_text": "Define a tree", "difficulty": "easy",
             "status": "active", "user_id": USER, "course_id": COURSE, "source_document_id": doc},
            {"id": qs[1], "topic_id": T_TREES, "question_text": "Height of a tree", "difficulty": "hard",
             "status": "active", "user_id": USER, "course_id": COURSE, "source_document_id": doc},
            {"id": qs[2], "topic_id": T_TREES, "question_text": "Archived one", "difficulty": "easy",
             "status": "archived", "user_id": USER, "course_id": COURSE, "source_document_id": doc})
    session = new_id()
    db.seed("practice_attempts",
            {"session_id": session, "question_id": qs[0], "user_id": USER, "is_correct": True},
            {"session_id": new_id(), "question_id": qs[0], "user_id": USER, "is_correct": False})
    db.seed("topic_progress", {"topic_id": T_TREES, "course_id": COURSE, "user_id": USER, "attempts": 2,
                               "correct_answers": 1, "accuracy_percentage": 50, "status": "Developing",
                               "last_practised_at": "2026-11-20T10:00:00+00:00"})
    return SimpleNamespace(chunks=chunks, questions=qs)


class TestKnowledgeMap:
    def get(self, api):
        r = api.get(f"{BASE}/knowledge-map", params=PARAMS)
        assert r.status_code == 200
        return r.json()

    def test_links_passages_past_questions_practice_and_mastery_per_topic(self, api, linked):
        trees = by_topic(self.get(api))[T_TREES]

        assert trees["passages"] == {"count": 3, "status": "enough"}
        assert trees["past_paper_questions"]["count"] == 2
        assert trees["practice_questions"]["count"] == 2            # archived question excluded
        assert trees["mastery"] == {"status": "Developing", "accuracy_percentage": 50, "attempts": 2,
                                    "last_practised_at": "2026-11-20T10:00:00+00:00"}

    def test_topics_with_nothing_show_the_gap(self, api, linked):
        sorting = by_topic(self.get(api))[T_SORT]

        assert sorting["passages"] == {"count": 0, "status": "no_material"}
        assert sorting["past_paper_questions"]["count"] == 0
        assert sorting["practice_questions"]["count"] == 0
        assert sorting["mastery"]["status"] == "Not Yet Assessed" and sorting["mastery"]["attempts"] == 0

    def test_thin_material_status(self, api, linked):
        assert by_topic(self.get(api))[T_GRAPHS]["passages"] == {"count": 1, "status": "thin"}

    def test_keeps_syllabus_order_and_hierarchy(self, api, linked):
        topics = self.get(api)["topics"]

        # the shared `world` fixture contributes one topic at position 1
        assert [t["topic_id"] for t in topics] == [world_topic(), T_TREES, T_GRAPHS, T_SORT, T_EMPTY]
        assert topics[4]["parent_topic_id"] == T_TREES

    def test_totals(self, api, linked):
        assert self.get(api)["totals"] == {
            "topics": 5, "topics_with_material": 2, "topics_with_past_questions": 2,
            "topics_with_practice": 1, "topics_assessed": 1}

    def test_empty_course(self, api, world):
        body = self.get(api)
        assert [t["topic_id"] for t in body["topics"]] == [world_topic()]    # the fixture's single topic
        assert body["totals"]["topics_with_material"] == 0

    def test_other_users_data_is_never_included(self, api, db, linked):
        db.seed("topic_chunk_links", {"topic_id": T_TREES, "chunk_id": new_id(), "document_id": new_id(),
                                      "course_id": COURSE, "user_id": OTHER_USER, "similarity": 0.9, "method": "ai"})
        db.seed("generated_questions", {"topic_id": T_TREES, "question_text": "Not mine", "difficulty": "easy",
                                        "status": "active", "user_id": OTHER_USER, "course_id": COURSE})

        trees = by_topic(self.get(api))[T_TREES]
        assert trees["passages"]["count"] == 3 and trees["practice_questions"]["count"] == 2


def world_topic():
    from tests.conftest import TOPIC
    return TOPIC


class TestTopicKnowledge:
    def get(self, api, topic_id=T_TREES, **params):
        return api.get(f"{BASE}/knowledge-map/topics/{topic_id}", params={**PARAMS, **params})

    def test_returns_sources_for_every_kind_of_evidence(self, api, linked):
        r = self.get(api)
        assert r.status_code == 200
        body = r.json()

        assert body["topic"]["title"] == "Trees"

        passages = body["passages"]
        assert passages["total"] == 3 and passages["status"] == "enough"
        assert [p["page_start"] for p in passages["items"]] == [3, 2, 1]     # best similarity first
        first = passages["items"][0]
        assert first["document_title"] == "Lecture 3" and first["method"] == "ai"
        assert len(first["excerpt"]) == 300

        past = body["past_paper_questions"]
        assert past["total"] == 2
        assert [q["paper_year"] for q in past["items"]] == [2024, 2023]      # newest first
        assert past["items"][0]["document_title"] == "Paper 2024" and past["items"][0]["page_number"] == 2

        practice = body["practice_questions"]
        assert practice["total"] == 2
        define = next(q for q in practice["items"] if q["question_text"] == "Define a tree")
        assert define["times_answered"] == 2 and define["times_correct"] == 1

        assert body["mastery"]["status"] == "Developing"

    def test_limit_caps_the_lists_but_not_the_totals(self, api, linked):
        body = self.get(api, limit=1).json()

        assert len(body["passages"]["items"]) == 1 and body["passages"]["total"] == 3
        assert len(body["past_paper_questions"]["items"]) == 1 and body["past_paper_questions"]["total"] == 2
        assert len(body["practice_questions"]["items"]) == 1 and body["practice_questions"]["total"] == 2

    def test_topic_with_nothing_linked(self, api, linked):
        body = self.get(api, T_SORT).json()

        assert body["passages"] == {"total": 0, "status": "no_material", "items": []}
        assert body["past_paper_questions"]["items"] == [] and body["practice_questions"]["items"] == []
        assert body["mastery"]["status"] == "Not Yet Assessed"

    def test_unknown_topic_or_someone_elses_topic_is_rejected(self, api, db, linked):
        foreign = new_id()
        db.seed("syllabus_topics", topic(foreign, "Secret", 1, user=OTHER_USER, course=OTHER_COURSE))

        assert self.get(api, new_id()).status_code == 400
        assert self.get(api, foreign).status_code == 400

    @pytest.mark.parametrize("limit", [0, 101, "x"])
    def test_limit_is_validated(self, api, linked, limit):
        assert self.get(api, limit=limit).status_code == 422
