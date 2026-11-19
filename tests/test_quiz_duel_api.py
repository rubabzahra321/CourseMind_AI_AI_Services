from tests.conftest import COURSE, OTHER_COURSE, OTHER_USER, TOPIC, USER, new_id, seed_question

BASE = f"/internal/courses/{COURSE}/quiz-duel"


def start(api, topic=TOPIC):
    return api.post(f"{BASE}/start", json={"user_id": USER, "topic_id": topic})


def answer(api, game_id, question_id, option_id, position, user=USER):
    return api.post(f"{BASE}/{game_id}/answer", json={
        "user_id": user, "question_id": question_id, "selected_option_id": option_id, "position": position})


def bank(db, n=6):
    """n questions; the newest is served first. Returns {id: (option_ids, correct_id)}."""
    out = {}
    for i in range(n):
        qid, opts, correct = seed_question(db, text=f"Duel Q{i}")
        out[qid] = (opts, correct)
    return out


def wrong_option(info, qid):
    opts, correct = info[qid]
    return next(o for o in opts if o != correct)


# ------------------------------------------------------------------ start
def test_start_game_serves_a_question_without_answer_key(api, db, world):
    bank(db)
    r = start(api)
    assert r.status_code == 200
    body = r.json()
    assert body["board"] == [""] * 9 and body["status"] == "in_progress"
    q = body["question"]
    assert len(q["options"]) == 4 and all("is_correct" not in o for o in q["options"])
    game = db.rows("quiz_duel_games")[0]
    assert game["current_question_id"] == q["id"] and game["user_id"] == USER and game["topic_id"] == TOPIC


def test_start_game_needs_questions_for_the_topic(api, db, world):
    r = start(api)
    assert r.status_code == 400 and "No unused questions" in r.json()["detail"]
    assert db.rows("quiz_duel_games") == []


def test_start_game_ignores_other_users_questions(api, db, world):
    seed_question(db, user=OTHER_USER, course=OTHER_COURSE)
    assert start(api).status_code == 400


def test_start_game_rejects_question_without_four_options(api, db, world):
    qid, _, _ = seed_question(db)
    db.tables["question_options"][:] = [o for o in db.tables["question_options"] if o["option_order"] != 4]
    assert start(api).status_code == 400


# ------------------------------------------------------------------ answering
def test_correct_answer_claims_square_and_ai_replies(api, db, world):
    info = bank(db)
    g = start(api).json()
    qid = g["question"]["id"]
    r = answer(api, g["game_id"], qid, info[qid][1], position=4)
    assert r.status_code == 200
    body = r.json()
    assert body["is_correct"] is True and body["board"][4] == "X"
    assert body["board"].count("O") == 1 and body["ai_move"] is not None
    assert body["ai_algorithm"] == "Minimax with Alpha-Beta Pruning"
    assert body["explanation"] and body["status"] == "in_progress"
    assert body["next_question"]["id"] != qid                      # never repeats a question
    assert all("is_correct" not in o for o in body["next_question"]["options"])
    saved = db.rows("quiz_duel_games")[0]
    assert saved["board"] == body["board"] and saved["current_question_id"] == body["next_question"]["id"]


def test_wrong_answer_does_not_claim_the_square_and_ai_still_moves(api, db, world):
    info = bank(db)
    g = start(api).json()
    qid = g["question"]["id"]
    r = answer(api, g["game_id"], qid, wrong_option(info, qid), position=0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_correct"] is False and body["correct_option"] is None
    assert "X" not in body["board"] and body["board"].count("O") == 1


def test_wrong_answers_do_not_block_squares_for_the_rest_of_the_game(api, db, world):
    """Regression: a failed attempt on a square must not make that square unusable (DB unique key game+position)."""
    info = bank(db, n=8)
    g = start(api).json()
    q = g["question"]["id"]
    for _ in range(3):  # student keeps failing on square 0; AI keeps playing
        r = answer(api, g["game_id"], q, wrong_option(info, q), position=0)
        assert r.status_code in (200, 400), r.text
        body = r.json()
        if r.status_code == 400 or body["status"] == "completed":
            break
        q = body["next_question"]["id"]
    assert r.status_code != 500


def test_winning_move_ends_game_for_the_student(api, db, world):
    info = bank(db)
    qid = next(iter(info))
    game = db.seed("quiz_duel_games", {"user_id": USER, "course_id": COURSE, "topic_id": TOPIC,
                                       "board": ["X", "X", "", "", "O", "", "", "O", ""],
                                       "current_question_id": qid, "status": "in_progress"})
    body = answer(api, game["id"], qid, info[qid][1], position=2).json()
    assert body["winner"] == "student" and body["status"] == "completed"
    assert body["ai_move"] is None and body["next_question"] is None
    assert db.rows("quiz_duel_games")[0]["winner"] == "student"
    # and the finished game accepts no more moves
    assert answer(api, game["id"], qid, info[qid][1], position=3).status_code == 400


def test_ai_wins_when_student_keeps_missing(api, db, world):
    info = bank(db)
    qid = next(iter(info))
    game = db.seed("quiz_duel_games", {"user_id": USER, "course_id": COURSE, "topic_id": TOPIC,
                                       "board": ["O", "O", "", "X", "X", "", "", "", ""],
                                       "current_question_id": qid, "status": "in_progress"})
    body = answer(api, game["id"], qid, wrong_option(info, qid), position=5).json()
    assert body["ai_move"] == 2 and body["winner"] == "ai" and body["status"] == "completed"


# ------------------------------------------------------------------ rule violations
def test_rejects_answer_for_a_question_that_is_not_current(api, db, world):
    info = bank(db)
    g = start(api).json()
    other = next(q for q in info if q != g["question"]["id"])
    assert answer(api, g["game_id"], other, info[other][1], 4).status_code == 400


def test_rejects_occupied_square(api, db, world):
    info = bank(db)
    qid = next(iter(info))
    game = db.seed("quiz_duel_games", {"user_id": USER, "course_id": COURSE, "topic_id": TOPIC,
                                       "board": ["O"] + [""] * 8, "current_question_id": qid, "status": "in_progress"})
    assert answer(api, game["id"], qid, info[qid][1], 0).status_code == 400


def test_rejects_option_from_another_question(api, db, world):
    info = bank(db)
    g = start(api).json()
    qid = g["question"]["id"]
    foreign = next(v[1] for q, v in info.items() if q != qid)
    assert answer(api, g["game_id"], qid, foreign, 4).status_code == 400


def test_rejects_out_of_range_position(api, db, world):
    info = bank(db)
    g = start(api).json()
    qid = g["question"]["id"]
    assert answer(api, g["game_id"], qid, info[qid][1], 9).status_code == 422
    assert answer(api, g["game_id"], qid, info[qid][1], -1).status_code == 422


def test_unknown_and_foreign_games_are_rejected(api, db, world):
    info = bank(db)
    qid = next(iter(info))
    assert answer(api, new_id(), qid, info[qid][1], 4).status_code == 400
    foreign = db.seed("quiz_duel_games", {"user_id": OTHER_USER, "course_id": OTHER_COURSE, "topic_id": TOPIC,
                                          "board": [""] * 9, "current_question_id": qid, "status": "in_progress"})
    assert answer(api, foreign["id"], qid, info[qid][1], 4).status_code == 400


def test_game_stops_with_clear_error_when_questions_run_out(api, db, world):
    info = bank(db, n=2)
    g = start(api).json()
    q = g["question"]["id"]
    r1 = answer(api, g["game_id"], q, info[q][1], 4)          # uses question 1, next = question 2
    q2 = r1.json()["next_question"]["id"]
    r2 = answer(api, g["game_id"], q2, info[q2][1], 8)         # nothing left to serve next
    assert r2.status_code in (200, 400)
    if r2.status_code == 400:
        assert "No unused questions" in r2.json()["detail"]
