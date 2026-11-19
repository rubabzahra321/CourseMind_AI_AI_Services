
from uuid import UUID

from app.db.client import get_db
from app.db.helpers import data_of
from app.search.algorithms import minimax


def _get_question(user_id, course_id, topic_id, game_id=None):
    db = get_db()

    query = (
        db.table("generated_questions")
        .select("id,question_text,topic_id")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("topic_id", str(topic_id))
        .eq("status", "active")
        .order("created_at", desc=True)
    )

    questions = query.execute().data or []

    used_ids = set()

    if game_id:
        moves = (
            db.table("quiz_duel_moves")
            .select("question_id")
            .eq("game_id", str(game_id))
            .execute()
            .data or []
        )

        used_ids = {
            move["question_id"]
            for move in moves
            if move.get("question_id")
        }

    question = next(
        (
            item for item in questions
            if item["id"] not in used_ids
        ),
        None,
    )

    if not question:
        raise ValueError(
            "No unused questions remain. Generate more MCQs for this topic."
        )

    options = (
        db.table("question_options")
        .select("id,option_text,option_order")
        .eq("question_id", question["id"])
        .order("option_order")
        .execute()
        .data or []
    )

    if len(options) != 4:
        raise ValueError(
            "The selected question does not have four options."
        )

    return {
        **question,
        "options": options,
    }


def _get_game(user_id, course_id, game_id):
    result = (
        get_db().table("quiz_duel_games")
        .select("*")
        .eq("id", str(game_id))
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .maybe_single()
        .execute()
    )

    return data_of(result)


def _winner(board):
    lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]

    for a, b, c in lines:
        if board[a] and board[a] == board[b] == board[c]:
            return "student" if board[a] == "X" else "ai"

    if all(board):
        return "draw"

    return None


def _public_question(question):
    return {
        "id": question["id"],
        "question_text": question["question_text"],
        "options": question["options"],
    }


def start_game(user_id: UUID, course_id: UUID, topic_id: UUID):
    db = get_db()

    # Get the first question for the selected topic.
    question = _get_question(
        user_id,
        course_id,
        topic_id,
    )

    board = [""] * 9

    game = (
        db.table("quiz_duel_games")
        .insert({
            "user_id": str(user_id),
            "course_id": str(course_id),
            "topic_id": str(topic_id),
            "board": board,
            "current_question_id": question["id"],
            "status": "in_progress",
        })
        .execute()
        .data[0]
    )

    return {
        "game_id": game["id"],
        "board": board,
        "status": game["status"],
        "question": _public_question(question),
        "instructions": (
            "Answer correctly to claim a square. "
            "The AI plays after your turn."
        ),
    }


def submit_game_answer(
    user_id: UUID,
    course_id: UUID,
    game_id: UUID,
    question_id: UUID,
    selected_option_id: UUID,
    position: int,
):
    db = get_db()

    if position < 0 or position > 8:
        raise ValueError("Board position must be between 0 and 8.")

    game = _get_game(user_id, course_id, game_id)

    if not game:
        raise ValueError("Game not found.")

    if game["status"] != "in_progress":
        raise ValueError("This game has already finished.")

    if game["current_question_id"] != str(question_id):
        raise ValueError("This is not the current question for this game.")

    board = list(game["board"])

    if board[position] != "":
        raise ValueError("That square is already occupied.")

    question = data_of(
        db.table("generated_questions")
        .select("id,explanation")
        .eq("id", str(question_id))
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .maybe_single()
        .execute()
    )

    if not question:
        raise ValueError("Question not found.")

    option = data_of(
        db.table("question_options")
        .select("id,is_correct,option_text")
        .eq("id", str(selected_option_id))
        .eq("question_id", str(question_id))
        .maybe_single()
        .execute()
    )

    if not option:
        raise ValueError(
            "Selected option does not belong to this question."
        )

    is_correct = bool(option["is_correct"])

    # Only a CORRECT answer occupies a square. (quiz_duel_moves is UNIQUE(game_id, position),
    # so storing a failed attempt on that square would make the AI's move crash.)
    # A failed question still cannot repeat: the AI move row below carries its question_id.
    moves = []

    if is_correct:
        board[position] = "X"
        moves.append({
            "game_id": str(game_id),
            "user_id": str(user_id),
            "position": position,
            "player": "student",
            "question_id": str(question_id),
            "is_correct": True,
        })

    result = _winner(board)

    ai_position = None

    if result is None:
        _, ai_position = minimax(board, True)

        if ai_position is not None:
            board[ai_position] = "O"

            moves.append({
                "game_id": str(game_id),
                "user_id": str(user_id),
                "position": ai_position,
                "player": "ai",
                "question_id": str(question_id),
                "is_correct": None,
            })

        result = _winner(board)

    # Store student and AI moves.
    if moves:
        db.table("quiz_duel_moves").insert(moves).execute()

    status = "completed" if result else "in_progress"

    next_question = None
    next_question_id = None

    if not result:
        # Exclude questions already answered in this game.
        next_question = _get_question(
            user_id,
            course_id,
            game["topic_id"],
            game_id,
        )

        next_question_id = next_question["id"]

    # Update the game state.
    db.table("quiz_duel_games").update({
        "board": board,
        "status": status,
        "winner": result,
        "current_question_id": next_question_id,
    }).eq(
        "id", str(game_id)
    ).eq(
        "user_id", str(user_id)
    ).eq(
        "course_id", str(course_id)
    ).execute()

    return {
        "game_id": str(game_id),
        "board": board,
        "is_correct": is_correct,
        "correct_option": (
            option["option_text"] if is_correct else None
        ),
        "explanation": question["explanation"],
        "ai_algorithm": "Minimax with Alpha-Beta Pruning",
        "ai_move": ai_position,
        "status": status,
        "winner": result,
        "next_question": (
            _public_question(next_question)
            if next_question else None
        ),
    }
