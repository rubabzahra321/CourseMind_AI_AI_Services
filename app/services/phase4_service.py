
from app.db.client import get_db
from app.repositories import phase4_repository as repo


def get_attempt_history(user_id, course_id, limit=50):
    sessions = repo.list_practice_sessions(
        user_id,
        course_id,
        limit,
    )

    output = []

    for session in sessions:
        attempts = repo.get_session_attempts(
            session["id"],
            user_id,
        )

        topic = repo.get_topic_title(
            session.get("topic_id"),
            user_id,
            course_id,
        )

        for attempt in attempts:
            question = None

            if attempt.get("question_id"):
                question = repo.get_generated_question(
                    attempt["question_id"],
                    user_id,
                    course_id,
                )

            paper_question = None

            if attempt.get("past_paper_question_id"):
                paper_question = repo.get_past_paper_question(
                    attempt["past_paper_question_id"],
                    user_id,
                    course_id,
                )

            output.append({
                "session_id": session["id"],
                "session_status": session["status"],
                "topic": topic,
                "attempt": attempt,
                "generated_question": question,
                "past_paper_question": paper_question,
            })

    return output


def get_incorrect_questions(user_id, course_id, limit=100):
    return repo.list_incorrect_attempts(
        user_id,
        course_id,
        limit,
    )


def start_past_paper_practice(
    user_id,
    course_id,
    topic_id=None,
    limit=20,
):
    questions = repo.list_reviewed_paper_questions(
        user_id,
        course_id,
        topic_id,
        limit,
    )

    if not questions:
        raise ValueError(
            "No reviewed past-paper questions are available for this selection"
        )

    session = repo.create_practice_session(
        user_id,
        course_id,
        topic_id,
        len(questions),
    )

    rows = []

    for index, question in enumerate(questions, start=1):
        rows.append({
            "session_id": session["id"],
            "question_id": None,
            "past_paper_question_id": question["id"],
            "user_id": user_id,
            "question_order": index,
        })

    repo.add_session_questions(rows)

    return {
        "session_id": session["id"],
        "source": "reviewed_past_paper",
        "total_questions": len(questions),
        "questions": [
            {
                "id": question["id"],
                "question_number": question.get("question_number"),
                "question_text": question["question_text"],
                "question_type": question["question_type"],
                "marks": question.get("marks"),
                "paper_year": question.get("paper_year"),
                "page_number": question.get("page_number"),
                "source_document": question.get("source_document"),
                "topic_mappings": question.get("topic_mappings", []),
            }
            for question in questions
        ],
    }


def submit_past_paper_answer(
    user_id,
    course_id,
    session_id,
    question_id,
    text_answer,
    response_time_seconds=None,
):
    session = repo.get_practice_session(
        session_id,
        user_id,
        course_id,
    )

    if not session:
        raise ValueError("Practice session not found")

    if session["status"] != "in_progress":
        raise ValueError("Practice session is not active")

    question = repo.get_past_paper_question(
        question_id,
        user_id,
        course_id,
    )

    if not question:
        raise ValueError(
            "Reviewed past-paper question not found"
        )

    if not repo.is_paper_question_assigned(
        session_id,
        question_id,
        user_id,
    ):
        raise ValueError(
            "Question does not belong to this practice session"
        )

    if repo.has_paper_attempt(session_id, question_id):
        raise ValueError(
            "This question has already been answered"
        )

    if not text_answer or not text_answer.strip():
        raise ValueError("Answer cannot be empty")

    attempt = repo.save_paper_attempt(
        session_id=session_id,
        question_id=question_id,
        user_id=user_id,
        text_answer=text_answer.strip(),
        response_time_seconds=response_time_seconds,
    )

    topic_ids = repo.get_paper_topic_ids(
        question_id,
        user_id,
    )

    follow_up = None

    if topic_ids:
        follow_up = repo.create_follow_up_task(
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_ids[0],
            session_id=session_id,
            attempt_id=attempt["id"],
            title="Review your past-paper answer",
            description=(
                "Compare your answer with your instructor's marking "
                "scheme and revisit the mapped topic."
            ),
            task_type="revisit_mistake",
        )

    return {
        "attempt_id": attempt["id"],
        "saved": True,
        "automatically_marked": False,
        "message": (
            "Your answer has been saved. This past-paper question "
            "does not have an official answer key in the database, "
            "so CourseMind has not assigned a correctness score."
        ),
        "source": {
            "document_id": question["document_id"],
            "page_number": question.get("page_number"),
        },
        "topic_ids": topic_ids,
        "follow_up_task": follow_up,
    }


def get_session_report(user_id, course_id, session_id):
    session = repo.get_practice_session(
        session_id,
        user_id,
        course_id,
    )

    if not session:
        raise ValueError("Practice session not found")

    attempts = repo.get_session_attempts(
        session_id,
        user_id,
    )

    total = session.get("total_questions", 0)
    answered = len(attempts)

    scored_attempts = [
        item for item in attempts
        if item.get("is_correct") is not None
    ]

    correct = sum(
        1 for item in scored_attempts
        if item["is_correct"]
    )

    incorrect = sum(
        1 for item in scored_attempts
        if item["is_correct"] is False
    )

    ungraded = answered - len(scored_attempts)

    accuracy = (
        round(correct / len(scored_attempts) * 100, 2)
        if scored_attempts else None
    )

    total_time = sum(
        item.get("response_time_seconds") or 0
        for item in attempts
    )

    topic_breakdown = {}
    mistakes = []

    for attempt in attempts:
        question = None
        topic_id = None
        question_text = None

        if attempt.get("question_id"):
            question = repo.get_generated_question(
                attempt["question_id"],
                user_id,
                course_id,
            )

            if question:
                topic_id = question.get("topic_id")
                question_text = question.get("question_text")

        elif attempt.get("past_paper_question_id"):
            question = repo.get_past_paper_question(
                attempt["past_paper_question_id"],
                user_id,
                course_id,
            )

            if question:
                question_text = question.get("question_text")

        topic = repo.get_topic_title(
            topic_id,
            user_id,
            course_id,
        )

        topic_name = topic["title"] if topic else "Unmapped topic"

        if topic_name not in topic_breakdown:
            topic_breakdown[topic_name] = {
                "answered": 0,
                "correct": 0,
                "incorrect": 0,
                "ungraded": 0,
            }

        topic_breakdown[topic_name]["answered"] += 1

        if attempt.get("is_correct") is True:
            topic_breakdown[topic_name]["correct"] += 1
        elif attempt.get("is_correct") is False:
            topic_breakdown[topic_name]["incorrect"] += 1
        else:
            topic_breakdown[topic_name]["ungraded"] += 1

        if attempt.get("is_correct") is False:
            mistakes.append({
                "attempt_id": attempt["id"],
                "question_text": question_text,
                "topic": topic_name,
                "feedback": attempt.get("evaluation_feedback"),
                "text_answer": attempt.get("text_answer"),
            })

    tasks = repo.get_follow_up_tasks(
        session_id,
        user_id,
    )

    return {
        "session": session,
        "score": {
            "total_questions": total,
            "answered": answered,
            "correct": correct,
            "incorrect": incorrect,
            "ungraded": ungraded,
            "accuracy_percentage": accuracy,
        },
        "time": {
            "total_response_time_seconds": total_time,
        },
        "topic_breakdown": topic_breakdown,
        "mistakes": mistakes,
        "follow_up_tasks": tasks,
    }

import json
import re

from app.ai.gemini import generate_answer
from app.repositories import practice_repository
from app.services import practice_service


def _extract_json(text):
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end < start:
        raise ValueError("AI returned an invalid question format")

    return json.loads(cleaned[start:end + 1])


def generate_typed_questions(
    user_id,
    course_id,
    topic_id,
    document_id,
    question_type,
    difficulty,
    number_of_questions,
):
    from app.repositories import question_repository
    from app.repositories import phase4_repository as repo

    allowed_types = {
        "true_false",
        "short_answer",
        "long_answer",
    }

    if question_type not in allowed_types:
        raise ValueError("Unsupported question type")

    topic = question_repository.get_topic(
        user_id,
        course_id,
        topic_id,
    )
    if not topic:
        raise ValueError("Topic not found in this course")

    document = question_repository.get_document(
        user_id,
        course_id,
        document_id,
    )
    if not document:
        raise ValueError("Ready source document not found")

    chunks = question_repository.get_document_chunks(
        user_id,
        course_id,
        document_id,
    )
    if not chunks:
        raise ValueError("Source document has no extracted text")

    material = "\n\n".join(
        f"Page {chunk.get('page_start')}:\n{chunk['content']}"
        for chunk in chunks
    )

    material = material[:18000]

    prompt = f"""
You are generating university practice questions for CourseMind AI.

Use ONLY the course material below.

Topic: {topic['title']}
Question type: {question_type}
Difficulty: {difficulty}
Number of questions: {number_of_questions}

Return valid JSON only, using this exact structure:
{{
  "questions": [
    {{
      "question_text": "Question text",
      "expected_answer": "Correct answer or model answer",
      "answer_rubric": "How the answer should be marked",
      "max_marks": 2,
      "options": [
        {{
          "option_text": "True",
          "is_correct": true
        }},
        {{
          "option_text": "False",
          "is_correct": false
        }}
      ]
    }}
  ]
}}

Rules:
- For true_false, provide exactly two options: True and False.
- For short_answer and long_answer, provide an empty options array.
- The expected answer must be supported by the material.
- The rubric must explain the marking criteria.
- Do not include markdown or text outside the JSON object.

COURSE MATERIAL:
{material}
"""

    generated = _extract_json(generate_answer(prompt))
    questions = generated.get("questions")

    if not isinstance(questions, list) or not questions:
        raise ValueError("AI did not generate any valid questions")

    saved = []

    for item in questions[:number_of_questions]:
        question_text = str(item.get("question_text", "")).strip()
        expected_answer = str(item.get("expected_answer", "")).strip()
        rubric = str(item.get("answer_rubric", "")).strip()

        if not question_text or not expected_answer:
            continue

        marks = float(item.get("max_marks") or 1)

        source_page = chunks[0].get("page_start")

        row = repo.save_typed_question(
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_id,
            document_id=document_id,
            question_text=question_text,
            question_type=question_type,
            difficulty=difficulty,
            expected_answer=expected_answer,
            answer_rubric=rubric,
            max_marks=marks,
            source_page_start=source_page,
            source_page_end=source_page,
        )

        options = item.get("options") or []

        if question_type == "true_false":
            if len(options) != 2 or sum(
                1 for option in options if option.get("is_correct")
            ) != 1:
                raise ValueError(
                    "AI generated invalid true/false options"
                )

            repo.save_typed_options(row["id"], options)

        saved.append({
            "id": row["id"],
            "question_text": row["question_text"],
            "question_type": row["question_type"],
            "difficulty": row["difficulty"],
            "max_marks": row["max_marks"],
            "source_document_id": document_id,
            "source_page_start": row.get("source_page_start"),
        })

    if not saved:
        raise ValueError("No valid questions could be saved")

    return {
        "question_type": question_type,
        "difficulty": difficulty,
        "generated_count": len(saved),
        "questions": saved,
    }


def submit_generated_text_answer(
    user_id,
    course_id,
    session_id,
    question_id,
    text_answer,
    response_time_seconds=None,
):
    from app.repositories import phase4_repository as repo

    session = repo.get_practice_session(
        session_id,
        user_id,
        course_id,
    )
    if not session:
        raise ValueError("Practice session not found")

    if session["status"] != "in_progress":
        raise ValueError("Practice session is not active")

    if not repo.is_generated_question_assigned(
        session_id,
        question_id,
        user_id,
    ):
        raise ValueError("Question does not belong to this session")

    if practice_repository.has_attempt(session_id, question_id):
        raise ValueError("This question has already been answered")

    question = repo.get_generated_question(
        question_id,
        user_id,
        course_id,
    )
    if not question:
        raise ValueError("Question not found")

    if question["question_type"] not in (
        "short_answer",
        "long_answer",
    ):
        raise ValueError(
            "Text answers are only accepted for short and long questions"
        )

    if not text_answer or not text_answer.strip():
        raise ValueError("Answer cannot be empty")

    prompt = f"""
You are marking a student's university practice answer.

Use the expected answer and rubric as the marking reference.
Do not introduce facts that are not in the reference.

Question:
{question['question_text']}

Expected answer:
{question.get('expected_answer')}

Marking rubric:
{question.get('answer_rubric')}

Maximum marks:
{question.get('max_marks')}

Student answer:
{text_answer}

Return valid JSON only:
{{
  "awarded_marks": 0,
  "is_correct": false,
  "feedback": "Short explanation of the marking"
}}

Rules:
- awarded_marks must be between zero and maximum marks.
- is_correct is true only if the answer is substantially correct.
- Give constructive, student-friendly feedback.
"""

    evaluation = _extract_json(generate_answer(prompt))

    maximum = float(question.get("max_marks") or 1)
    awarded = float(evaluation.get("awarded_marks") or 0)
    awarded = max(0, min(maximum, awarded))
    is_correct = bool(evaluation.get("is_correct"))

    attempt = repo.save_text_attempt(
        session_id=session_id,
        question_id=question_id,
        user_id=user_id,
        text_answer=text_answer.strip(),
        response_time_seconds=response_time_seconds,
        is_correct=is_correct,
        awarded_marks=awarded,
        max_marks=maximum,
        evaluation_feedback=str(
            evaluation.get("feedback") or ""
        ),
    )

    topic_id = question.get("topic_id")
    follow_up = None

    if not is_correct:
        follow_up = repo.create_follow_up_task(
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_id,
            session_id=session_id,
            attempt_id=attempt["id"],
            title="Revisit an incorrect answer",
            description=(
                "Review the source material and expected answer, "
                "then try explaining this concept again."
            ),
            task_type="revisit_mistake",
        )

    # Recalculate session score using only scored attempts.
    attempts = repo.get_session_attempts(session_id, user_id)
    scored = [
        item for item in attempts
        if item.get("is_correct") is not None
    ]
    correct = sum(
        1 for item in scored if item["is_correct"]
    )
    accuracy = (
        round(correct / len(scored) * 100, 2)
        if scored else 0
    )

    repo.update_practice_session(
        session_id,
        {
            "correct_answers": correct,
            "accuracy_percentage": accuracy,
        },
    )

    source_document = repo.get_document(
        question.get("source_document_id"),
        user_id,
        course_id,
    )

    return {
        "attempt_id": attempt["id"],
        "is_correct": is_correct,
        "awarded_marks": awarded,
        "max_marks": maximum,
        "feedback": evaluation.get("feedback"),
        "expected_answer": question.get("expected_answer"),
        "source": {
            "document": source_document,
            "page_start": question.get("source_page_start"),
            "page_end": question.get("source_page_end"),
        },
        "follow_up_task": follow_up,
    }


def submit_generated_mcq_with_sources(
    user_id,
    course_id,
    session_id,
    question_id,
    selected_option_id,
    response_time_seconds=None,
):
    from app.repositories import phase4_repository as repo

    result = practice_service.submit_answer(
        user_id=user_id,
        course_id=course_id,
        session_id=session_id,
        question_id=question_id,
        selected_option_id=selected_option_id,
        response_time_seconds=response_time_seconds,
    )

    question = repo.get_generated_question(
        question_id,
        user_id,
        course_id,
    )

    if not question:
        return result

    document = repo.get_document(
        question.get("source_document_id"),
        user_id,
        course_id,
    )

    source_chunks = repo.get_question_source_chunks(
        question.get("source_document_id"),
        user_id,
        course_id,
        question.get("source_page_start"),
    ) if question.get("source_document_id") else []

    source = {
        "document": document,
        "page_start": question.get("source_page_start"),
        "page_end": question.get("source_page_end"),
        "chunks": [
            {
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "excerpt": chunk.get("content", "")[:500],
            }
            for chunk in source_chunks
        ],
    }

    follow_up = None

    if not result.get("is_correct"):
        follow_up = repo.create_follow_up_task(
            user_id=user_id,
            course_id=course_id,
            topic_id=question.get("topic_id"),
            session_id=session_id,
            attempt_id=result.get("attempt_id"),
            title="Revise an incorrect practice question",
            description=(
                "Read the linked source passage and explanation, "
                "then attempt a similar question."
            ),
            task_type="revisit_mistake",
        )

    result["source"] = source
    result["follow_up_task"] = follow_up
    result["topic_id"] = question.get("topic_id")
    result["explanation"] = question.get("explanation")

    return result
