import json

from google import genai
from google.genai import types

from app.core.config import settings


client = genai.Client(
    api_key=settings.gemini_api_key
)


SUPPORTED_TYPES = {
    "mcq",
    "true_false",
    "short_answer",
    "long_answer",
}


def generate_practice_questions(
    topic_title,
    question_type,
    difficulty,
    number_of_questions,
    material,
):
    if question_type not in SUPPORTED_TYPES:
        raise ValueError("Unsupported question type")

    if question_type == "mcq":
        format_instructions = """
Each question must contain exactly four options.
Only one option must be correct.
Each option must contain:
- option_text
- is_correct
"""

    elif question_type == "true_false":
        format_instructions = """
Each question must be a statement that can be judged true or false.
Provide exactly two options:
- True
- False
Exactly one option must be correct.
Each option must contain:
- option_text
- is_correct
"""

    elif question_type == "short_answer":
        format_instructions = """
Create short-answer questions.
Do not provide options.
Each question must contain:
- expected_answer: a concise model answer
- answer_rubric: key points that a correct answer should contain
- max_marks: a positive number
"""

    else:
        format_instructions = """
Create long-answer questions suitable for university students.
Do not provide options.
Each question must contain:
- expected_answer: a model answer with the main points
- answer_rubric: marking criteria and important points
- max_marks: a positive number
"""

    prompt = f"""
You are an academic practice-question generator.

Generate exactly {number_of_questions} questions.

Topic: {topic_title}
Question type: {question_type}
Difficulty: {difficulty}

Use ONLY the supplied study material.

Rules:
1. Questions must be answerable from the supplied material.
2. Do not invent facts outside the supplied material.
3. Do not create duplicate questions.
4. Use clear university-level language.
5. Explanations must be supported by the supplied material.
6. Follow the question-format instructions exactly.

Format instructions:
{format_instructions}

Return valid JSON in this structure:

{{
    "questions": [
        {{
            "question_text": "Question text",
            "explanation": "Explanation based on the material",
            "options": [],
            "expected_answer": null,
            "answer_rubric": null,
            "max_marks": null
        }}
    ]
}}

For MCQ and true_false, fill options and set written-answer fields to null.

For short_answer and long_answer, use an empty options array and fill
expected_answer, answer_rubric, and max_marks.

Study material:
{material}
"""

    response = client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=10000,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    data = json.loads(response.text)
    questions = data.get("questions", [])

    if len(questions) != number_of_questions:
        raise ValueError(
            "Gemini returned an unexpected number of questions"
        )

    for question in questions:
        if not question.get("question_text"):
            raise ValueError("Question text is missing")

        if not question.get("explanation"):
            raise ValueError("Question explanation is missing")

        options = question.get("options") or []

        if question_type == "mcq":
            if len(options) != 4:
                raise ValueError("Each MCQ must have four options")

            if sum(bool(option.get("is_correct")) for option in options) != 1:
                raise ValueError(
                    "Each MCQ must have exactly one correct option"
                )

        elif question_type == "true_false":
            if len(options) != 2:
                raise ValueError(
                    "True/false questions must have two options"
                )

            option_texts = {
                option.get("option_text", "").strip().lower()
                for option in options
            }

            if option_texts != {"true", "false"}:
                raise ValueError(
                    "True/false options must be True and False"
                )

            if sum(bool(option.get("is_correct")) for option in options) != 1:
                raise ValueError(
                    "Exactly one true/false option must be correct"
                )

        else:
            if options:
                raise ValueError(
                    "Written questions must not contain options"
                )

            if not question.get("expected_answer"):
                raise ValueError(
                    "Written question expected answer is missing"
                )

            if not question.get("answer_rubric"):
                raise ValueError(
                    "Written question marking rubric is missing"
                )

            if not isinstance(question.get("max_marks"), (int, float)):
                raise ValueError(
                    "Written question max_marks must be numeric"
                )

            if question["max_marks"] <= 0:
                raise ValueError(
                    "Written question max_marks must be positive"
                )

        for option in options:
            if not option.get("option_text"):
                raise ValueError("An option is empty")

    return questions