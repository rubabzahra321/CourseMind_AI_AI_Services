import json
from google import genai
from google.genai import types

from app.core.config import settings


client = genai.Client(
    api_key=settings.gemini_api_key
)


def generate_mcqs(
    topic_title: str,
    difficulty: str,
    number_of_questions: int,
    material: str,
) -> list[dict]:

    prompt = f"""
You are an academic MCQ generator for university students.

Generate exactly {number_of_questions} multiple-choice questions.

Topic: {topic_title}
Difficulty: {difficulty}

Use ONLY the supplied study material.

Rules:
1. Every question must be answerable from the supplied material.
2. Each question must have exactly four options.
3. Only one option must be correct.
4. Incorrect options must be plausible but clearly incorrect.
5. Do not create duplicate questions.
6. Do not invent facts outside the material.
7. Explanations must explain why the correct option is correct.
8. Avoid questions that test information absent from the material.

Return valid JSON with this structure:

{{
  "questions": [
    {{
      "question_text": "Question?",
      "explanation": "Explanation",
      "options": [
        {{
          "option_text": "Option A",
          "is_correct": false
        }},
        {{
          "option_text": "Option B",
          "is_correct": true
        }},
        {{
          "option_text": "Option C",
          "is_correct": false
        }},
        {{
          "option_text": "Option D",
          "is_correct": false
        }}
      ]
    }}
  ]
}}

Study material:
{material}
"""

    response = client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=6000,
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

        options = question.get("options", [])

        if len(options) != 4:
            raise ValueError("Each MCQ must have four options")

        if sum(bool(o.get("is_correct")) for o in options) != 1:
            raise ValueError(
                "Each MCQ must have exactly one correct option"
            )

        if any(not o.get("option_text") for o in options):
            raise ValueError("An option is empty")

    return questions