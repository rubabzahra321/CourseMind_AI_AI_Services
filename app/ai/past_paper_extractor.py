
import json

from google.genai import types

from app.ai.gemini import get_gemini_client
from app.core.config import settings
from app.core.errors import PermanentJobError


def extract_past_paper_questions(
    pages: list[tuple[int, str]],
    academic_year: int | None = None
) -> list[dict]:

    material = "\n\n".join(
        f"[PDF PAGE {number}]\n{text}"
        for number, text in pages if text.strip()
    )

    if not material.strip():
        raise PermanentJobError("No readable past-paper text was found.")

    prompt = f'''Extract every exam question from this past paper. Return JSON only:
{{"questions":[{{"question_number":"Q1(a)","question_text":"Full question text","question_type":"short","marks":5,"paper_year":2024,"page_number":1,"confidence":0.95}}]}}
Rules:
- Include question number, complete text, marks when printed, paper year when identifiable, and PDF page number.
- question_type must be one of short, long, mcq, numerical, unknown.
- marks must be a number or null; year must be an integer or null.
- Do not invent missing marks/year. Use the supplied academic year {academic_year!r} only when the paper itself does not state a year.
- Include sub-questions as separate records only when they are independently answerable/marked; preserve numbering (e.g. 2(a)).
- Do not include instructions, headers or answer keys as questions.
- confidence must be between 0 and 1.
- Do not include markdown fences or commentary.

PAST PAPER TEXT:
{material}'''

    response = get_gemini_client().models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=12000,
        ),
    )

    try:
        payload = json.loads(response.text or "")
        questions = payload["questions"]

        if not isinstance(questions, list):
            raise ValueError("questions must be a list")

        cleaned = []

        for item in questions:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("question_text"), str)
                or not item["question_text"].strip()
            ):
                raise ValueError("Question text is missing")

            kind = item.get("question_type", "unknown")

            if kind not in {
                "short", "long", "mcq", "numerical", "unknown"
            }:
                kind = "unknown"

            marks = item.get("marks")

            if not isinstance(marks, (int, float)) or marks < 0:
                marks = None

            year = item.get("paper_year")

            if not isinstance(year, int) or not 1900 <= year <= 2200:
                year = academic_year

            page = item.get("page_number")

            if not isinstance(page, int) or page < 1:
                page = None

            confidence = item.get("confidence")

            if (
                not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                confidence = None

            cleaned.append({
                "question_number": str(
                    item.get("question_number") or ""
                )[:40] or None,
                "question_text": item["question_text"].strip(),
                "question_type": kind,
                "marks": marks,
                "paper_year": year,
                "page_number": page,
                "extraction_confidence": confidence,
            })

        if not cleaned:
            raise ValueError("No questions found")

        return cleaned

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PermanentJobError(
            "Gemini returned an invalid past-paper question structure."
        ) from exc
