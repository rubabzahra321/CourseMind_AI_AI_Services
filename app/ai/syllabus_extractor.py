
import json

from google.genai import types

from app.ai.gemini import get_gemini_client
from app.core.config import settings
from app.core.errors import PermanentJobError


def extract_syllabus_topics(pages: list[tuple[int, str]]) -> dict:
    material = "\n\n".join(
        f"[PDF PAGE {number}]\n{text}"
        for number, text in pages if text.strip()
    )

    if not material.strip():
        raise PermanentJobError("No readable syllabus text was found.")

    prompt = f'''Extract the syllabus hierarchy from the supplied PDF text.
Return JSON only, using this exact structure:
{{"chapters":[{{"title":"Chapter title","description":null,"page_start":1,"page_end":2,
"topics":[{{"title":"Topic title","description":null,"page_start":1,"page_end":2,
"subtopics":[{{"title":"Subtopic title","description":null,"page_start":1,"page_end":1}}]}}]}}]}}
Rules:
- Only extract content explicitly supported by the syllabus. Never invent content.
- Preserve the source wording and hierarchy.
- Chapters may be inferred from clearly labelled units/modules only; do not invent chapter names.
- Page numbers must be the supplied PDF page numbers; use null if uncertain.
- descriptions may be null. Return an empty topics/subtopics array when appropriate.
- Do not include markdown fences or commentary.

SYLLABUS TEXT:
{material}'''

    response = get_gemini_client().models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=8000,
        ),
    )

    try:
        payload = json.loads(response.text or "")
        chapters = payload["chapters"]

        if not isinstance(chapters, list):
            raise ValueError("chapters must be a list")

        for chapter in chapters:
            _validate_node(chapter, "topics")

            for topic in chapter.get("topics", []):
                _validate_node(topic, "subtopics")

                for subtopic in topic.get("subtopics", []):
                    _validate_node(subtopic, None)

        if not chapters:
            raise ValueError("No chapters found")

        return {"chapters": chapters}

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PermanentJobError(
            "Gemini returned an invalid syllabus structure."
        ) from exc


def _validate_node(node: dict, children_key: str | None) -> None:
    if (
        not isinstance(node, dict)
        or not isinstance(node.get("title"), str)
        or not node["title"].strip()
    ):
        raise ValueError("Every extracted item must have a title")

    if children_key is not None and not isinstance(
        node.get(children_key, []), list
    ):
        raise ValueError(f"{children_key} must be a list")

    for key in ("page_start", "page_end"):
        value = node.get(key)

        if value is not None and (
            not isinstance(value, int) or value < 1
        ):
            node[key] = None
