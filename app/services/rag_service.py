
from app.core.config import settings

from app.ai.gemini import embed_query, generate_answer

from app.repositories import (
    rag_repository,
    chat_repository,
)


INSUFFICIENT_ANSWER = (
    "I could not find sufficient information about this question "
    "in your uploaded course material. Try asking about a topic "
    "covered in your documents."
)


DEPTH_INSTRUCTIONS = {
    "short": (
        "Give a short and direct answer. Use only the most important "
        "points. Avoid unnecessary detail."
    ),
    "detailed": (
        "Give a detailed explanation. Explain the important concepts, "
        "include relevant details from the sources, and use examples "
        "only when the sources support them."
    ),
    "step_by_step": (
        "Explain the answer step by step. Use numbered steps where "
        "helpful. Explain each step in simple student-friendly language."
    ),
}


def _evidence_strength(sources):
    if not sources:
        return "insufficient"

    similarities = [
        float(source.get("similarity") or 0)
        for source in sources
    ]

    strongest = max(similarities, default=0)
    average = sum(similarities) / len(similarities)

    if (
        strongest >= settings.evidence_strong_similarity
        and average >= settings.evidence_partial_similarity
    ):
        return "strong"

    return "partial"


def _build_prompt(question, history, sources, depth_mode):
    source_context = []

    for index, source in enumerate(sources, start=1):
        source_context.append(
            f"""
SOURCE [{index}]
Document: {source["document_title"]}
Pages: {source["page_start"]}-{source["page_end"]}

Content:
{source["content"]}
"""
        )

    context = "\n".join(source_context)

    conversation = "\n".join(
        f'{item["role"]}: {item["content"]}'
        for item in history
    )

    depth_instruction = DEPTH_INSTRUCTIONS[depth_mode]

    return f"""
You are CourseMind AI, a university study assistant.

Answer the student's current question using ONLY the provided
course-material sources.

Rules:
1. Do not use outside knowledge as factual evidence.
2. Do not invent facts, page numbers or document names.
3. Cite factual claims using source labels such as [1] or [2].
4. Use clear, student-friendly language.
5. Previous conversation is context only, not evidence.
6. Never claim a source supports something it does not contain.
7. If the sources only partly answer the question, clearly explain
   which part is supported and which part is not supported.

Answer depth:
{depth_instruction}

Previous conversation:
{conversation}

Retrieved course material:
{context}

Student's current question:
{question}

Write the answer with source citations.
"""


def answer_question(
    user_id: str,
    course_id: str,
    question: str,
    session_id=None,
    depth_mode="detailed",
):
    if depth_mode not in DEPTH_INSTRUCTIONS:
        raise ValueError("Invalid answer depth mode")

    # 1. Create or validate the chat session.
    if session_id:
        session = chat_repository.get_session(
            session_id,
            user_id,
            course_id,
        )
        if not session:
            raise ValueError("Chat session not found")
    else:
        session = chat_repository.create_session(
            user_id,
            course_id,
            question,
        )
        session_id = session["id"]

    # 2. Load conversation context before saving this question.
    history = chat_repository.get_recent_messages(
        session_id,
        user_id,
    )

    # 3. Save the student's question.
    chat_repository.save_message(
        session_id,
        user_id,
        "user",
        question,
        depth_mode=depth_mode,
    )

    # 4. Retrieve relevant course passages.
    query_vector = embed_query(question)

    matches = rag_repository.search_chunks(
        user_id=user_id,
        course_id=course_id,
        query_embedding=query_vector,
        top_k=settings.rag_top_k,
        min_similarity=settings.rag_min_similarity,
    )

    # 5. No matching material means no generated factual answer.
    if not matches:
        answer = INSUFFICIENT_ANSWER

        message = chat_repository.save_message(
            session_id,
            user_id,
            "assistant",
            answer,
            insufficient_evidence=True,
            model_name=settings.gemini_chat_model,
            depth_mode=depth_mode,
            evidence_strength="insufficient",
        )

        return {
            "session_id": session_id,
            "answer": answer,
            "insufficient_evidence": True,
            "evidence_strength": "insufficient",
            "depth_mode": depth_mode,
            "sources": [],
            "message_id": message["id"],
        }

    # 6. Number the retrieved source passages.
    sources = []
    for index, item in enumerate(matches, start=1):
        sources.append({
            **item,
            "rank": index,
        })

    strength = _evidence_strength(sources)

    # 7. Generate a grounded answer.
    prompt = _build_prompt(
        question,
        history,
        sources,
        depth_mode,
    )

    answer = generate_answer(prompt)

    # 8. Save the assistant response and its evidence label.
    message = chat_repository.save_message(
        session_id,
        user_id,
        "assistant",
        answer,
        insufficient_evidence=False,
        model_name=settings.gemini_chat_model,
        depth_mode=depth_mode,
        evidence_strength=strength,
    )

    # 9. Save source links for the answer.
    chat_repository.save_sources(
        message["id"],
        user_id,
        sources,
    )

    return {
        "session_id": session_id,
        "answer": answer,
        "insufficient_evidence": False,
        "evidence_strength": strength,
        "depth_mode": depth_mode,
        "message_id": message["id"],
        "sources": [
            {
                "rank": source["rank"],
                "document_id": source["document_id"],
                "document_title": source["document_title"],
                "page_start": source["page_start"],
                "page_end": source["page_end"],
                "similarity": source["similarity"],
                "excerpt": source["content"][:500],
            }
            for source in sources
        ],
    }
