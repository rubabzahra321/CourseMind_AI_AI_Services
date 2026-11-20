
from app.core.config import settings

from app.ai.gemini import (
    embed_query,
    generate_answer,
)

from app.repositories import (
    rag_repository,
    chat_repository,
)


INSUFFICIENT_ANSWER = (
    "I could not find sufficient information about this question "
    "in your uploaded course material. Try asking about a topic "
    "covered in your documents."
)

STRONG_SIMILARITY_THRESHOLD = 0.70


def _get_depth_instruction(depth):
    """
    Return answer instructions for the selected depth.
    """

    if depth == "short":
        return """
Answer depth: SHORT.

Give a direct and concise answer.
Use approximately 2-4 sentences when possible.
Avoid unnecessary background information.
Include source citations for factual claims.
"""

    if depth == "step_by_step":
        return """
Answer depth: STEP BY STEP.

Explain the answer in numbered steps.
Explain each step using simple student-friendly language.
Do not skip important steps.
Use examples only when supported by the provided material.
Include source citations for factual claims.
"""

    return """
Answer depth: DETAILED.

Give a clear and detailed explanation.
Organize the answer using headings or bullet points when useful.
Explain important terms and concepts.
Use examples only when supported by the provided material.
Include source citations for factual claims.
"""


def _build_prompt(question, history, sources, depth="detailed"):
    """
    Build a prompt using retrieved course material
    and previous conversation context.
    """

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

    depth_instruction = _get_depth_instruction(depth)

    return f"""
You are CourseMind AI, a university study assistant.

Answer the student's current question using ONLY the provided
course-material sources.

Rules:
1. Do not use outside knowledge as factual evidence.
2. If the sources do not contain the answer, clearly say so.
3. Do not invent facts, page numbers or document names.
4. Cite factual claims using source labels such as [1] or [2].
5. Use clear, student-friendly language.
6. Previous conversation is context only, not evidence.
7. Never claim a source supports something it does not contain.
8. If the available sources only partially answer the question,
   clearly explain which part is not covered.

{depth_instruction}

Previous conversation:
{conversation}

Retrieved course material:
{context}

Student's current question:
{question}

Write the answer with source citations.
"""


def _calculate_evidence_strength(sources, answer):
    """
    Estimate retrieval support for the generated answer.

    This is a heuristic based on source similarity and citation
    presence. It is not a factual verification system.
    """

    if not sources:
        return "insufficient"

    similarities = [
        float(source.get("similarity") or 0)
        for source in sources
    ]

    highest_similarity = max(similarities, default=0)

    has_source_citation = (
        "[1]" in answer
        or any(
            f"[{index}]" in answer
            for index in range(2, len(sources) + 1)
        )
    )

    if (
        highest_similarity >= STRONG_SIMILARITY_THRESHOLD
        and has_source_citation
    ):
        return "strong"

    return "partial"


def answer_question(
    user_id: str,
    course_id: str,
    question: str,
    session_id=None,
    depth="detailed",
):
    """
    Main RAG chat function.
    """

    # 1. Validate the requested depth.
    if depth not in ("short", "detailed", "step_by_step"):
        raise ValueError("Invalid chat depth")

    # 2. Create or validate chat session.
    if session_id:
        session = chat_repository.get_session(
            session_id,
            user_id,
            course_id
        )

        if not session:
            raise ValueError("Chat session not found")

    else:
        session = chat_repository.create_session(
            user_id,
            course_id,
            question
        )

        session_id = session["id"]

    # 3. Get previous conversation before saving this question.
    history = chat_repository.get_recent_messages(
        session_id,
        user_id
    )

    # 4. Save the user's question.
    chat_repository.save_message(
        session_id,
        user_id,
        "user",
        question
    )

    # 5. Convert question into an embedding.
    query_vector = embed_query(question)

    # 6. Retrieve relevant passages.
    matches = rag_repository.search_chunks(
        user_id=user_id,
        course_id=course_id,
        query_embedding=query_vector,
        top_k=settings.rag_top_k,
        min_similarity=settings.rag_min_similarity,
    )

    # 7. No evidence: do not ask Gemini to guess.
    if not matches:
        answer = INSUFFICIENT_ANSWER

        message = chat_repository.save_message(
            session_id,
            user_id,
            "assistant",
            answer,
            insufficient_evidence=True,
            model_name=settings.gemini_chat_model,
            depth_mode=depth,
            evidence_strength="insufficient",
        )

        return {
            "session_id": session_id,
            "answer": answer,
            "depth": depth,
            "evidence_strength": "insufficient",
            "insufficient_evidence": True,
            "sources": [],
            "message_id": message["id"],
        }

    # 8. Prepare source references.
    sources = []

    for index, item in enumerate(matches, start=1):
        sources.append({
            **item,
            "rank": index,
        })

    # 9. Generate grounded answer.
    prompt = _build_prompt(
        question,
        history,
        sources,
        depth,
    )

    answer = generate_answer(prompt)

    # 10. Estimate evidence strength.
    evidence_strength = _calculate_evidence_strength(
        sources,
        answer,
    )

    # 11. Save assistant response.
    message = chat_repository.save_message(
        session_id,
        user_id,
        "assistant",
        answer,
        insufficient_evidence=False,
        model_name=settings.gemini_chat_model,
        depth_mode=depth,
        evidence_strength=evidence_strength,
    )

    # 12. Save source references.
    chat_repository.save_sources(
        message["id"],
        user_id,
        sources
    )

    # 13. Return answer and source information.
    return {
        "session_id": session_id,
        "answer": answer,
        "depth": depth,
        "evidence_strength": evidence_strength,
        "insufficient_evidence": False,
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
