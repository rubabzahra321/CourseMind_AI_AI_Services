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


def _build_prompt(question, history, sources):

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
6. If explaining a technical concept, structure the answer
   into understandable steps when useful.
7. Previous conversation is context only, not evidence.
8. Never claim a source supports something it does not contain.

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
):

    # 1. Create or validate chat session.
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

    # 2. Get previous conversation before saving this question.
    history = chat_repository.get_recent_messages(
        session_id,
        user_id
    )

    # 3. Save the user's question.
    chat_repository.save_message(
        session_id,
        user_id,
        "user",
        question
    )

    # 4. Convert question into an embedding.
    query_vector = embed_query(question)

    # 5. Retrieve relevant passages.
    matches = rag_repository.search_chunks(
        user_id=user_id,
        course_id=course_id,
        query_embedding=query_vector,
        top_k=settings.rag_top_k,
        min_similarity=settings.rag_min_similarity,
    )

    # 6. No evidence: do not ask Gemini to guess.
    if not matches:

        answer = INSUFFICIENT_ANSWER

        message = chat_repository.save_message(
            session_id,
            user_id,
            "assistant",
            answer,
            insufficient_evidence=True,
            model_name=settings.gemini_chat_model,
        )

        return {
            "session_id": session_id,
            "answer": answer,
            "insufficient_evidence": True,
            "sources": [],
            "message_id": message["id"],
        }

    # 7. Prepare source references.
    sources = []

    for index, item in enumerate(matches, start=1):

        sources.append({
            **item,
            "rank": index,
        })

    # 8. Generate grounded answer.
    prompt = _build_prompt(
        question,
        history,
        sources
    )

    answer = generate_answer(prompt)

    # 9. Save assistant response.
    message = chat_repository.save_message(
        session_id,
        user_id,
        "assistant",
        answer,
        insufficient_evidence=False,
        model_name=settings.gemini_chat_model,
    )

    # 10. Save source references.
    chat_repository.save_sources(
        message["id"],
        user_id,
        sources
    )

    # 11. Return answer and source information.
    return {
        "session_id": session_id,
        "answer": answer,
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