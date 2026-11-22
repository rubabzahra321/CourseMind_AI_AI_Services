"""
Machine learning orchestration: build training data from the database,
train and compare models, save them, and use them for predictions.

Currently: topic classification (question text -> syllabus topic).
"""

from collections import Counter

from app.core.config import settings
from app.db.client import get_db
from app.ml import registry
from app.ml import topic_classifier as tc
from app.repositories import ml_repository
from app.services.insights_service import _all_rows, _topics


MODEL_TYPE = "topic_classifier"
BATCH = 50                                     # ids per "in" query (URL length)


class InsufficientTrainingData(ValueError):
    """Not enough labelled data to train yet (HTTP 422)."""


# ---------------------------------------------------------------- training data
def _passage_texts(user_id, chunk_ids):
    texts = {}

    for start in range(0, len(chunk_ids), BATCH):
        batch = chunk_ids[start:start + BATCH]

        rows = (
            get_db().table("document_chunks")
            .select("id, content")
            .eq("user_id", str(user_id))
            .in_("id", batch)
            .execute().data or []
        )

        for row in rows:
            texts[str(row["id"])] = row["content"]

    return texts


def build_training_set(user_id, course_id, include_ai_links=False):
    """
    Labelled examples for this course. Only HUMAN-made labels are used, so the
    model never learns from its own guesses:

      syllabus  - each topic's title and description (1 example per topic)
      question  - past-paper questions the student mapped by hand
      passage   - lecture passages linked to a topic by hand
                  (AI-linked passages only if include_ai_links is set)
    """

    topics = _topics(user_id, course_id)
    topic_ids = {str(t["id"]) for t in topics}
    samples = []

    for topic in topics:
        description = (topic.get("description") or "").strip()
        title = topic["title"].strip()

        samples.append({
            "text": f"{title}. {description}" if description else title,
            "topic_id": str(topic["id"]),
            "source": "syllabus",
        })

    questions = {
        str(q["id"]): q
        for q in _all_rows(
            lambda: get_db().table("past_paper_questions")
            .select("id, question_text")
            .eq("user_id", str(user_id))
            .eq("course_id", str(course_id))
        )
    }

    for row in _all_rows(
        lambda: get_db().table("question_topic_mappings")
        .select("past_paper_question_id, topic_id")
        .eq("user_id", str(user_id))
        .eq("mapping_method", "manual")
    ):
        question = questions.get(str(row["past_paper_question_id"]))

        if question and str(row["topic_id"]) in topic_ids:
            samples.append({
                "text": question["question_text"],
                "topic_id": str(row["topic_id"]),
                "source": "question",
            })

    methods = {"manual"} | ({"ai"} if include_ai_links else set())

    links = [
        link for link in _all_rows(
            lambda: get_db().table("topic_chunk_links")
            .select("topic_id, chunk_id, method")
            .eq("user_id", str(user_id))
            .eq("course_id", str(course_id))
        )
        if link["method"] in methods and str(link["topic_id"]) in topic_ids
    ]

    texts = _passage_texts(user_id, sorted({str(l["chunk_id"]) for l in links}))

    for link in links:
        text = texts.get(str(link["chunk_id"]))

        if text:
            samples.append({
                "text": text,
                "topic_id": str(link["topic_id"]),
                "source": "passage",
            })

    samples = [s for s in samples if s["text"] and s["text"].strip()]

    counts = Counter(s["source"] for s in samples)

    return topics, samples, {
        "syllabus_topics": counts.get("syllabus", 0),
        "labelled_questions": counts.get("question", 0),
        "linked_passages": counts.get("passage", 0),
        "total_samples": len(samples),
        "includes_ai_links": include_ai_links,
    }


def _usable(topics, samples):
    """Keep only topics with enough examples to learn and to cross-validate."""

    per_topic = Counter(s["topic_id"] for s in samples)
    minimum = settings.ml_min_examples_per_topic

    kept = {t for t, n in per_topic.items() if n >= minimum}
    titles = {str(t["id"]): t["title"] for t in topics}

    dropped = [
        {"topic_id": tid, "topic_title": title, "examples": per_topic.get(tid, 0)}
        for tid, title in titles.items()
        if tid not in kept
    ]

    return kept, dropped, titles, per_topic


def _insufficient_message(counts, kept_topics):
    return (
        "Not enough labelled data to train a topic classifier yet. "
        f"Found {counts['labelled_questions']} hand-mapped questions and "
        f"{counts['linked_passages']} linked passages; only {kept_topics} "
        f"topic(s) have at least {settings.ml_min_examples_per_topic} "
        f"examples and at least {settings.ml_min_topics} are needed. "
        "Map more past-paper questions to topics (or link lecture passages) "
        "and train again. Until then suggestions use keyword matching."
    )


# ---------------------------------------------------------------- cards
def _card(row, activated=None, reason=None):
    card = {
        "id": row["id"],
        "model_type": row["model_type"],
        "algorithm": row["algorithm"],
        "algorithm_label": tc.LABELS.get(row["algorithm"], row["algorithm"]),
        "version": row["version"],
        "is_active": bool(row.get("is_active")),
        "n_training_samples": row["n_training_samples"],
        "trained_at": row.get("trained_at"),
        "metrics": row.get("metrics") or {},
        "comparison": row.get("comparison") or [],
    }

    if activated is not None:
        card["activated"] = activated
        card["activation_reason"] = reason

    return card


# ---------------------------------------------------------------- train
def train_topic_classifier(user_id, course_id, include_ai_links=False, force=False):
    topics, samples, counts = build_training_set(
        user_id, course_id, include_ai_links)

    kept, dropped, titles, _per_topic = _usable(topics, samples)

    if len(kept) < settings.ml_min_topics:
        raise InsufficientTrainingData(_insufficient_message(counts, len(kept)))

    samples = [s for s in samples if s["topic_id"] in kept]
    per_topic = Counter(s["topic_id"] for s in samples)
    n_splits = min(settings.ml_cv_max_folds, min(per_topic.values()))

    if n_splits < 2:
        raise InsufficientTrainingData(_insufficient_message(counts, len(kept)))

    texts = [s["text"] for s in samples]
    labels = [s["topic_id"] for s in samples]
    sources = [s["source"] for s in samples]

    try:
        results, best, basis, classes = tc.evaluate_candidates(
            texts, labels, sources, n_splits,
            settings.ml_random_seed,
            settings.ml_min_question_samples_for_selection,
        )
    except ValueError as exc:                     # e.g. empty vocabulary
        raise InsufficientTrainingData(
            f"The training text could not be used: {exc}") from exc

    metrics = tc.detailed_metrics(
        results, best, labels, titles, classes, basis, n_splits)

    metrics["training_data"] = {
        **counts,
        "samples_used": len(samples),
        "topics_trained": len(kept),
        "samples_per_topic": {
            titles[t]: n for t, n in sorted(per_topic.items(), key=lambda i: titles[i[0]])
        },
        "topics_left_out": dropped,
        "random_seed": settings.ml_random_seed,
    }

    comparison = tc.comparison_table(results, best)

    pipeline = tc.fit_final(
        best, texts, labels, n_splits, settings.ml_random_seed)

    bundle = {
        "pipeline": pipeline,
        "algorithm": best,
        "titles": titles,
        "topic_ids": sorted(kept),
    }

    row = registry.save_version(
        user_id, course_id, MODEL_TYPE, best,
        len(samples), metrics, comparison, bundle,
    )

    activated, reason = registry.activate_if_better(row, force=force)
    row["is_active"] = activated

    return _card(row, activated, reason)


# ---------------------------------------------------------------- status
def get_status(user_id, course_id):
    active = ml_repository.get_active(user_id, course_id, MODEL_TYPE)
    history = ml_repository.list_models(user_id, course_id, MODEL_TYPE)

    topics, samples, counts = build_training_set(user_id, course_id)
    kept, dropped, _titles, _per_topic = _usable(topics, samples)

    can_train = len(kept) >= settings.ml_min_topics

    return {
        "model_type": MODEL_TYPE,
        "active_model": _card(active) if active else None,
        "using_fallback": active is None,
        "history": [
            {
                "version": row["version"],
                "algorithm": row["algorithm"],
                "algorithm_label": tc.LABELS.get(row["algorithm"], row["algorithm"]),
                "selection_score": (row.get("metrics") or {}).get("selection_score"),
                "n_training_samples": row["n_training_samples"],
                "is_active": bool(row.get("is_active")),
                "trained_at": row.get("trained_at"),
            }
            for row in history
        ],
        "training_data": {
            **counts,
            "topics_ready": len(kept),
            "topics_left_out": dropped,
            "minimum_examples_per_topic": settings.ml_min_examples_per_topic,
            "minimum_topics": settings.ml_min_topics,
        },
        "can_train": can_train,
        "message": (
            None if can_train
            else _insufficient_message(counts, len(kept))
        ),
    }


# ---------------------------------------------------------------- classify
def classify_questions(user_id, course_id, question_ids=None, save=False,
                       min_confidence=None, limit=None):
    """
    Suggest topics for past-paper questions.

    By default it looks at questions that have no topic yet. Suggestions are
    only SAVED (as `ai_suggested`) when ALL of these hold: save=true, a trained
    model is in use, the confidence reaches the threshold, the question is still
    `pending` review and it has no topic yet. Reviewed questions are never
    changed, so AI guesses cannot leak into exam-frequency statistics.
    """

    min_confidence = (
        settings.ml_min_confidence if min_confidence is None else min_confidence)
    limit = min(limit or settings.ml_max_classify_questions,
                settings.ml_max_classify_questions)

    topics = _topics(user_id, course_id)
    topic_by_id = {str(t["id"]): t for t in topics}

    questions = _all_rows(
        lambda: get_db().table("past_paper_questions")
        .select("id, question_number, question_text, review_status")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
    )
    by_id = {str(q["id"]): q for q in questions}

    mapped = {
        str(m["past_paper_question_id"])
        for m in _all_rows(
            lambda: get_db().table("question_topic_mappings")
            .select("past_paper_question_id")
            .eq("user_id", str(user_id))
        )
    }

    not_found = []

    if question_ids:
        wanted = [str(q) for q in question_ids]
        not_found = [q for q in wanted if q not in by_id]
        selected = [by_id[q] for q in wanted if q in by_id]
    else:
        selected = [q for q in questions if str(q["id"]) not in mapped]

    selected = selected[:limit]

    row, bundle = registry.load_active_bundle(user_id, course_id, MODEL_TYPE)
    texts = [q["question_text"] for q in selected]

    if bundle:
        predictions = tc.predict_top_k(bundle["pipeline"], texts, 3)
        model_used = MODEL_TYPE
    else:
        predictions = tc.keyword_fallback(topics, texts, 3)
        model_used = "fallback_keyword"

    items, saved = [], 0

    for question, prediction in zip(selected, predictions):
        qid = str(question["id"])

        suggestions = [
            {
                "topic_id": tid,
                "topic_title": topic_by_id[tid]["title"],
                "probability": round(prob, 4),
            }
            for tid, prob in prediction
            if tid in topic_by_id               # topic may have been deleted
        ]

        top = suggestions[0]["probability"] if suggestions else 0.0
        needs_review = not suggestions or top < min_confidence
        was_saved = False

        if (
            save and bundle and suggestions and not needs_review
            and question["review_status"] == "pending"
            and qid not in mapped
        ):
            try:
                get_db().table("question_topic_mappings").insert({
                    "past_paper_question_id": qid,
                    "topic_id": suggestions[0]["topic_id"],
                    "user_id": str(user_id),
                    "mapping_method": "ai_suggested",
                    "confidence": top,
                }).execute()
                was_saved = True
                saved += 1
            except Exception:
                was_saved = False

        items.append({
            "question_id": qid,
            "question_number": question.get("question_number"),
            "question_text": (question["question_text"] or "")[:200],
            "review_status": question["review_status"],
            "already_mapped": qid in mapped,
            "suggestions": suggestions,
            "needs_review": needs_review,
            "saved": was_saved,
        })

    return {
        "model_used": model_used,
        "model_version": row["version"] if (row and bundle) else None,
        "algorithm": row["algorithm"] if (row and bundle) else None,
        "min_confidence": min_confidence,
        "total": len(items),
        "needs_review_count": sum(1 for i in items if i["needs_review"]),
        "saved_count": saved,
        "not_found": not_found,
        "items": items,
    }
