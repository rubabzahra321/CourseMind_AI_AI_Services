"""
Saving, loading and versioning trained models.

The model file (joblib) goes to private storage; its metrics go to the
ml_models table. Models are only ever loaded from files this service wrote
with the service-role key (joblib uses pickle, so never load untrusted files).
"""

import io

import joblib

from app.core.config import settings
from app.db.client import get_db
from app.repositories import ml_repository


class SupabaseArtifactStore:
    def put(self, path, data):
        get_db().storage.from_(settings.storage_bucket).upload(
            path,
            data,
            {"content-type": "application/octet-stream", "upsert": "true"},
        )

    def get(self, path):
        return get_db().storage.from_(settings.storage_bucket).download(path)


class MemoryArtifactStore:
    """Used by tests."""

    def __init__(self):
        self.files = {}

    def put(self, path, data):
        self.files[path] = data

    def get(self, path):
        return self.files[path]


_store = SupabaseArtifactStore()


def set_artifact_store(store):
    global _store
    _store = store


def artifact_path(user_id, course_id, model_type, version):
    if user_id is None:                         # global (pooled) model
        return f"ml-models/global/{model_type}_v{version}.joblib"

    return f"ml-models/{user_id}/{course_id}/{model_type}_v{version}.joblib"


def _serialise(bundle):
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    return buffer.getvalue()


def save_version(user_id, course_id, model_type, algorithm,
                 n_samples, metrics, comparison, bundle):
    """Store a new inactive version and return its row."""

    versions = ml_repository.list_models(user_id, course_id, model_type)
    version = (versions[0]["version"] + 1) if versions else 1
    path = artifact_path(user_id, course_id, model_type, version)

    _store.put(path, _serialise(bundle))

    return ml_repository.insert_model({
        "user_id": str(user_id) if user_id is not None else None,
        "course_id": str(course_id) if course_id is not None else None,
        "model_type": model_type,
        "algorithm": algorithm,
        "version": version,
        "n_training_samples": int(n_samples),
        "metrics": metrics,
        "comparison": comparison,
        "artifact_path": path,
        "is_active": False,
    })


def activate_if_better(row, force=False):
    """
    The new version becomes active only if it is at least as good as the
    active one (same selection metric), unless `force` is set.

    Returns (activated, reason).
    """

    current = ml_repository.get_active(
        row["user_id"], row["course_id"], row["model_type"])

    new_score = (row.get("metrics") or {}).get("selection_score", 0)

    if current and not force:
        old_score = (current.get("metrics") or {}).get("selection_score", 0)

        if new_score < old_score:
            return False, (
                f"Kept version {current['version']} (score {old_score}) "
                f"because the new version scored lower ({new_score})."
            )

    ml_repository.activate(
        row["id"], row["user_id"], row["course_id"], row["model_type"])

    if not current:
        return True, "First model for this course."

    if force:
        return True, "Activated by request."

    return True, (
        f"Replaced version {current['version']} (score "
        f"{(current.get('metrics') or {}).get('selection_score', 0)})."
    )


def load_bundle(row):
    return joblib.load(io.BytesIO(_store.get(row["artifact_path"])))


def load_active_bundle(user_id, course_id, model_type):
    """(row, bundle) of the active model, or (None, None)."""

    row = ml_repository.get_active(user_id, course_id, model_type)

    if not row:
        return None, None

    try:
        return row, load_bundle(row)
    except Exception:
        # File missing or unreadable: behave as "no model" instead of failing.
        return row, None


def save_global_version(model_type, algorithm, n_samples, metrics, comparison, bundle):
    return save_version(
        None, None, model_type, algorithm, n_samples, metrics, comparison, bundle
    )


def activate_global_if_better(row, force=False):
    return activate_if_better(row, force=force)


def load_global_active_bundle(model_type):
    return load_active_bundle(None, None, model_type)
