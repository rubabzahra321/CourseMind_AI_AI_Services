
"""
In-memory stand-in for the Supabase client used by the AI service.

It copies the behaviour of the real supabase-py / postgrest client that matters
for correctness:
  * maybe_single().execute() returns None when no row matches.
  * single() raises when the row count is not exactly 1.
  * insert() returns the inserted rows; .select(cols) limits returned columns.
  * UNIQUE constraints from the SQL schema are enforced.
"""

import copy
import uuid

from collections import defaultdict
from datetime import datetime, timedelta, timezone


class FakeAPIError(Exception):

    def __init__(self, message, code="PGRST000"):
        super().__init__(message)
        self.code = code


class UniqueViolation(FakeAPIError):

    def __init__(self, table, cols):
        super().__init__(
            f"duplicate key on {table}{cols}",
            code="23505"
        )


class Response:

    def __init__(self, data, count=None):
        self.data = data
        self.count = count


# ---- constraints copied from the SQL schema -----------------------------

UNIQUE = {
    "quiz_duel_moves": [
        ("game_id", "position")
    ],
    "practice_attempts": [
        ("session_id", "question_id")
    ],
    "practice_session_questions": [
        ("session_id", "question_id"),
        ("session_id", "question_order")
    ],
    "study_plan_items": [
        ("plan_id", "topic_id", "planned_date")
    ],
    "question_options": [
        ("question_id", "option_order")
    ],
    "document_chunks": [
        ("document_id", "chunk_index")
    ],
}


DEFAULTS = {
    "generated_questions": {
        "status": "active"
    },
    "chat_messages": {
        "insufficient_evidence": False
    },
    "practice_sessions": {
        "status": "in_progress",
        "correct_answers": 0,
        "accuracy_percentage": None
    },
    "study_plan_items": {
        "is_done": False,
        "done_at": None
    },
    "quiz_duel_games": {
        "winner": None
    },
    "processing_jobs": {
        "attempts": 0,
        "max_attempts": 3
    },
}


BASE_TIME = datetime(
    2026, 1, 1, tzinfo=timezone.utc
)


class Query:

    def __init__(self, db, table):

        self.db = db
        self.name = table

        self.op = "select"
        self.payload = None

        self.filters = []
        self.orders = []

        self.max_rows = None
        self.cols = "*"

        self.mode = None
        # None | "single" | "maybe"


    # ---- builders ----

    def select(self, cols="*", **_):
        self.cols = cols
        return self


    def insert(self, payload, **_):
        self.op = "insert"
        self.payload = payload
        return self


    def update(self, payload, **_):
        self.op = "update"
        self.payload = payload
        return self


    def delete(self):
        self.op = "delete"
        return self


    def eq(self, col, val):
        self.filters.append((col, "eq", val))
        return self


    def lt(self, col, val):
        self.filters.append((col, "lt", val))
        return self


    def in_(self, col, vals):
        self.filters.append((col, "in", vals))
        return self


    def order(self, col, desc=False, **_):
        self.orders.append((col, desc))
        return self


    def limit(self, n):
        self.max_rows = n
        return self


    def single(self):
        self.mode = "single"
        return self


    def maybe_single(self):
        self.mode = "maybe"
        return self


    # ---- helpers ----

    def _match(self, row):

        for col, op, val in self.filters:

            cur = row.get(col)

            if op == "eq" and str(cur) != str(val):
                return False

            if op == "lt" and not (
                cur is not None and str(cur) < str(val)
            ):
                return False

            if op == "in" and str(cur) not in {
                str(v) for v in val
            }:
                return False

        return True


    def _project(self, row):

        if self.cols in ("*", None):
            return copy.deepcopy(row)

        keep = [
            c.strip()
            for c in self.cols.split(",")
        ]

        return {
            c: copy.deepcopy(row.get(c))
            for c in keep
        }


    def _finish(self, rows):

        rows = [
            self._project(r)
            for r in rows
        ]

        if self.mode == "single":

            if len(rows) != 1:
                raise FakeAPIError(
                    "JSON object requested, multiple (or no) rows returned",
                    "PGRST116"
                )

            return Response(rows[0])


        if self.mode == "maybe":

            if len(rows) == 0:
                return None

            if len(rows) > 1:
                raise FakeAPIError(
                    "more than one row",
                    "406"
                )

            return Response(rows[0])


        return Response(rows)


    # ---- execution ----

    def execute(self):

        table = self.db.tables[self.name]

        if self.op == "insert":

            items = (
                self.payload
                if isinstance(self.payload, list)
                else [self.payload]
            )

            inserted = []

            for item in items:

                row = {
                    **DEFAULTS.get(self.name, {}),
                    **copy.deepcopy(item)
                }

                row.setdefault(
                    "id",
                    str(uuid.uuid4())
                )

                stamp = self.db.tick()

                row.setdefault(
                    "created_at",
                    stamp
                )

                if self.name == "practice_attempts":

                    row.setdefault(
                        "answered_at",
                        stamp
                    )

                self.db.check_unique(
                    self.name,
                    row
                )

                table.append(row)
                inserted.append(row)

            return self._finish(inserted)


        matched = [
            r
            for r in table
            if self._match(r)
        ]


        if self.op == "update":

            for row in matched:

                row.update(
                    copy.deepcopy(self.payload)
                )

                self.db.check_unique(
                    self.name,
                    row,
                    ignore=row
                )

            return self._finish(matched)


        if self.op == "delete":

            for row in matched:
                table.remove(row)

            return self._finish(matched)


        for col, desc in reversed(self.orders):

            matched.sort(
                key=lambda r: (
                    r.get(col) is None,
                    str(r.get(col))
                ),
                reverse=desc
            )

        if self.max_rows is not None:
            matched = matched[:self.max_rows]

        return self._finish(matched)


class RpcCall:

    def __init__(self, db, name, params):
        self.db = db
        self.name = name
        self.params = params


    def execute(self):

        self.db.rpc_calls.append(
            (self.name, self.params)
        )

        handler = self.db.rpc_handlers.get(
            self.name
        )

        if handler is None:
            raise FakeAPIError(
                f"function {self.name} not found",
                "PGRST202"
            )

        return Response(
            handler(self.params)
        )


class _Bucket:

    def __init__(self, files):
        self.files = files


    def download(self, path):

        if path not in self.files:
            raise FakeAPIError(
                "Object not found"
            )

        return self.files[path]


class _Storage:

    def __init__(self, files):
        self.files = files


    def from_(self, _bucket):
        return _Bucket(self.files)


class FakeDB:

    def __init__(self):

        self.tables = defaultdict(list)

        self.rpc_handlers = {
            "refresh_topic_progress": lambda p: 1,
            "claim_processing_job": self._claim_job,
        }

        self.rpc_calls = []

        self.files = {}

        self.storage = _Storage(
            self.files
        )

        self._clock = 0


    # Deterministic, strictly increasing timestamps.
    def tick(self):

        self._clock += 1

        return (
            BASE_TIME
            + timedelta(seconds=self._clock)
        ).isoformat()


    def table(self, name):
        return Query(self, name)


    def rpc(self, name, params):
        return RpcCall(
            self,
            name,
            params
        )


    def seed(self, name, *rows):

        out = []

        for item in rows:

            row = {
                **DEFAULTS.get(name, {}),
                **copy.deepcopy(item)
            }

            row.setdefault(
                "id",
                str(uuid.uuid4())
            )

            row.setdefault(
                "created_at",
                self.tick()
            )

            self.tables[name].append(row)
            out.append(row)

        return (
            out[0]
            if len(out) == 1
            else out
        )


    def rows(self, name, **where):

        return [
            r
            for r in self.tables[name]
            if all(
                str(r.get(k)) == str(v)
                for k, v in where.items()
            )
        ]


    def check_unique(
        self,
        name,
        row,
        ignore=None
    ):

        for cols in UNIQUE.get(name, []):

            for other in self.tables[name]:

                if other is row or other is ignore:
                    continue

                if all(
                    str(other.get(c)) == str(row.get(c))
                    for c in cols
                ):
                    raise UniqueViolation(
                        name,
                        cols
                    )


        if (
            name == "study_plans"
            and row.get("status") == "active"
        ):

            for other in self.tables[name]:

                if other is row or other is ignore:
                    continue

                if (
                    other.get("status") == "active"
                    and str(other.get("course_id"))
                    == str(row.get("course_id"))
                ):
                    raise UniqueViolation(
                        name,
                        ("course_id where active",)
                    )


        if name == "processing_jobs":

            if row.get("idempotency_key"):

                for other in self.tables[name]:

                    if other is row or other is ignore:
                        continue

                    if (
                        other.get("idempotency_key")
                        == row.get("idempotency_key")
                    ):
                        raise UniqueViolation(
                            name,
                            ("idempotency_key",)
                        )


            if row.get("status") in {
                "queued",
                "processing"
            }:

                for other in self.tables[name]:

                    if other is row or other is ignore:
                        continue

                    if (
                        other.get("status")
                        in {"queued", "processing"}
                        and str(other.get("document_id"))
                        == str(row.get("document_id"))
                        and other.get("job_type")
                        == row.get("job_type")
                    ):
                        raise UniqueViolation(
                            name,
                            (
                                "document_id",
                                "job_type",
                                "active"
                            )
                        )


    def _claim_job(self, params):

        now = datetime.now(
            timezone.utc
        ).isoformat()

        for job in self.tables["processing_jobs"]:

            if (
                job["status"] == "queued"
                and job.get("available_at", "") <= now
                and job["attempts"] < job["max_attempts"]
            ):

                job.update(
                    status="processing",
                    attempts=job["attempts"] + 1,
                    locked_at=now,
                    locked_by=params["p_worker_id"]
                )

                return [
                    copy.deepcopy(job)
                ]

        return []
