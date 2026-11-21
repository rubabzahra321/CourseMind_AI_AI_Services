
import logging
import os
import signal
import socket
import time

from app.core.config import settings
from app.repositories import job_repository
from app.repositories.health_repository import update_worker_heartbeat
from app.services import processing_service

log = logging.getLogger("worker")
running = True


def _stop(*_):
    global running
    running = False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    log.info("worker %s started", worker_id)

    processing_service.recover_stale_jobs()
    last_sweep = time.monotonic()
    last_heartbeat = 0

    while running:
        if time.monotonic() - last_heartbeat >= 15:
            try:
                update_worker_heartbeat(worker_id)
                last_heartbeat = time.monotonic()
            except Exception:
                log.exception("Could not update worker heartbeat")

        try:
            if time.monotonic() - last_sweep > 60:
                processing_service.recover_stale_jobs()
                last_sweep = time.monotonic()

            job = job_repository.claim_next(worker_id)

        except Exception:  # noqa: BLE001  (network blip, paused database...)
            log.exception("could not reach the database; retrying in 10s")
            time.sleep(10)
            continue

        if job is None:
            time.sleep(settings.worker_poll_seconds)
            continue

        processing_service.run_job(job)

    log.info("worker stopped")


if __name__ == "__main__":
    main()
