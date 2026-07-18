"""Single-instance lock for nightly stages.

Why: the pipeline is deliberately redundant (a supervisor runs each stage AND
the original scheduled tasks remain as backstops). Every stage is idempotent
against a PRIOR run, but not against a CONCURRENT one — two paper_trade
processes would each read the ledger before the other writes and place the
same entries twice. This lock makes redundancy safe: whoever starts second
sees the lock and exits cleanly (exit 0 — the work is being done).

O_EXCL file create is atomic on Windows/NTFS. A lock older than STALE_MIN is
treated as a crash leftover and stolen.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
LOCK_DIR = BACKEND / "data_store" / "locks"
STALE_MIN = 120           # a nightly stage never legitimately runs 2h


@contextmanager
def single_instance(name: str):
    """Yields True if this process holds the lock, False if another does."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCK_DIR / f"{name}.lock"
    fd = None
    try:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age_min = (time.time() - path.stat().st_mtime) / 60 if path.exists() else 0
            if age_min < STALE_MIN:
                yield False
                return
            # stale crash leftover: steal it
            try:
                path.unlink()
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except OSError:
                yield False
                return
        os.write(fd, f"{os.getpid()} {time.time():.0f}".encode())
        yield True
    finally:
        if fd is not None:
            os.close(fd)
            try:
                path.unlink()
            except OSError:
                pass
