"""SQLite persistence for alarm/detection events. Single-writer, WAL mode
for safe concurrent reads from the dashboard while the pipeline writes.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    class_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    zones TEXT,
    box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
    snapshot_path TEXT,
    clip_path TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
"""


@dataclass
class EventRecord:
    timestamp: float
    class_name: str
    severity: str
    confidence: float
    zones: str
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float
    snapshot_path: str | None = None
    clip_path: str | None = None
    acknowledged: int = 0
    id: int | None = None


class EventDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert_event(self, record: EventRecord) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO events
                   (timestamp, class_name, severity, confidence, zones,
                    box_x1, box_y1, box_x2, box_y2, snapshot_path, clip_path, acknowledged)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.timestamp, record.class_name, record.severity, record.confidence,
                    record.zones, record.box_x1, record.box_y1, record.box_x2, record.box_y2,
                    record.snapshot_path, record.clip_path, record.acknowledged,
                ),
            )
            return cur.lastrowid

    def acknowledge(self, event_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE events SET acknowledged = 1 WHERE id = ?", (event_id,))

    def recent_events(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def prune_older_than(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86400
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            return cur.rowcount
