"""
attendance_log.py
==================

Records attendance to SQLite and answers reporting questions about it.

The rule that matters most: **one attendance record per person per day.**
A recognition system sees the same face in dozens of consecutive video
frames, so without a uniqueness guarantee a single person walking past
the camera produces a hundred "present" rows. That guarantee is enforced
by a database UNIQUE constraint, not by application-side checking —
application checks lose races, schema constraints don't.

Status is derived from a configurable cutoff time: on time before it,
late after it. Anyone enrolled who never appears is reported absent,
which is computed at report time rather than stored, since "absent" is
the absence of a record and storing it would just create a second source
of truth to keep in sync.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    check_in_time TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('present', 'late')),
    confidence REAL NOT NULL,
    UNIQUE (person_id, date)
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
"""

STATUS_PRESENT = "present"
STATUS_LATE = "late"
STATUS_ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class AttendanceRecord:
    """One person's attendance for one day."""

    person_id: str
    name: str
    date: str  # ISO date, e.g. "2026-09-15"
    check_in_time: str  # "HH:MM:SS"
    status: str
    confidence: float


@dataclass(frozen=True, slots=True)
class DailyReport:
    """A day's attendance summary across everyone enrolled."""

    date: str
    present: list[AttendanceRecord]
    late: list[AttendanceRecord]
    absent: list[tuple[str, str]]  # (person_id, name)

    @property
    def total_enrolled(self) -> int:
        return len(self.present) + len(self.late) + len(self.absent)

    @property
    def attendance_rate(self) -> float:
        """Fraction of enrolled people who showed up at all, 0-1."""
        if self.total_enrolled == 0:
            return 0.0
        return (len(self.present) + len(self.late)) / self.total_enrolled


class AttendanceLog:
    """SQLite-backed attendance recording and reporting.

    Args:
        db_path: Path to the SQLite file. Created if absent.
        late_after: Check-ins at or after this time are marked late.
    """

    def __init__(self, db_path: str | Path = "data/attendance.db", late_after: time = time(9, 0)) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.late_after = late_after
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _status_for(self, check_in: datetime) -> str:
        return STATUS_LATE if check_in.time() >= self.late_after else STATUS_PRESENT

    def record(
        self,
        person_id: str,
        name: str,
        confidence: float,
        check_in: datetime | None = None,
    ) -> tuple[AttendanceRecord, bool]:
        """Record a check-in, ignoring it if the person is already logged today.

        Args:
            person_id: The recognized person's id.
            name: Their display name.
            confidence: The match confidence, stored for auditability.
            check_in: When they were seen. Defaults to now.

        Returns:
            ``(record, was_newly_recorded)``. When the person was already
            logged for that date, the *existing* record is returned with
            ``False`` — the first sighting of the day is the one that
            counts, so a later frame can't overwrite an on-time check-in
            with a late one.

        Raises:
            ValueError: If ``person_id`` or ``name`` is empty.
        """
        if not person_id or not person_id.strip():
            raise ValueError("person_id must not be empty")
        if not name or not name.strip():
            raise ValueError("name must not be empty")

        moment = check_in or datetime.now()
        day = moment.date().isoformat()
        status = self._status_for(moment)

        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO attendance "
                "(person_id, name, date, check_in_time, status, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (person_id.strip(), name.strip(), day, moment.strftime("%H:%M:%S"), status, confidence),
            )
            was_new = cursor.rowcount > 0

            row = conn.execute(
                "SELECT person_id, name, date, check_in_time, status, confidence "
                "FROM attendance WHERE person_id = ? AND date = ?",
                (person_id.strip(), day),
            ).fetchone()

        return AttendanceRecord(*row), was_new

    def records_for(self, day: str | date_type | None = None) -> list[AttendanceRecord]:
        """All attendance records for one date, earliest check-in first.

        Args:
            day: The date to query. Defaults to today.

        Returns:
            That day's records.
        """
        target = self._resolve_date(day)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT person_id, name, date, check_in_time, status, confidence "
                "FROM attendance WHERE date = ? ORDER BY check_in_time ASC",
                (target,),
            ).fetchall()
        return [AttendanceRecord(*row) for row in rows]

    def daily_report(
        self, enrolled: list[tuple[str, str]], day: str | date_type | None = None
    ) -> DailyReport:
        """Build a day's report, deriving absences from who didn't appear.

        Args:
            enrolled: ``(person_id, name)`` for everyone enrolled.
            day: The date to report on. Defaults to today.

        Returns:
            The day's report.
        """
        target = self._resolve_date(day)
        records = self.records_for(target)
        seen_ids = {r.person_id for r in records}

        return DailyReport(
            date=target,
            present=[r for r in records if r.status == STATUS_PRESENT],
            late=[r for r in records if r.status == STATUS_LATE],
            absent=[(pid, name) for pid, name in enrolled if pid not in seen_ids],
        )

    def history_for(self, person_id: str, limit: int = 30) -> list[AttendanceRecord]:
        """One person's most recent attendance records, newest first.

        Args:
            person_id: Whose history to fetch.
            limit: Maximum records to return.

        Returns:
            Their records, newest date first.

        Raises:
            ValueError: If ``limit`` is not positive.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT person_id, name, date, check_in_time, status, confidence "
                "FROM attendance WHERE person_id = ? ORDER BY date DESC LIMIT ?",
                (person_id.strip(), limit),
            ).fetchall()
        return [AttendanceRecord(*row) for row in rows]

    def clear_date(self, day: str | date_type | None = None) -> int:
        """Delete all records for one date.

        Args:
            day: The date to clear. Defaults to today.

        Returns:
            How many records were deleted.
        """
        target = self._resolve_date(day)
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM attendance WHERE date = ?", (target,))
            return cursor.rowcount

    @staticmethod
    def _resolve_date(day: str | date_type | None) -> str:
        if day is None:
            return date_type.today().isoformat()
        if isinstance(day, date_type):
            return day.isoformat()
        return str(day)
