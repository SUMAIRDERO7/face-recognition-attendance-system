"""
attendance_service.py
======================

The orchestration layer: wires the face encoder, the enrolled-face
database, and the attendance log into the two operations the UIs need —
enrolling a person, and processing a frame into attendance records.

Contains no camera or UI code, so it's shared identically by the
Streamlit app and the CLI, and is fully testable with synthetic images.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.attendance_log import AttendanceLog, AttendanceRecord, DailyReport
from src.face_database import FaceDatabase, MatchResult
from src.face_encoder import FaceEncoder, FaceLocation

# Sensible starting thresholds per encoder. They differ because the two
# encoders' embeddings live on completely different scales — dlib's are
# learned 128-d vectors where ~0.6 is the community-standard cutoff,
# while the OpenCV fallback's are L2-normalized pixel vectors where
# distances are much smaller. Both are tunable by the caller.
DEFAULT_THRESHOLDS = {"dlib": 0.6, "opencv": 0.35}


class EnrollmentError(RuntimeError):
    """Raised when a person can't be enrolled from a given image."""


@dataclass(frozen=True, slots=True)
class RecognitionOutcome:
    """One face found in a frame, with its match and what was logged."""

    location: FaceLocation
    match: MatchResult
    record: AttendanceRecord | None
    newly_recorded: bool

    @property
    def label(self) -> str:
        """The display label for drawing on the frame."""
        if not self.match.is_match:
            return "Unknown"
        suffix = " ✓" if self.newly_recorded else ""
        return f"{self.match.name}{suffix}"


class AttendanceService:
    """Coordinates enrollment and attendance recognition.

    Args:
        encoder: The face encoder to use for all operations.
        database: The enrolled-face database.
        log: The attendance log.
    """

    def __init__(self, encoder: FaceEncoder, database: FaceDatabase, log: AttendanceLog) -> None:
        self.encoder = encoder
        self.database = database
        self.log = log

    @classmethod
    def build(
        cls,
        encoder: FaceEncoder,
        db_path: str | Path = "data/faces.json",
        attendance_db: str | Path = "data/attendance.db",
        threshold: float | None = None,
    ) -> "AttendanceService":
        """Build a service, loading any existing face database from disk.

        Args:
            encoder: The face encoder to use.
            db_path: Where the enrolled-face JSON lives.
            attendance_db: Where the attendance SQLite file lives.
            threshold: Match threshold. Defaults to the encoder-appropriate
                value from :data:`DEFAULT_THRESHOLDS`.

        Returns:
            A ready service.
        """
        resolved_threshold = (
            threshold if threshold is not None else DEFAULT_THRESHOLDS.get(encoder.name, 0.6)
        )

        path = Path(db_path)
        if path.exists():
            database = FaceDatabase.load(path)
            database.threshold = resolved_threshold
        else:
            database = FaceDatabase(threshold=resolved_threshold)

        return cls(encoder=encoder, database=database, log=AttendanceLog(attendance_db))

    # --------------------------------------------------------- enrollment --

    def enroll_from_image(self, person_id: str, name: str, image: np.ndarray) -> int:
        """Enroll one face sample for a person from an image.

        Args:
            person_id: Their stable unique id.
            name: Their display name.
            image: An image containing exactly one clearly visible face.

        Returns:
            How many samples that person now has enrolled.

        Raises:
            EnrollmentError: If no face is found, or more than one is —
                enrolling from a group photo would silently bind the wrong
                face to the person's identity, so it's rejected outright.
        """
        faces = self.encoder.encode(image)

        if not faces:
            raise EnrollmentError(
                "No face detected in the image. Use a clear, well-lit, front-facing photo."
            )
        if len(faces) > 1:
            raise EnrollmentError(
                f"Found {len(faces)} faces. Enroll from a photo containing only "
                "the person being registered."
            )

        person = self.database.enroll(person_id, name, faces[0].embedding)
        return person.sample_count

    def save_database(self, db_path: str | Path = "data/faces.json") -> Path:
        """Persist the enrolled-face database to disk."""
        return self.database.save(db_path)

    # -------------------------------------------------------- recognition --

    def process_frame(
        self, image: np.ndarray, check_in: datetime | None = None, record: bool = True
    ) -> list[RecognitionOutcome]:
        """Recognize every face in a frame and log attendance for matches.

        Args:
            image: The frame to process.
            check_in: Timestamp to record. Defaults to now.
            record: If False, recognize without writing anything — used
                for a live preview that shouldn't mark attendance.

        Returns:
            One outcome per detected face, in detection order. Faces that
            don't match anyone enrolled are returned as unknown rather
            than dropped, so the UI can show them.
        """
        outcomes: list[RecognitionOutcome] = []

        for face in self.encoder.encode(image):
            match = self.database.match(face.embedding)

            attendance_record = None
            newly_recorded = False
            if record and match.is_match and match.person_id is not None:
                attendance_record, newly_recorded = self.log.record(
                    person_id=match.person_id,
                    name=match.name,
                    confidence=match.confidence,
                    check_in=check_in,
                )

            outcomes.append(
                RecognitionOutcome(
                    location=face.location,
                    match=match,
                    record=attendance_record,
                    newly_recorded=newly_recorded,
                )
            )

        return outcomes

    # ----------------------------------------------------------- reporting --

    def daily_report(self, day: str | None = None) -> DailyReport:
        """Build a day's attendance report across everyone enrolled.

        Args:
            day: The date to report on. Defaults to today.

        Returns:
            The report, with absences derived from the enrolled list.
        """
        enrolled = [(p.person_id, p.name) for p in self.database.people]
        return self.log.daily_report(enrolled, day=day)
