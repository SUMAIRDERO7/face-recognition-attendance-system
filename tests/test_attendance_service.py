"""Tests for src.attendance_service.

A fake encoder is injected so recognition outcomes are deterministic —
the service's job is orchestration (enroll, match, log, report), and
that logic is what these tests verify, independent of whether a real
face detector finds a real face.
"""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pytest

from src.attendance_log import STATUS_LATE, STATUS_PRESENT, AttendanceLog
from src.attendance_service import (
    DEFAULT_THRESHOLDS,
    AttendanceService,
    EnrollmentError,
)
from src.face_database import FaceDatabase
from src.face_encoder import EncodedFace, FaceLocation


class _FakeEncoder:
    """Returns pre-programmed faces regardless of the image passed in."""

    def __init__(self, faces: list[EncodedFace] | None = None, name: str = "fake"):
        self._faces = faces if faces is not None else []
        self._name = name
        self.calls: list[np.ndarray] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return 2

    def encode(self, image):
        self.calls.append(image)
        return self._faces


def _face(x: float, y: float, left: int = 0) -> EncodedFace:
    return EncodedFace(
        location=FaceLocation(top=0, right=left + 50, bottom=50, left=left),
        embedding=np.array([x, y], dtype=np.float32),
    )


IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)
MORNING = datetime(2026, 9, 15, 8, 30)
LATE = datetime(2026, 9, 15, 9, 30)


def _service(faces, tmp_path, threshold=0.5) -> AttendanceService:
    return AttendanceService(
        encoder=_FakeEncoder(faces),
        database=FaceDatabase(threshold=threshold),
        log=AttendanceLog(db_path=tmp_path / "attendance.db", late_after=time(9, 0)),
    )


class TestEnrollFromImage:
    def test_enrolls_a_single_face(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        assert service.enroll_from_image("1", "Ali", IMAGE) == 1
        assert len(service.database) == 1

    def test_repeat_enrollment_adds_a_sample(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        service.enroll_from_image("1", "Ali", IMAGE)
        assert service.enroll_from_image("1", "Ali", IMAGE) == 2
        assert len(service.database) == 1

    def test_no_face_raises(self, tmp_path):
        service = _service([], tmp_path)
        with pytest.raises(EnrollmentError, match="No face detected"):
            service.enroll_from_image("1", "Ali", IMAGE)

    def test_multiple_faces_raises(self, tmp_path):
        # Enrolling from a group photo would bind the wrong face to a
        # person's identity — it must be rejected, not silently guessed.
        service = _service([_face(1.0, 0.0), _face(0.0, 1.0, left=60)], tmp_path)
        with pytest.raises(EnrollmentError, match="2 faces"):
            service.enroll_from_image("1", "Ali", IMAGE)


class TestProcessFrame:
    def test_no_faces_yields_no_outcomes(self, tmp_path):
        assert _service([], tmp_path).process_frame(IMAGE) == []

    def test_recognized_face_is_recorded(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        service.enroll_from_image("1", "Ali", IMAGE)

        outcomes = service.process_frame(IMAGE, check_in=MORNING)

        assert len(outcomes) == 1
        assert outcomes[0].match.name == "Ali"
        assert outcomes[0].newly_recorded is True
        assert outcomes[0].record.status == STATUS_PRESENT

    def test_unknown_face_is_returned_not_dropped(self, tmp_path):
        service = _service([_face(50.0, 50.0)], tmp_path)
        service.database.enroll("1", "Ali", np.array([1.0, 0.0], dtype=np.float32))

        outcomes = service.process_frame(IMAGE, check_in=MORNING)

        assert len(outcomes) == 1
        assert outcomes[0].match.is_match is False
        assert outcomes[0].label == "Unknown"
        assert outcomes[0].record is None

    def test_unknown_face_is_never_logged(self, tmp_path):
        service = _service([_face(50.0, 50.0)], tmp_path)
        service.database.enroll("1", "Ali", np.array([1.0, 0.0], dtype=np.float32))
        service.process_frame(IMAGE, check_in=MORNING)
        assert service.log.records_for("2026-09-15") == []

    def test_repeated_frames_only_record_once(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        service.enroll_from_image("1", "Ali", IMAGE)

        first = service.process_frame(IMAGE, check_in=MORNING)
        second = service.process_frame(IMAGE, check_in=MORNING)

        assert first[0].newly_recorded is True
        assert second[0].newly_recorded is False
        assert len(service.log.records_for("2026-09-15")) == 1

    def test_record_false_recognizes_without_logging(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        service.enroll_from_image("1", "Ali", IMAGE)

        outcomes = service.process_frame(IMAGE, check_in=MORNING, record=False)

        assert outcomes[0].match.name == "Ali"
        assert outcomes[0].record is None
        assert service.log.records_for("2026-09-15") == []

    def test_late_check_in_is_marked_late(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        service.enroll_from_image("1", "Ali", IMAGE)
        outcomes = service.process_frame(IMAGE, check_in=LATE)
        assert outcomes[0].record.status == STATUS_LATE

    def test_multiple_faces_all_processed(self, tmp_path):
        service = AttendanceService(
            encoder=_FakeEncoder([_face(1.0, 0.0), _face(0.0, 1.0, left=60)]),
            database=FaceDatabase(threshold=0.5),
            log=AttendanceLog(db_path=tmp_path / "a.db"),
        )
        service.database.enroll("1", "Ali", np.array([1.0, 0.0], dtype=np.float32))
        service.database.enroll("2", "Sara", np.array([0.0, 1.0], dtype=np.float32))

        outcomes = service.process_frame(IMAGE, check_in=MORNING)

        assert {o.match.name for o in outcomes} == {"Ali", "Sara"}

    def test_label_marks_a_new_check_in(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        service.enroll_from_image("1", "Ali", IMAGE)
        outcomes = service.process_frame(IMAGE, check_in=MORNING)
        assert "Ali" in outcomes[0].label
        assert "✓" in outcomes[0].label


class TestDailyReport:
    def test_reports_present_and_absent(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        service.enroll_from_image("1", "Ali", IMAGE)
        service.database.enroll("2", "Sara", np.array([0.0, 1.0], dtype=np.float32))

        service.process_frame(IMAGE, check_in=MORNING)
        report = service.daily_report(day="2026-09-15")

        assert [r.name for r in report.present] == ["Ali"]
        assert report.absent == [("2", "Sara")]

    def test_empty_database_gives_empty_report(self, tmp_path):
        report = _service([], tmp_path).daily_report(day="2026-09-15")
        assert report.total_enrolled == 0


class TestBuild:
    def test_uses_encoder_appropriate_threshold(self, tmp_path):
        service = AttendanceService.build(
            encoder=_FakeEncoder(name="opencv"),
            db_path=tmp_path / "faces.json",
            attendance_db=tmp_path / "a.db",
        )
        assert service.database.threshold == DEFAULT_THRESHOLDS["opencv"]

    def test_explicit_threshold_overrides_default(self, tmp_path):
        service = AttendanceService.build(
            encoder=_FakeEncoder(name="opencv"),
            db_path=tmp_path / "faces.json",
            attendance_db=tmp_path / "a.db",
            threshold=0.99,
        )
        assert service.database.threshold == pytest.approx(0.99)

    def test_unknown_encoder_name_gets_a_fallback_threshold(self, tmp_path):
        service = AttendanceService.build(
            encoder=_FakeEncoder(name="mystery"),
            db_path=tmp_path / "faces.json",
            attendance_db=tmp_path / "a.db",
        )
        assert service.database.threshold == 0.6

    def test_loads_an_existing_database_from_disk(self, tmp_path):
        db_path = tmp_path / "faces.json"
        seed = FaceDatabase(threshold=0.5)
        seed.enroll("1", "Ali", np.array([1.0, 0.0], dtype=np.float32))
        seed.save(db_path)

        service = AttendanceService.build(
            encoder=_FakeEncoder(name="opencv"),
            db_path=db_path,
            attendance_db=tmp_path / "a.db",
        )

        assert len(service.database) == 1
        assert service.database.get("1").name == "Ali"


class TestSaveDatabase:
    def test_round_trips_through_disk(self, tmp_path):
        service = _service([_face(1.0, 0.0)], tmp_path)
        service.enroll_from_image("1", "Ali", IMAGE)

        path = service.save_database(tmp_path / "faces.json")
        restored = FaceDatabase.load(path)

        assert restored.get("1").name == "Ali"
