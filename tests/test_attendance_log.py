"""Tests for src.attendance_log."""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from src.attendance_log import (
    STATUS_LATE,
    STATUS_PRESENT,
    AttendanceLog,
)


@pytest.fixture()
def log(tmp_path) -> AttendanceLog:
    return AttendanceLog(db_path=tmp_path / "attendance.db", late_after=time(9, 0))


MORNING = datetime(2026, 9, 15, 8, 30, 0)
LATE_MORNING = datetime(2026, 9, 15, 9, 30, 0)
EXACTLY_NINE = datetime(2026, 9, 15, 9, 0, 0)
NEXT_DAY = datetime(2026, 9, 16, 8, 30, 0)


class TestRecord:
    def test_rejects_empty_person_id(self, log):
        with pytest.raises(ValueError):
            log.record("  ", "Ali", 0.9)

    def test_rejects_empty_name(self, log):
        with pytest.raises(ValueError):
            log.record("1", "", 0.9)

    def test_first_check_in_is_newly_recorded(self, log):
        record, was_new = log.record("1", "Ali", 0.9, check_in=MORNING)
        assert was_new is True
        assert record.name == "Ali"
        assert record.date == "2026-09-15"
        assert record.check_in_time == "08:30:00"

    def test_before_cutoff_is_present(self, log):
        record, _ = log.record("1", "Ali", 0.9, check_in=MORNING)
        assert record.status == STATUS_PRESENT

    def test_after_cutoff_is_late(self, log):
        record, _ = log.record("1", "Ali", 0.9, check_in=LATE_MORNING)
        assert record.status == STATUS_LATE

    def test_exactly_at_cutoff_counts_as_late(self, log):
        # Boundary case: "late_after 09:00" means 09:00:00 itself is late.
        record, _ = log.record("1", "Ali", 0.9, check_in=EXACTLY_NINE)
        assert record.status == STATUS_LATE

    def test_second_check_in_same_day_is_ignored(self, log):
        # A camera sees the same face in many consecutive frames — without
        # this, one person produces dozens of rows.
        log.record("1", "Ali", 0.9, check_in=MORNING)
        record, was_new = log.record("1", "Ali", 0.9, check_in=LATE_MORNING)

        assert was_new is False
        assert len(log.records_for("2026-09-15")) == 1

    def test_first_sighting_wins_over_later_one(self, log):
        # An on-time check-in must not be overwritten by a later frame
        # that would be classified as late.
        log.record("1", "Ali", 0.9, check_in=MORNING)
        record, _ = log.record("1", "Ali", 0.9, check_in=LATE_MORNING)
        assert record.status == STATUS_PRESENT
        assert record.check_in_time == "08:30:00"

    def test_same_person_next_day_is_a_new_record(self, log):
        log.record("1", "Ali", 0.9, check_in=MORNING)
        _, was_new = log.record("1", "Ali", 0.9, check_in=NEXT_DAY)
        assert was_new is True

    def test_different_people_same_day_both_recorded(self, log):
        log.record("1", "Ali", 0.9, check_in=MORNING)
        log.record("2", "Sara", 0.9, check_in=MORNING)
        assert len(log.records_for("2026-09-15")) == 2

    def test_confidence_is_stored(self, log):
        record, _ = log.record("1", "Ali", 0.87, check_in=MORNING)
        assert record.confidence == pytest.approx(0.87)

    def test_defaults_to_now_when_no_timestamp(self, log):
        record, _ = log.record("1", "Ali", 0.9)
        assert record.date == date.today().isoformat()


class TestRecordsFor:
    def test_empty_day_returns_empty_list(self, log):
        assert log.records_for("2026-01-01") == []

    def test_ordered_by_check_in_time(self, log):
        log.record("2", "Sara", 0.9, check_in=datetime(2026, 9, 15, 8, 55))
        log.record("1", "Ali", 0.9, check_in=datetime(2026, 9, 15, 8, 10))
        names = [r.name for r in log.records_for("2026-09-15")]
        assert names == ["Ali", "Sara"]

    def test_accepts_a_date_object(self, log):
        log.record("1", "Ali", 0.9, check_in=MORNING)
        assert len(log.records_for(date(2026, 9, 15))) == 1

    def test_only_returns_the_requested_day(self, log):
        log.record("1", "Ali", 0.9, check_in=MORNING)
        log.record("1", "Ali", 0.9, check_in=NEXT_DAY)
        assert len(log.records_for("2026-09-15")) == 1

    def test_defaults_to_today_when_no_date_given(self, log):
        log.record("1", "Ali", 0.9)  # defaults to now
        assert len(log.records_for()) == 1


class TestDailyReport:
    ENROLLED = [("1", "Ali"), ("2", "Sara"), ("3", "Omar")]

    def test_splits_present_late_and_absent(self, log):
        log.record("1", "Ali", 0.9, check_in=MORNING)
        log.record("2", "Sara", 0.9, check_in=LATE_MORNING)

        report = log.daily_report(self.ENROLLED, day="2026-09-15")

        assert [r.name for r in report.present] == ["Ali"]
        assert [r.name for r in report.late] == ["Sara"]
        assert report.absent == [("3", "Omar")]

    def test_absent_is_derived_not_stored(self, log):
        # Nobody checked in, so everyone enrolled is absent.
        report = log.daily_report(self.ENROLLED, day="2026-09-15")
        assert len(report.absent) == 3
        assert report.attendance_rate == 0.0

    def test_attendance_rate_counts_late_as_attended(self, log):
        log.record("1", "Ali", 0.9, check_in=MORNING)
        log.record("2", "Sara", 0.9, check_in=LATE_MORNING)
        report = log.daily_report(self.ENROLLED, day="2026-09-15")
        assert report.attendance_rate == pytest.approx(2 / 3)

    def test_total_enrolled_matches_input(self, log):
        report = log.daily_report(self.ENROLLED, day="2026-09-15")
        assert report.total_enrolled == 3

    def test_empty_enrollment_gives_zero_rate_not_division_error(self, log):
        report = log.daily_report([], day="2026-09-15")
        assert report.attendance_rate == 0.0
        assert report.total_enrolled == 0


class TestHistory:
    def test_returns_newest_first(self, log):
        log.record("1", "Ali", 0.9, check_in=MORNING)
        log.record("1", "Ali", 0.9, check_in=NEXT_DAY)
        history = log.history_for("1")
        assert [r.date for r in history] == ["2026-09-16", "2026-09-15"]

    def test_respects_limit(self, log):
        for day in range(1, 6):
            log.record("1", "Ali", 0.9, check_in=datetime(2026, 9, day, 8, 0))
        assert len(log.history_for("1", limit=2)) == 2

    def test_rejects_non_positive_limit(self, log):
        with pytest.raises(ValueError):
            log.history_for("1", limit=0)

    def test_unknown_person_has_empty_history(self, log):
        assert log.history_for("nobody") == []


class TestClearDate:
    def test_removes_only_that_day(self, log):
        log.record("1", "Ali", 0.9, check_in=MORNING)
        log.record("1", "Ali", 0.9, check_in=NEXT_DAY)

        deleted = log.clear_date("2026-09-15")

        assert deleted == 1
        assert log.records_for("2026-09-15") == []
        assert len(log.records_for("2026-09-16")) == 1

    def test_clearing_empty_day_returns_zero(self, log):
        assert log.clear_date("2026-01-01") == 0


class TestPersistence:
    def test_data_survives_reopening(self, tmp_path):
        path = tmp_path / "attendance.db"
        first = AttendanceLog(db_path=path)
        first.record("1", "Ali", 0.9, check_in=MORNING)

        second = AttendanceLog(db_path=path)
        assert len(second.records_for("2026-09-15")) == 1

    def test_custom_cutoff_is_respected(self, tmp_path):
        strict = AttendanceLog(db_path=tmp_path / "a.db", late_after=time(8, 0))
        record, _ = strict.record("1", "Ali", 0.9, check_in=MORNING)  # 08:30
        assert record.status == STATUS_LATE
