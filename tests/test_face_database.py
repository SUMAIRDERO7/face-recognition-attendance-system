"""Tests for src.face_database."""

from __future__ import annotations

import numpy as np
import pytest

from src.face_database import (
    EnrolledPerson,
    FaceDatabase,
    FaceDatabaseError,
    euclidean_distance,
)


def _emb(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


class TestEuclideanDistance:
    def test_identical_embeddings_have_zero_distance(self):
        assert euclidean_distance(_emb(1, 2, 3), _emb(1, 2, 3)) == 0.0

    def test_known_distance_is_correct(self):
        # 3-4-5 triangle: hand-verified.
        assert euclidean_distance(_emb(0, 0), _emb(3, 4)) == pytest.approx(5.0)

    def test_mismatched_shapes_raise(self):
        # Comparing a dlib embedding to an OpenCV one is a bug, not a
        # near-miss — it must fail loudly rather than silently compare.
        with pytest.raises(ValueError, match="different shapes"):
            euclidean_distance(_emb(1, 2), _emb(1, 2, 3))


class TestEnrolledPerson:
    def test_mean_of_single_sample_is_that_sample(self):
        person = EnrolledPerson(person_id="1", name="Ali", samples=[_emb(1, 2)])
        np.testing.assert_allclose(person.mean_embedding, _emb(1, 2))

    def test_mean_averages_multiple_samples(self):
        person = EnrolledPerson(person_id="1", name="Ali", samples=[_emb(0, 0), _emb(2, 4)])
        np.testing.assert_allclose(person.mean_embedding, _emb(1, 2))

    def test_mean_without_samples_raises(self):
        with pytest.raises(FaceDatabaseError):
            EnrolledPerson(person_id="1", name="Ali").mean_embedding

    def test_sample_count(self):
        person = EnrolledPerson(person_id="1", name="Ali", samples=[_emb(1), _emb(2)])
        assert person.sample_count == 2


class TestEnroll:
    def test_rejects_empty_person_id(self):
        db = FaceDatabase()
        with pytest.raises(ValueError):
            db.enroll("  ", "Ali", _emb(1, 2))

    def test_rejects_empty_name(self):
        db = FaceDatabase()
        with pytest.raises(ValueError):
            db.enroll("1", "", _emb(1, 2))

    def test_first_enrollment_creates_person(self):
        db = FaceDatabase()
        person = db.enroll("1", "Ali", _emb(1, 2))
        assert person.name == "Ali"
        assert len(db) == 1

    def test_repeat_enrollment_appends_sample_not_new_person(self):
        db = FaceDatabase()
        db.enroll("1", "Ali", _emb(1, 2))
        person = db.enroll("1", "Ali", _emb(3, 4))
        assert person.sample_count == 2
        assert len(db) == 1

    def test_mismatched_embedding_shape_raises(self):
        db = FaceDatabase()
        db.enroll("1", "Ali", _emb(1, 2))
        with pytest.raises(FaceDatabaseError, match="does not match"):
            db.enroll("1", "Ali", _emb(1, 2, 3))

    def test_ids_are_stripped(self):
        db = FaceDatabase()
        db.enroll("  1  ", "Ali", _emb(1, 2))
        assert db.get("1") is not None


class TestMatch:
    def test_empty_database_returns_unknown(self):
        assert FaceDatabase().match(_emb(1, 2)).is_match is False

    def test_exact_match_is_found(self):
        db = FaceDatabase(threshold=0.5)
        db.enroll("1", "Ali", _emb(1, 0))
        result = db.match(_emb(1, 0))
        assert result.is_match
        assert result.name == "Ali"
        assert result.distance == pytest.approx(0.0)

    def test_closest_person_wins(self):
        db = FaceDatabase(threshold=10.0)
        db.enroll("1", "Ali", _emb(0, 0))
        db.enroll("2", "Sara", _emb(10, 10))
        assert db.match(_emb(0.5, 0.5)).name == "Ali"

    def test_beyond_threshold_is_unknown_not_nearest_name(self):
        # The critical rule: an attendance system must never mark the
        # least-wrong person present.
        db = FaceDatabase(threshold=0.5)
        db.enroll("1", "Ali", _emb(0, 0))
        result = db.match(_emb(100, 100))
        assert result.is_match is False
        assert result.name == "Unknown"
        assert result.person_id is None

    def test_unknown_result_still_reports_real_distance(self):
        db = FaceDatabase(threshold=0.5)
        db.enroll("1", "Ali", _emb(0, 0))
        result = db.match(_emb(3, 4))
        assert result.distance == pytest.approx(5.0)  # not infinity

    def test_match_uses_mean_of_samples(self):
        db = FaceDatabase(threshold=0.2)
        db.enroll("1", "Ali", _emb(0, 0))
        db.enroll("1", "Ali", _emb(2, 0))
        # Mean is (1, 0) — an exact hit there, but far from either sample.
        assert db.match(_emb(1, 0)).is_match

    def test_person_with_no_samples_is_skipped(self):
        db = FaceDatabase(threshold=10.0)
        db.enroll("1", "Ali", _emb(1, 1))
        db._people["2"] = EnrolledPerson(person_id="2", name="Empty", samples=[])
        assert db.match(_emb(1, 1)).name == "Ali"

    def test_confidence_is_bounded_between_zero_and_one(self):
        db = FaceDatabase(threshold=10.0)
        db.enroll("1", "Ali", _emb(0, 0))
        assert 0.0 <= db.match(_emb(0, 0)).confidence <= 1.0
        assert 0.0 <= db.match(_emb(50, 50)).confidence <= 1.0


class TestThreshold:
    def test_rejects_non_positive_threshold(self):
        with pytest.raises(ValueError):
            FaceDatabase(threshold=0)
        with pytest.raises(ValueError):
            FaceDatabase(threshold=-1)


class TestRemoveAndInspect:
    def test_remove_deletes_person(self):
        db = FaceDatabase()
        db.enroll("1", "Ali", _emb(1, 2))
        assert db.remove("1") is True
        assert len(db) == 0

    def test_remove_unknown_id_returns_false(self):
        assert FaceDatabase().remove("nobody") is False

    def test_people_sorted_by_name(self):
        db = FaceDatabase()
        db.enroll("1", "Zara", _emb(1))
        db.enroll("2", "Ali", _emb(1))
        assert [p.name for p in db.people] == ["Ali", "Zara"]

    def test_get_returns_none_for_unknown(self):
        assert FaceDatabase().get("nobody") is None


class TestPersistence:
    def test_round_trip_preserves_people_and_samples(self, tmp_path):
        db = FaceDatabase(threshold=0.42)
        db.enroll("1", "Ali", _emb(1, 2, 3))
        db.enroll("1", "Ali", _emb(4, 5, 6))
        db.enroll("2", "Sara", _emb(7, 8, 9))

        path = db.save(tmp_path / "nested" / "faces.json")
        restored = FaceDatabase.load(path)

        assert len(restored) == 2
        assert restored.threshold == pytest.approx(0.42)
        ali = restored.get("1")
        assert ali.sample_count == 2
        np.testing.assert_allclose(ali.samples[0], _emb(1, 2, 3))

    def test_restored_database_matches_identically(self, tmp_path):
        db = FaceDatabase(threshold=0.5)
        db.enroll("1", "Ali", _emb(1, 0))
        restored = FaceDatabase.load(db.save(tmp_path / "faces.json"))
        assert restored.match(_emb(1, 0)).name == "Ali"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FaceDatabaseError, match="not found"):
            FaceDatabase.load(tmp_path / "nope.json")

    def test_malformed_file_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(FaceDatabaseError, match="Malformed"):
            FaceDatabase.load(bad)

    def test_valid_json_wrong_shape_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"threshold": 0.5, "people": [{"no_person_id": true}]}')
        with pytest.raises(FaceDatabaseError):
            FaceDatabase.load(bad)
