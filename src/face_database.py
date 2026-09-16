"""
face_database.py
=================

Stores enrolled people's face embeddings and matches an unknown face
against them.

Two design points worth naming:

**Multiple samples per person, averaged.** A single enrollment photo
encodes one pose under one lighting condition. Registering several
samples and comparing against their mean embedding is what makes
recognition survive the person turning their head or standing under a
different lamp. The database therefore stores every sample and derives
the mean, rather than overwriting.

**A match must clear a threshold, and "no match" is a valid answer.**
The nearest enrolled person is not automatically the right answer — if
the closest distance is still large, the correct output is *unknown*,
not the least-wrong name. An attendance system that always returns
somebody will happily mark the wrong person present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class FaceDatabaseError(RuntimeError):
    """Raised when the database can't be loaded, saved, or updated."""


@dataclass
class EnrolledPerson:
    """One registered person and every face sample enrolled for them."""

    person_id: str
    name: str
    samples: list[np.ndarray] = field(default_factory=list)
    enrolled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def mean_embedding(self) -> np.ndarray:
        """The average of all enrolled samples — the person's reference face.

        Raises:
            FaceDatabaseError: If the person has no samples.
        """
        if not self.samples:
            raise FaceDatabaseError(f"{self.name!r} has no enrolled face samples")
        return np.mean(np.stack(self.samples), axis=0)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The outcome of matching one unknown face against the database."""

    person_id: str | None
    name: str
    distance: float
    is_match: bool

    @property
    def confidence(self) -> float:
        """A 0-1 readability score derived from distance.

        This is a monotonic transform of distance for display purposes —
        it is explicitly *not* a calibrated probability, and the code
        never treats it as one.
        """
        return float(max(0.0, min(1.0, 1.0 - self.distance)))


UNKNOWN = MatchResult(person_id=None, name="Unknown", distance=float("inf"), is_match=False)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two embeddings.

    Args:
        a: First embedding.
        b: Second embedding.

    Returns:
        The distance as a float.

    Raises:
        ValueError: If the embeddings have different lengths — comparing
            a dlib embedding against an OpenCV one is a bug, not a
            near-miss, and should fail loudly.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Cannot compare embeddings of different shapes: {a.shape} vs {b.shape}. "
            "This usually means the database was enrolled with a different encoder."
        )
    return float(np.linalg.norm(a - b))


class FaceDatabase:
    """Holds enrolled people and matches unknown faces against them.

    Args:
        threshold: Maximum distance for a match to count. Anything
            farther is reported as unknown. The right value depends
            entirely on the encoder — dlib's 128-d embeddings and the
            OpenCV fallback's pixel vectors live on different scales, so
            this is set per-encoder by the caller, not guessed here.

    Raises:
        ValueError: If ``threshold`` is not positive.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self._people: dict[str, EnrolledPerson] = {}

    # ------------------------------------------------------------ enroll --

    def enroll(self, person_id: str, name: str, embedding: np.ndarray) -> EnrolledPerson:
        """Add a face sample for a person, creating them if new.

        Args:
            person_id: A stable unique id (e.g. a roll number).
            name: The person's display name.
            embedding: One face embedding to add as a sample.

        Returns:
            The person record, including the newly added sample.

        Raises:
            ValueError: If ``person_id`` or ``name`` is empty.
            FaceDatabaseError: If the embedding's shape doesn't match the
                samples already enrolled for that person.
        """
        if not person_id or not person_id.strip():
            raise ValueError("person_id must not be empty")
        if not name or not name.strip():
            raise ValueError("name must not be empty")

        person_id = person_id.strip()
        person = self._people.get(person_id)

        if person is None:
            person = EnrolledPerson(person_id=person_id, name=name.strip())
            self._people[person_id] = person
        elif person.samples and person.samples[0].shape != embedding.shape:
            raise FaceDatabaseError(
                f"Embedding shape {embedding.shape} does not match existing samples "
                f"{person.samples[0].shape} for {person.name!r}"
            )

        person.samples.append(np.asarray(embedding, dtype=np.float32))
        return person

    def remove(self, person_id: str) -> bool:
        """Delete a person and all their samples.

        Args:
            person_id: The person to remove.

        Returns:
            True if someone was removed, False if the id wasn't enrolled.
        """
        return self._people.pop(person_id.strip(), None) is not None

    # ------------------------------------------------------------- match --

    def match(self, embedding: np.ndarray) -> MatchResult:
        """Find the closest enrolled person to an unknown face.

        Args:
            embedding: The unknown face's embedding.

        Returns:
            The best match if it clears ``threshold``, otherwise a result
            with ``is_match=False`` and the name ``"Unknown"``. An empty
            database always yields unknown.
        """
        if not self._people:
            return UNKNOWN

        best: MatchResult = UNKNOWN
        for person in self._people.values():
            if not person.samples:
                continue
            distance = euclidean_distance(embedding, person.mean_embedding)
            if distance < best.distance:
                best = MatchResult(
                    person_id=person.person_id,
                    name=person.name,
                    distance=distance,
                    is_match=distance <= self.threshold,
                )

        # A near-miss is still a miss — report it as unknown but keep the
        # real distance so the UI can show how close it got.
        if not best.is_match and best.person_id is not None:
            return MatchResult(
                person_id=None, name="Unknown", distance=best.distance, is_match=False
            )
        return best

    # ----------------------------------------------------------- inspect --

    @property
    def people(self) -> list[EnrolledPerson]:
        """All enrolled people, ordered by name."""
        return sorted(self._people.values(), key=lambda p: p.name.lower())

    def get(self, person_id: str) -> EnrolledPerson | None:
        """Look up one person by id, or None if not enrolled."""
        return self._people.get(person_id.strip())

    def __len__(self) -> int:
        return len(self._people)

    # -------------------------------------------------------- persistence --

    def save(self, path: str | Path) -> Path:
        """Write the database to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            The path written to.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threshold": self.threshold,
            "people": [
                {
                    "person_id": p.person_id,
                    "name": p.name,
                    "enrolled_at": p.enrolled_at,
                    "samples": [s.tolist() for s in p.samples],
                }
                for p in self._people.values()
            ],
        }
        destination.write_text(json.dumps(payload))
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FaceDatabase":
        """Load a database previously written by :meth:`save`.

        Args:
            path: The JSON file to load.

        Returns:
            The restored database.

        Raises:
            FaceDatabaseError: If the file is missing or malformed.
        """
        source = Path(path)
        if not source.exists():
            raise FaceDatabaseError(f"Face database not found: {source}")

        try:
            payload = json.loads(source.read_text())
            database = cls(threshold=payload.get("threshold", 0.6))
            for record in payload["people"]:
                person = EnrolledPerson(
                    person_id=record["person_id"],
                    name=record["name"],
                    samples=[np.asarray(s, dtype=np.float32) for s in record["samples"]],
                    enrolled_at=record.get("enrolled_at", ""),
                )
                database._people[person.person_id] = person
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise FaceDatabaseError(f"Malformed face database at {source}: {exc}") from exc

        return database
