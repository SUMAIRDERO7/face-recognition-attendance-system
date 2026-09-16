"""
face_encoder.py
================

Turns a face image into a fixed-length numeric embedding that can be
compared against other faces by distance.

Two backends are provided behind one ``FaceEncoder`` protocol:

- :class:`DlibFaceEncoder` — the accurate option, using the
  ``face_recognition`` library (dlib's ResNet embeddings, 128-d). This is
  what a production system should use. It requires dlib, which ships only
  as a C++ source distribution and needs a multi-minute compile.
- :class:`OpenCvFaceEncoder` — a dependency-light fallback using OpenCV's
  bundled Haar cascade for detection plus a normalized, downscaled
  pixel-intensity vector for the embedding. It needs nothing beyond
  OpenCV, which is already a dependency for camera capture.

The honest difference, stated plainly rather than glossed over: the
OpenCV fallback compares faces on raw appearance, so it is far more
sensitive to lighting and pose than dlib's learned embeddings, and it
should not be trusted for real attendance. It exists so the full
pipeline — enrollment, matching, attendance logging, reporting — is
runnable and testable anywhere. See the README's "Which Encoder Am I
Getting?" section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class FaceEncoderError(RuntimeError):
    """Raised when a face cannot be detected or encoded."""


@dataclass(frozen=True, slots=True)
class FaceLocation:
    """A detected face's bounding box, in pixels."""

    top: int
    right: int
    bottom: int
    left: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def to_xywh(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, w, h)`` — the form OpenCV drawing calls expect."""
        return (self.left, self.top, self.width, self.height)


@dataclass(frozen=True, slots=True)
class EncodedFace:
    """One detected face with its embedding."""

    location: FaceLocation
    embedding: np.ndarray


class FaceEncoder(Protocol):
    """The interface the rest of the system depends on."""

    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def encode(self, image: np.ndarray) -> list[EncodedFace]: ...


class DlibFaceEncoder:
    """Accurate face embeddings via the ``face_recognition`` library.

    Args:
        model: dlib's detection model — ``"hog"`` (fast, CPU) or
            ``"cnn"`` (more accurate, much slower without a GPU).

    Raises:
        FaceEncoderError: If ``face_recognition`` is not installed.
    """

    def __init__(self, model: str = "hog") -> None:
        try:
            import face_recognition  # noqa: F401
        except ImportError as exc:
            raise FaceEncoderError(
                "face_recognition is not installed. It requires dlib, which "
                "compiles from source and takes several minutes. Install with "
                "`pip install face_recognition`, or use OpenCvFaceEncoder."
            ) from exc
        self.model = model

    @property
    def name(self) -> str:
        return "dlib"

    @property
    def dimensions(self) -> int:
        return 128

    def encode(self, image: np.ndarray) -> list[EncodedFace]:
        """Detect and encode every face in an RGB image.

        Args:
            image: An RGB image array.

        Returns:
            One :class:`EncodedFace` per detected face, in detection order.
            Empty list if no face is found.

        Raises:
            FaceEncoderError: If the underlying library fails.
        """
        import face_recognition

        try:
            locations = face_recognition.face_locations(image, model=self.model)
            embeddings = face_recognition.face_encodings(image, locations)
        except Exception as exc:
            raise FaceEncoderError(f"dlib encoding failed: {exc}") from exc

        return [
            EncodedFace(
                location=FaceLocation(top=t, right=r, bottom=b, left=l),
                embedding=np.asarray(emb, dtype=np.float32),
            )
            for (t, r, b, l), emb in zip(locations, embeddings)
        ]


class OpenCvFaceEncoder:
    """Dependency-light face embeddings using OpenCV only.

    Detection uses OpenCV's bundled frontal-face Haar cascade. The
    embedding is the detected face crop, converted to grayscale,
    histogram-equalized (which removes a lot of raw brightness variation),
    resized to a fixed square, flattened, and L2-normalized — so cosine
    distance between two embeddings is a meaningful appearance comparison.

    Args:
        crop_size: Side length the face crop is resized to. The embedding
            has ``crop_size ** 2`` dimensions.
        scale_factor: Haar cascade image pyramid scale.
        min_neighbors: Haar cascade neighbor threshold — higher means
            fewer false positives but more missed faces.

    Raises:
        ValueError: If ``crop_size`` is not positive.
        FaceEncoderError: If OpenCV's bundled cascade file can't be loaded.
    """

    def __init__(self, crop_size: int = 32, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")

        try:
            import cv2

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
        except (AttributeError, ImportError, OSError, RuntimeError) as exc:
            raise FaceEncoderError(
                "OpenCV face detection is unavailable. Reinstall "
                "opencv-python-headless so its Haar cascade is included."
            ) from exc
        if cascade.empty():
            raise FaceEncoderError(f"Could not load OpenCV cascade from {cascade_path}")

        self._cascade = cascade
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    @property
    def name(self) -> str:
        return "opencv"

    @property
    def dimensions(self) -> int:
        return self.crop_size**2

    def encode(self, image: np.ndarray) -> list[EncodedFace]:
        """Detect and encode every frontal face in an image.

        Args:
            image: An RGB or grayscale image array.

        Returns:
            One :class:`EncodedFace` per detected face. Empty list if no
            face is found.

        Raises:
            FaceEncoderError: If the image is malformed or detection fails.
        """
        import cv2

        if image is None or image.size == 0:
            raise FaceEncoderError("Cannot encode an empty image")

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        try:
            detections = self._cascade.detectMultiScale(
                gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors
            )
        except Exception as exc:
            raise FaceEncoderError(f"OpenCV face detection failed: {exc}") from exc

        faces: list[EncodedFace] = []
        for x, y, w, h in detections:
            crop = gray[y : y + h, x : x + w]
            embedding = self._embed_crop(crop)
            faces.append(
                EncodedFace(
                    location=FaceLocation(top=int(y), right=int(x + w), bottom=int(y + h), left=int(x)),
                    embedding=embedding,
                )
            )
        return faces

    def _embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Turn one grayscale face crop into an L2-normalized vector."""
        import cv2

        equalized = cv2.equalizeHist(crop)
        resized = cv2.resize(equalized, (self.crop_size, self.crop_size))
        vector = resized.astype(np.float32).flatten()

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            # A uniformly black crop has no direction to normalize toward.
            return vector
        return vector / norm


def build_default_encoder() -> FaceEncoder:
    """Return the best encoder available in this environment.

    Prefers :class:`DlibFaceEncoder` and silently falls back to
    :class:`OpenCvFaceEncoder` when dlib isn't installed — so the app
    runs everywhere, while automatically using the accurate backend
    wherever it's available.

    Returns:
        A ready-to-use encoder.
    """
    try:
        return DlibFaceEncoder()
    except FaceEncoderError:
        return OpenCvFaceEncoder()
