"""Tests for src.face_encoder.

The OpenCV encoder is exercised against real numpy image arrays through
the real cascade — not mocked — since the thing worth verifying is that
the embedding pipeline produces correctly-shaped, normalized vectors.

Note on what is deliberately NOT asserted: these tests do not claim the
Haar cascade detects faces in synthetic drawings. It doesn't, and it
shouldn't — it's trained on photographs. Detection accuracy on real
photos is an empirical property of the cascade, not of this code, so
asserting it here would be testing OpenCV rather than this project.
What *is* tested is every branch this project owns: the crop → equalize
→ resize → flatten → normalize pipeline, the error paths, and the
protocol contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.face_encoder import (
    DlibFaceEncoder,
    FaceEncoderError,
    FaceLocation,
    OpenCvFaceEncoder,
    build_default_encoder,
)


class TestFaceLocation:
    def test_width_and_height(self):
        location = FaceLocation(top=10, right=110, bottom=160, left=20)
        assert location.width == 90
        assert location.height == 150

    def test_to_xywh_matches_opencv_convention(self):
        location = FaceLocation(top=10, right=110, bottom=160, left=20)
        assert location.to_xywh() == (20, 10, 90, 150)


class TestOpenCvFaceEncoderInit:
    def test_rejects_non_positive_crop_size(self):
        with pytest.raises(ValueError):
            OpenCvFaceEncoder(crop_size=0)

    def test_name_and_dimensions(self):
        encoder = OpenCvFaceEncoder(crop_size=16)
        assert encoder.name == "opencv"
        assert encoder.dimensions == 256  # 16 * 16

    def test_missing_cascade_file_raises(self, monkeypatch):
        import cv2

        class _EmptyCascade:
            def __init__(self, path):
                pass

            def empty(self):
                return True

        monkeypatch.setattr(cv2, "CascadeClassifier", _EmptyCascade)
        with pytest.raises(FaceEncoderError, match="Could not load OpenCV cascade"):
            OpenCvFaceEncoder()


class TestOpenCvEncode:
    def test_empty_image_raises(self):
        encoder = OpenCvFaceEncoder()
        with pytest.raises(FaceEncoderError, match="empty image"):
            encoder.encode(np.array([], dtype=np.uint8))

    def test_none_image_raises(self):
        encoder = OpenCvFaceEncoder()
        with pytest.raises(FaceEncoderError):
            encoder.encode(None)

    def test_blank_image_yields_no_faces(self):
        encoder = OpenCvFaceEncoder()
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        assert encoder.encode(blank) == []

    def test_accepts_grayscale_input(self):
        encoder = OpenCvFaceEncoder()
        gray = np.zeros((200, 200), dtype=np.uint8)
        assert encoder.encode(gray) == []  # no crash on 2-D input

    def test_detection_failure_is_wrapped(self):
        encoder = OpenCvFaceEncoder()

        class _BoomCascade:
            def detectMultiScale(self, *args, **kwargs):
                raise RuntimeError("cascade exploded")

        # The whole cascade object is swapped rather than patching its
        # method — cv2.CascadeClassifier is a C extension whose attributes
        # are read-only.
        encoder._cascade = _BoomCascade()
        with pytest.raises(FaceEncoderError, match="detection failed"):
            encoder.encode(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_detected_faces_are_encoded_with_locations(self):
        encoder = OpenCvFaceEncoder(crop_size=8)

        class _StubCascade:
            def detectMultiScale(self, gray, **kwargs):
                return [(10, 20, 30, 40)]  # x, y, w, h

        encoder._cascade = _StubCascade()
        faces = encoder.encode(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

        assert len(faces) == 1
        location = faces[0].location
        assert location.left == 10
        assert location.top == 20
        assert location.width == 30
        assert location.height == 40
        assert faces[0].embedding.shape == (64,)


class TestOpenCvEmbedding:
    """The crop → embedding pipeline, tested directly on known crops."""

    def test_embedding_has_expected_dimensions(self):
        encoder = OpenCvFaceEncoder(crop_size=8)
        crop = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        assert encoder._embed_crop(crop).shape == (64,)

    def test_embedding_is_l2_normalized(self):
        encoder = OpenCvFaceEncoder(crop_size=8)
        crop = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        embedding = encoder._embed_crop(crop)
        assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-5)

    def test_all_black_crop_does_not_divide_by_zero(self):
        # A uniformly black crop has zero norm — normalizing would be a
        # division by zero, so it must be returned unnormalized instead.
        encoder = OpenCvFaceEncoder(crop_size=8)
        black = np.zeros((50, 50), dtype=np.uint8)
        embedding = encoder._embed_crop(black)
        assert np.all(np.isfinite(embedding))

    def test_identical_crops_give_identical_embeddings(self):
        encoder = OpenCvFaceEncoder(crop_size=8)
        crop = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        np.testing.assert_allclose(encoder._embed_crop(crop), encoder._embed_crop(crop.copy()))

    def test_different_crops_give_different_embeddings(self):
        encoder = OpenCvFaceEncoder(crop_size=8)
        a = encoder._embed_crop(np.full((50, 50), 30, dtype=np.uint8))
        b = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        assert not np.allclose(a, encoder._embed_crop(b))

    def test_crop_size_is_respected_regardless_of_input_size(self):
        encoder = OpenCvFaceEncoder(crop_size=12)
        small = encoder._embed_crop(np.random.randint(0, 255, (20, 20), dtype=np.uint8))
        large = encoder._embed_crop(np.random.randint(0, 255, (300, 300), dtype=np.uint8))
        assert small.shape == large.shape == (144,)


class TestDlibFaceEncoder:
    def test_raises_clear_error_when_unavailable(self):
        try:
            import face_recognition  # noqa: F401
        except ImportError:
            with pytest.raises(FaceEncoderError, match="dlib"):
                DlibFaceEncoder()
        else:
            encoder = DlibFaceEncoder()
            assert encoder.name == "dlib"
            assert encoder.dimensions == 128


class TestDlibEncodeWithStubbedLibrary:
    """Covers the dlib code path by injecting a stub `face_recognition`.

    dlib ships only as a C++ source distribution and takes many minutes to
    compile, so it isn't installed in every environment (including the one
    this was built in). Stubbing the module lets the wrapper's own logic —
    location/embedding zipping, error wrapping — be verified regardless.
    """

    @pytest.fixture()
    def stub_module(self, monkeypatch):
        import sys
        import types

        module = types.ModuleType("face_recognition")
        module.face_locations = lambda image, model="hog": [(10, 110, 160, 20)]
        module.face_encodings = lambda image, locations: [np.ones(128, dtype=np.float32)]
        monkeypatch.setitem(sys.modules, "face_recognition", module)
        return module

    def test_encoder_reports_dlib_identity(self, stub_module):
        encoder = DlibFaceEncoder()
        assert encoder.name == "dlib"
        assert encoder.dimensions == 128

    def test_encode_maps_locations_and_embeddings(self, stub_module):
        encoder = DlibFaceEncoder()
        faces = encoder.encode(np.zeros((200, 200, 3), dtype=np.uint8))

        assert len(faces) == 1
        location = faces[0].location
        assert (location.top, location.right, location.bottom, location.left) == (10, 110, 160, 20)
        assert faces[0].embedding.shape == (128,)

    def test_no_faces_yields_empty_list(self, stub_module):
        stub_module.face_locations = lambda image, model="hog": []
        stub_module.face_encodings = lambda image, locations: []
        assert DlibFaceEncoder().encode(np.zeros((50, 50, 3), dtype=np.uint8)) == []

    def test_library_failure_is_wrapped(self, stub_module):
        def _boom(*args, **kwargs):
            raise RuntimeError("dlib exploded")

        stub_module.face_locations = _boom
        with pytest.raises(FaceEncoderError, match="dlib encoding failed"):
            DlibFaceEncoder().encode(np.zeros((50, 50, 3), dtype=np.uint8))

    def test_model_choice_is_passed_through(self, stub_module):
        captured = {}

        def _capture(image, model="hog"):
            captured["model"] = model
            return []

        stub_module.face_locations = _capture
        stub_module.face_encodings = lambda image, locations: []

        DlibFaceEncoder(model="cnn").encode(np.zeros((50, 50, 3), dtype=np.uint8))
        assert captured["model"] == "cnn"

    def test_build_default_prefers_dlib_when_available(self, stub_module):
        assert build_default_encoder().name == "dlib"


class TestBuildDefaultEncoder:
    def test_returns_a_usable_encoder(self):
        encoder = build_default_encoder()
        assert encoder.name in {"dlib", "opencv"}
        assert encoder.dimensions > 0

    def test_falls_back_to_opencv_when_dlib_missing(self, monkeypatch):
        from src import face_encoder

        def _unavailable(*args, **kwargs):
            raise FaceEncoderError("dlib not installed")

        monkeypatch.setattr(face_encoder, "DlibFaceEncoder", _unavailable)
        assert face_encoder.build_default_encoder().name == "opencv"
