"""MediaPipe Face Landmarker for face geometry and head orientation."""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Face Mesh indices near the ears / face outline (subject's left/right).
# MediaPipe: image coords; landmark 234 ≈ left cheek near ear, 454 ≈ right.
LEFT_EAR_REGION = (234, 127, 162, 21, 54, 103, 67, 109)
RIGHT_EAR_REGION = (454, 356, 389, 251, 284, 332, 297, 338)
# Face oval extremes for size
FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class FaceGeometry:
    """Face landmarks and derived head metrics in pixel space."""

    landmarks: np.ndarray  # (N, 3) x, y, z in pixels (z relative)
    bbox: Tuple[float, float, float, float]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    face_width: float
    face_height: float

    def region_center(self, indices: Tuple[int, ...]) -> Tuple[float, float]:
        pts = self.landmarks[list(indices), :2]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    @property
    def left_ear_hint(self) -> Tuple[float, float]:
        return self.region_center(LEFT_EAR_REGION)

    @property
    def right_ear_hint(self) -> Tuple[float, float]:
        return self.region_center(RIGHT_EAR_REGION)


def ensure_face_landmarker_model(path: str) -> str:
    """Download Face Landmarker .task bundle if missing."""
    import ssl
    import subprocess

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file() and p.stat().st_size > 1000:
        return str(p)

    print(f"[MediaPipe] Downloading Face Landmarker model → {p}")
    # Prefer curl (handles system certs on macOS better than some Python builds).
    try:
        subprocess.run(
            ["curl", "-fsSL", "-o", str(p), MODEL_URL],
            check=True,
        )
        if p.is_file() and p.stat().st_size > 1000:
            return str(p)
    except (OSError, subprocess.CalledProcessError):
        pass

    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()

    with urllib.request.urlopen(MODEL_URL, context=ctx) as resp, open(p, "wb") as f:
        f.write(resp.read())
    return str(p)


def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract yaw/pitch/roll (degrees) from a 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-R[2, 1], R[2, 2])
        yaw = math.atan2(R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[2, 1], R[2, 2])
        yaw = 0.0
        roll = math.atan2(-R[0, 1], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class MediaPipeFaceDetector:
    """Face Landmarker (VIDEO) with transformation-matrix head pose."""

    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = ensure_face_landmarker_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._vision = vision
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceGeometry]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        coords = np.zeros((len(lm), 3), dtype=np.float32)
        for i, p in enumerate(lm):
            coords[i, 0] = p.x * w
            coords[i, 1] = p.y * h
            coords[i, 2] = p.z * w

        xs, ys = coords[:, 0], coords[:, 1]
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)

        yaw = pitch = roll = 0.0
        if result.facial_transformation_matrixes:
            M = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
            if M.shape == (4, 4):
                yaw, pitch, roll = _rotation_matrix_to_euler(M[:3, :3])

        return FaceGeometry(
            landmarks=coords,
            bbox=(x1, y1, x2, y2),
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            face_width=face_w,
            face_height=face_h,
        )
