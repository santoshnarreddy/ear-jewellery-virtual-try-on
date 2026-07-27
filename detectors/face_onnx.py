"""
MediaPipe Face Landmarks via ONNX.

Uses ``face_landmarks_detector.onnx`` (converted from face_landmarker.task).
Face *box* comes from YOLO pose (stable) — BlazeFace ONNX decode was misplacing
the ROI; landmarks still run as ONNX on a head crop so the mesh topology matches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from detectors.mediapipe_face import FaceGeometry
from detectors.yolo_pose import LEFT_EAR, NOSE, RIGHT_EAR, LEFT_EYE, RIGHT_EYE, PoseDetection


class MediaPipeFaceOnnxDetector:
    """478-pt face mesh (ONNX) on a YOLO-derived head crop."""

    def __init__(
        self,
        detector_onnx: str,
        landmarks_onnx: str,
        score_threshold: float = 0.5,
        providers: Optional[list[str]] = None,
    ) -> None:
        import onnxruntime as ort

        # detector_onnx kept for API/manifest compatibility (optional BlazeFace).
        self._detector_path = Path(detector_onnx).resolve()
        lm_p = Path(landmarks_onnx).resolve()
        if not lm_p.is_file():
            raise FileNotFoundError(
                f"Face landmarks ONNX missing: {lm_p}\nRun: python export_all_onnx.py"
            )

        avail = ort.get_available_providers()
        if providers is None:
            preferred = [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in preferred if p in avail] or ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.lm = ort.InferenceSession(
                str(lm_p), sess_options=so, providers=providers
            )
        except Exception:
            self.lm = ort.InferenceSession(
                str(lm_p),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
        self.lm_in = self.lm.get_inputs()[0].name
        self.score_threshold = score_threshold
        self._pose: Optional[PoseDetection] = None
        print(
            f"[Face-ONNX] landmarks={lm_p.name} (head crop from YOLO) "
            f"providers={self.lm.get_providers()}"
        )

    def close(self) -> None:
        pass

    def set_pose(self, pose: Optional[PoseDetection]) -> None:
        """Provide latest YOLO pose so we can build a head/ear-aware crop."""
        self._pose = pose

    def _head_box(
        self, frame_shape: Tuple[int, ...], pose: PoseDetection
    ) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame_shape[:2]
        k = pose.keypoints
        pts = []
        for idx in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR):
            x, y, c = k[idx]
            if c >= 0.2:
                pts.append((float(x), float(y)))
        if len(pts) < 2:
            x1, y1, x2, y2 = pose.bbox
            # upper portion of person box ≈ head
            hh = y2 - y1
            return (
                int(max(0, x1)),
                int(max(0, y1)),
                int(min(w, x2)),
                int(min(h, y1 + 0.55 * hh)),
            )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        span = max(max(xs) - min(xs), max(ys) - min(ys), 40.0)
        side = span * 2.8
        x1 = int(max(0, cx - side / 2))
        y1 = int(max(0, cy - side / 2))
        x2 = int(min(w, cx + side / 2))
        y2 = int(min(h, cy + side / 2))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        return (x1, y1, x2, y2)

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceGeometry]:
        pose = self._pose
        if pose is None:
            return None
        box = self._head_box(frame_bgr.shape, pose)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_r = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        outs = self.lm.run(None, {self.lm_in: rgb[None, ...]})
        flat = outs[0].reshape(-1)
        n = flat.size // 3
        if n < 468:
            return None
        pts = flat[: n * 3].reshape(n, 3).astype(np.float32)
        sx = (x2 - x1) / 256.0
        sy = (y2 - y1) / 256.0
        pts[:, 0] = pts[:, 0] * sx + x1
        pts[:, 1] = pts[:, 1] * sy + y1
        pts[:, 2] = pts[:, 2] * sx

        xs, ys = pts[:, 0], pts[:, 1]
        bx1, by1 = float(xs.min()), float(ys.min())
        bx2, by2 = float(xs.max()), float(ys.max())
        face_w = max(1.0, bx2 - bx1)
        face_h = max(1.0, by2 - by1)
        nose = pts[1, 0]
        mid = 0.5 * (pts[234, 0] + pts[454, 0])
        yaw = float(np.clip((nose - mid) / face_w * -90.0, -90, 90))

        return FaceGeometry(
            landmarks=pts,
            bbox=(bx1, by1, bx2, by2),
            yaw_deg=yaw,
            pitch_deg=0.0,
            roll_deg=0.0,
            face_width=face_w,
            face_height=face_h,
        )
