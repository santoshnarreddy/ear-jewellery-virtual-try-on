"""YOLO26n-Pose person / head / ear keypoint detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# COCO-Pose keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


@dataclass
class PoseDetection:
    """Single-person pose result in pixel coordinates."""

    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: np.ndarray  # (17, 3) x, y, conf
    conf: float

    @property
    def left_ear(self) -> Tuple[float, float, float]:
        x, y, c = self.keypoints[LEFT_EAR]
        return float(x), float(y), float(c)

    @property
    def right_ear(self) -> Tuple[float, float, float]:
        x, y, c = self.keypoints[RIGHT_EAR]
        return float(x), float(y), float(c)

    @property
    def nose(self) -> Tuple[float, float, float]:
        x, y, c = self.keypoints[NOSE]
        return float(x), float(y), float(c)

    def eye_distance(self) -> Optional[float]:
        le = self.keypoints[LEFT_EYE]
        re = self.keypoints[RIGHT_EYE]
        if le[2] < 0.2 or re[2] < 0.2:
            return None
        return float(np.hypot(le[0] - re[0], le[1] - re[1]))


class YoloPoseDetector:
    """Periodic YOLO26n-Pose wrapper."""

    def __init__(
        self,
        weights: str,
        conf: float = 0.35,
        iou: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self._last: List[PoseDetection] = []

    def detect(
        self, frame_bgr: np.ndarray, imgsz: Optional[int] = None
    ) -> List[PoseDetection]:
        kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
        }
        if imgsz is not None:
            kwargs["imgsz"] = int(imgsz)
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(frame_bgr, **kwargs)
        detections: List[PoseDetection] = []
        if not results:
            self._last = detections
            return detections

        r0 = results[0]
        if r0.boxes is None or r0.keypoints is None:
            self._last = detections
            return detections

        boxes = r0.boxes.xyxy.cpu().numpy()
        box_conf = r0.boxes.conf.cpu().numpy()
        kps = r0.keypoints.data.cpu().numpy()  # (N, 17, 3)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            detections.append(
                PoseDetection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    keypoints=kps[i].astype(np.float32),
                    conf=float(box_conf[i]),
                )
            )

        detections.sort(key=lambda d: d.conf, reverse=True)
        self._last = detections
        return detections

    @property
    def last(self) -> List[PoseDetection]:
        return self._last
