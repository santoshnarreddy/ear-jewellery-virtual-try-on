"""YOLO26n-Pose via Ultralytics ONNX Runtime backend (correct letterbox / NMS)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from detectors.yolo_pose import PoseDetection


class YoloPoseOnnxDetector:
    """
    Load ``yolo26n-pose.onnx`` through Ultralytics so decode matches ``.pt``.

    Custom raw-ORT decoding previously misplaced boxes/keypoints (ROI off-ear).
    """

    def __init__(
        self,
        onnx_path: str,
        conf: float = 0.35,
        iou: float = 0.45,
        device: Optional[str] = None,
        **_: object,
    ) -> None:
        from ultralytics import YOLO

        path = Path(onnx_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLO ONNX not found: {path}\nRun: python export_all_onnx.py"
            )
        self.model = YOLO(str(path), task="pose")
        self.conf = conf
        self.iou = iou
        self.device = device
        print(f"[YOLO-ONNX] Loaded {path.name} via Ultralytics/ORT")

    def detect(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(frame_bgr, **kwargs)
        detections: List[PoseDetection] = []
        if not results:
            return detections
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return detections
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        kpts = (
            r0.keypoints.data.cpu().numpy()
            if r0.keypoints is not None
            else np.zeros((len(xyxy), 17, 3), dtype=np.float32)
        )
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            kp = kpts[i]
            if kp.shape[-1] == 2:
                # add conf channel
                ones = np.ones((kp.shape[0], 1), dtype=np.float32)
                kp = np.concatenate([kp, ones], axis=1)
            detections.append(
                PoseDetection(
                    bbox=(x1, y1, x2, y2),
                    keypoints=kp.astype(np.float32),
                    conf=float(confs[i]),
                )
            )
        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections[:1]
