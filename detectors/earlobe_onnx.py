"""
Ear bbox detector from earlobe-tracker ONNX (YOLO26n-pose, 3 kpts).

Used only to localize the ear for 2-SHGNet (55 landmarks). Decode matches
earlobe-tracker reports/decode_recipe.md:
  output0 [1,300,15] = xyxy + conf + cls + lobe/tragus/helix (x,y,v)
  letterbox 640, pad gray 114
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


IMGSZ = 640
PAD_COLOR = 114


@dataclass
class EarBoxDetection:
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2 in frame pixels
    conf: float
    lobe: Tuple[float, float]
    tragus: Tuple[float, float]
    helix_top: Tuple[float, float]
    side: str


def letterbox_bgr(
    frame_bgr: np.ndarray, imgsz: int = IMGSZ, pad_color: int = PAD_COLOR
) -> tuple[np.ndarray, float, int, int]:
    h, w = frame_bgr.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x = (imgsz - nw) // 2
    pad_y = (imgsz - nh) // 2
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), pad_color, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def _unmap(mx: float, my: float, scale: float, pad_x: int, pad_y: int) -> Tuple[float, float]:
    return (mx - pad_x) / scale, (my - pad_y) / scale


class EarlobeOnnxEarFinder:
    """Find ear bbox (+ optional 3 kpts) via the shipped earlobe.onnx."""

    DISPLAY_NAME = "earlobe-ONNX"

    def __init__(
        self,
        onnx_path: str,
        conf: float = 0.35,
        max_detections: int = 2,
        providers: Optional[list[str]] = None,
    ) -> None:
        import onnxruntime as ort

        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing earlobe ONNX: {path}")

        pref = providers or ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        use = [p for p in pref if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=use)
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.max_detections = max_detections
        print(f"[earlobe] Loaded {path.name} via {self.session.get_providers()}")

    def detect(self, frame_bgr: np.ndarray) -> List[EarBoxDetection]:
        h, w = frame_bgr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox_bgr(frame_bgr)
        # RGB CHW float32 [0,1] — matches their browser / verify path
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]

        out = self.session.run(None, {self.input_name: chw})[0]
        # [1,300,15] or [1,15,300]
        arr = np.asarray(out)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape[0] == 15 and arr.shape[1] != 15:
            arr = arr.T  # [15,300] -> [300,15]
        if arr.shape[-1] < 15:
            raise RuntimeError(f"Unexpected earlobe output shape {arr.shape}")

        dets: List[EarBoxDetection] = []
        for row in arr:
            conf = float(row[4])
            if conf < self.conf:
                continue
            x1, y1 = _unmap(float(row[0]), float(row[1]), scale, pad_x, pad_y)
            x2, y2 = _unmap(float(row[2]), float(row[3]), scale, pad_x, pad_y)
            lobe = _unmap(float(row[6]), float(row[7]), scale, pad_x, pad_y)
            tragus = _unmap(float(row[9]), float(row[10]), scale, pad_x, pad_y)
            helix = _unmap(float(row[12]), float(row[13]), scale, pad_x, pad_y)
            # Clip bbox to frame
            x1 = float(np.clip(x1, 0, w - 1))
            y1 = float(np.clip(y1, 0, h - 1))
            x2 = float(np.clip(x2, 1, w))
            y2 = float(np.clip(y2, 1, h))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue
            # Helix is lateral: LEFT ear → helix.x > tragus.x (unmirrored frame)
            if abs(helix[0] - tragus[0]) > 2.0:
                side = "LEFT" if helix[0] > tragus[0] else "RIGHT"
            else:
                cx = 0.5 * (x1 + x2)
                side = "LEFT" if cx >= 0.5 * w else "RIGHT"
            dets.append(
                EarBoxDetection(
                    bbox=(x1, y1, x2, y2),
                    conf=conf,
                    lobe=lobe,
                    tragus=tragus,
                    helix_top=helix,
                    side=side,
                )
            )
            if len(dets) >= self.max_detections:
                break

        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets
