"""ONNX Runtime backend for 2-SHGNet ear landmarks (no .pth)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def preprocess_ear_bgr_numpy(ear_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """Match training preprocess → float32 NCHW BGR [0,1]."""
    if ear_bgr.size == 0:
        raise ValueError("Empty ear crop")
    img = cv2.resize(ear_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    img = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    arr = img.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def heatmaps_to_points_soft(
    hm: np.ndarray, input_size: int = 256, radius: int = 2
) -> np.ndarray:
    """Soft-argmax around peak → (55, 2) in input_size space."""
    single = hm.ndim == 3
    if single:
        hm = hm[np.newaxis, ...]
    b, n, h, w = hm.shape
    scale_x = input_size / float(w)
    scale_y = input_size / float(h)
    pts = np.zeros((b, n, 2), dtype=np.float32)
    for bi in range(b):
        for i in range(n):
            flat = hm[bi, i]
            idx = int(flat.argmax())
            yy, xx = divmod(idx, w)
            y0 = max(0, yy - radius)
            y1 = min(h - 1, yy + radius)
            x0 = max(0, xx - radius)
            x1 = min(w - 1, xx + radius)
            patch = flat[y0 : y1 + 1, x0 : x1 + 1]
            patch = patch - float(patch.max())
            wt = np.exp(patch)
            s = float(wt.sum())
            if s < 1e-12:
                pts[bi, i, 0] = xx * scale_x
                pts[bi, i, 1] = yy * scale_y
            else:
                ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
                pts[bi, i, 0] = float((wt * xs).sum() / s) * scale_x
                pts[bi, i, 1] = float((wt * ys).sum() / s) * scale_y
    return pts[0] if single else pts


class SHGNetOnnxEarLandmarker:
    """2-SHGNet via ONNX Runtime (CoreML / CUDA / CPU) — no PyTorch weights."""

    DISPLAY_NAME = "2-SHGNet-ONNX"

    def __init__(
        self,
        onnx_path: str,
        input_size: int = 256,
        providers: Optional[list[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for SHGNet ONNX.\n"
                "  pip install onnxruntime"
            ) from exc

        path = Path(onnx_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {path}\n"
                "Export first: python export_onnx.py --single-file"
            )

        avail = set(ort.get_available_providers())
        if providers is None:
            preferred = [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in preferred if p in avail] or [
                "CPUExecutionProvider"
            ]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=so, providers=providers
            )
        except Exception as first_err:
            if providers != ["CPUExecutionProvider"]:
                print(
                    f"[2-SHGNet-ONNX] {providers} failed ({first_err}); "
                    "falling back to CPU"
                )
                self.session = ort.InferenceSession(
                    str(path),
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
            else:
                raise

        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(
            f"[2-SHGNet-ONNX] Loaded {path.name} "
            f"providers={self.session.get_providers()}"
        )

    def predict(self, ear_bgr: np.ndarray) -> np.ndarray:
        pts, _ = self.predict_with_score(ear_bgr)
        return pts

    def predict_with_score(self, ear_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return ((55, 2) in 256-space, mean peak heatmap score)."""
        x = preprocess_ear_bgr_numpy(ear_bgr, self.input_size)
        outs = self.session.run([self.output_name], {self.input_name: x})
        hm = np.asarray(outs[0])
        if hm.ndim == 4:
            hm0 = hm[0]
        else:
            hm0 = hm
        pts = heatmaps_to_points_soft(hm0, self.input_size)
        if pts.shape != (55, 2):
            raise ValueError(f"Expected (55, 2) landmarks, got {pts.shape}")
        # Peak confidence
        flat = hm0.reshape(hm0.shape[0], -1)
        score = float(flat.max(axis=1).mean())
        return pts, score
