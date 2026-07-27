"""Coordinate helpers for ROI crops and landmark remapping (CP2 crop math)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def clip_roi(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    """Clip float ROI to integer bounds inside the frame."""
    ix1 = int(max(0, min(width - 1, round(x1))))
    iy1 = int(max(0, min(height - 1, round(y1))))
    ix2 = int(max(0, min(width, round(x2))))
    iy2 = int(max(0, min(height, round(y2))))
    if ix2 <= ix1:
        ix2 = min(width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(height, iy1 + 1)
    return ix1, iy1, ix2, iy2


def roi_from_center(
    cx: float,
    cy: float,
    box_w: float,
    box_h: float,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    """Build axis-aligned ROI from center + size, clipped to frame."""
    x1 = cx - box_w * 0.5
    y1 = cy - box_h * 0.5
    x2 = cx + box_w * 0.5
    y2 = cy + box_h * 0.5
    return clip_roi(x1, y1, x2, y2, frame_w, frame_h)


def crop_roi(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Return a view/copy of the ROI region."""
    x1, y1, x2, y2 = roi
    return frame_bgr[y1:y2, x1:x2].copy()


def ear_square_geometry(
    bbox: Tuple[float, float, float, float],
    helix: Tuple[float, float],
    tragus: Tuple[float, float],
    lobe: Tuple[float, float],
    pad: float = 1.45,
) -> Tuple[float, float, float]:
    """
    Square ear crop center + side — identical to CP2 / earlobe.onnx crop path.

    Centers on helix/tragus/lobe (blended with bbox) so SHGNet sees the pinna
    at AudioEar framing.
    """
    bx1, by1, bx2, by2 = bbox
    hx, hy = helix
    tx, ty = tragus
    lx, ly = lobe
    cx = 0.7 * ((hx + tx + lx) / 3.0) + 0.3 * (0.5 * (bx1 + bx2))
    cy = 0.7 * ((hy + ty + ly) / 3.0) + 0.3 * (0.5 * (by1 + by2))
    ear_h = max(abs(ly - hy), abs(by2 - by1) * 0.9, 24.0)
    ear_w = max(abs(tx - hx) * 1.35, abs(bx2 - bx1) * 0.9, 16.0)
    side = max(ear_h, ear_w, 32.0) * float(pad)
    return float(cx), float(cy), float(side)


def landmarks_to_earlobe_kpts(
    pts: np.ndarray,
    side: str,
) -> Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float, float, float],
]:
    """Helix / tragus / lobe + bbox from 55 landmarks (same roles as earlobe.onnx)."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    x0, y0 = float(p[:, 0].min()), float(p[:, 1].min())
    x1, y1 = float(p[:, 0].max()), float(p[:, 1].max())
    helix = (float(p[int(p[:, 1].argmin()), 0]), y0)
    lobe = (float(p[int(p[:, 1].argmax()), 0]), y1)

    y_lo = y0 + 0.25 * (y1 - y0)
    y_hi = y0 + 0.70 * (y1 - y0)
    mid = p[(p[:, 1] >= y_lo) & (p[:, 1] <= y_hi)]
    if mid.size < 2:
        mid = p
    if side == "LEFT":
        t = mid[int(mid[:, 0].argmin())]
    else:
        t = mid[int(mid[:, 0].argmax())]
    tragus = (float(t[0]), float(t[1]))

    pad = 0.06 * max(x1 - x0, y1 - y0, 1.0)
    bbox = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    return helix, tragus, lobe, bbox


def extract_square_crop(
    frame_bgr: np.ndarray,
    cx: float,
    cy: float,
    side: float,
    pad_color: int = 114,
) -> tuple[np.ndarray, int, int, int]:
    """
    Always-square ear crop with gray pad when the box leaves the frame.

    Returns (crop_bgr, origin_x, origin_y, side_px). Landmarks in 256-space map as:
      frame_x = origin_x + x_256 * side_px / 256
    """
    h, w = frame_bgr.shape[:2]
    side_i = max(32, int(round(side)))
    ox = int(round(cx - side_i * 0.5))
    oy = int(round(cy - side_i * 0.5))
    canvas = np.full((side_i, side_i, 3), int(pad_color), dtype=np.uint8)
    sx1, sy1 = max(0, ox), max(0, oy)
    sx2, sy2 = min(w, ox + side_i), min(h, oy + side_i)
    dx, dy = sx1 - ox, sy1 - oy
    if sx2 > sx1 and sy2 > sy1:
        canvas[dy : dy + (sy2 - sy1), dx : dx + (sx2 - sx1)] = frame_bgr[
            sy1:sy2, sx1:sx2
        ]
    return canvas, ox, oy, side_i


def map_points_from_crop(
    points: np.ndarray,
    roi: Tuple[int, int, int, int],
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    """Map landmark coordinates from crop pixel space into full-frame coordinates."""
    x1, y1, x2, y2 = roi
    out_w = max(1, x2 - x1)
    out_h = max(1, y2 - y1)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if crop_w <= 0 or crop_h <= 0:
        return pts
    pts[:, 0] = pts[:, 0] * (out_w / float(crop_w)) + x1
    pts[:, 1] = pts[:, 1] * (out_h / float(crop_h)) + y1
    return pts


def ema_box(
    prev: Tuple[float, float, float, float] | None,
    new: Tuple[float, float, float, float],
    alpha: float,
) -> Tuple[float, float, float, float]:
    """Exponential moving average on (x1, y1, x2, y2)."""
    if prev is None:
        return tuple(float(v) for v in new)  # type: ignore[return-value]
    a = float(alpha)
    return tuple((1.0 - a) * p + a * n for p, n in zip(prev, new))  # type: ignore[return-value]


def ema_points(prev: np.ndarray | None, new: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average on (N, 2) landmarks."""
    new_arr = np.asarray(new, dtype=np.float32)
    if prev is None:
        return new_arr.copy()
    a = float(alpha)
    return (1.0 - a) * np.asarray(prev, dtype=np.float32) + a * new_arr
