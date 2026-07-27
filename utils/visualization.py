"""Drawing helpers for ROI, landmarks, and HUD."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def draw_roi(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (80, 200, 120),
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def draw_landmarks(
    frame: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 220, 255),
    radius: int = 2,
    draw_indices: bool = False,
) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    for i, (x, y) in enumerate(pts):
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(frame, (ix, iy), radius, color, -1, lineType=cv2.LINE_AA)
        if draw_indices:
            cv2.putText(
                frame,
                str(i + 1),
                (ix + 3, iy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    ear_side: Optional[str],
    num_landmarks: int,
    model_name: str,
    color: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    lines: Sequence[str] = (
        f"FPS: {fps:.0f}",
        f"EAR: {ear_side or '—'}",
        f"LANDMARKS: {num_landmarks}",
        f"MODEL: {model_name}",
    )
    x, y = 12, 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 28
