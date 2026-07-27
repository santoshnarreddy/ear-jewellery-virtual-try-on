"""Temporal smoothing for ROI and landmarks."""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.coordinates import ema_points


class LandmarkSmoother:
    """EMA on 55 landmarks; resets when ear side changes or track is lost."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None
        self._side: Optional[str] = None

    def reset(self) -> None:
        self._prev = None
        self._side = None

    def update(self, points: np.ndarray, side: Optional[str] = None) -> np.ndarray:
        if side is not None and self._side is not None and side != self._side:
            self._prev = None
        if side is not None:
            self._side = side
        smoothed = ema_points(self._prev, points, self.alpha)
        self._prev = smoothed
        return smoothed
