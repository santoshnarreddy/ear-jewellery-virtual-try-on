"""One Euro Filter for low-latency temporal smoothing of 2D points."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class OneEuroFilter1D:
    """1D One Euro Filter (Casiez et al.)."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x: float, dt: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        dx = (x - self._x_prev) / max(dt, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev  # type: ignore[operator]

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class OneEuroLandmarkFilter:
    """Independent One Euro filters for all 55 (x, y) landmarks.

    When median landmark speed stays below ``rest_speed_px``, output is held
    (zero rest jitter). Above that, One Euro runs with low cutoffs so motion
    stays smooth without high-frequency noise.
    """

    def __init__(
        self,
        num_landmarks: int = 55,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        rest_speed_px: float = 12.0,
        rest_hold_frames: int = 2,
    ) -> None:
        self.num_landmarks = num_landmarks
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.rest_speed_px = rest_speed_px
        self.rest_hold_frames = max(1, int(rest_hold_frames))
        self._fx = [OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(num_landmarks)]
        self._fy = [OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(num_landmarks)]
        self._side: Optional[str] = None
        self._last_out: Optional[np.ndarray] = None
        self._rest_frames = 0

    def reset(self) -> None:
        for f in self._fx:
            f.reset()
        for f in self._fy:
            f.reset()
        self._side = None
        self._last_out = None
        self._rest_frames = 0

    def update(
        self,
        points: np.ndarray,
        dt: float,
        side: Optional[str] = None,
        max_step_px: float = 12.0,
        snap: bool = False,
    ) -> np.ndarray:
        if side is not None and self._side is not None and side != self._side:
            self.reset()
        if side is not None:
            self._side = side

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        n = min(len(pts), self.num_landmarks)

        # Instant lock: seed filters to the new pose (no lag / clamp)
        if snap or self._last_out is None:
            for i in range(n):
                self._fx[i]._x_prev = float(pts[i, 0])
                self._fy[i]._x_prev = float(pts[i, 1])
                self._fx[i]._dx_prev = 0.0
                self._fy[i]._dx_prev = 0.0
            self._last_out = pts[:n].copy()
            self._rest_frames = 0
            return pts[:n].copy() if n == pts.shape[0] else pts.copy()

        # Clamp raw outliers vs last output (kills single-frame spikes)
        if max_step_px > 0:
            delta = pts[:n] - self._last_out[:n]
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_step_px / np.maximum(dist, 1e-6))
            pts[:n] = self._last_out[:n] + delta * scale

        # Rest freeze on raw motion before One Euro (steadier hold)
        if self.rest_speed_px > 0:
            raw_disp = np.linalg.norm(pts[:n] - self._last_out[:n], axis=1)
            raw_speed = float(np.median(raw_disp) / max(dt, 1e-6))
            if raw_speed < self.rest_speed_px:
                self._rest_frames += 1
                if self._rest_frames >= self.rest_hold_frames:
                    for i in range(n):
                        self._fx[i]._x_prev = float(self._last_out[i, 0])
                        self._fy[i]._x_prev = float(self._last_out[i, 1])
                        self._fx[i]._dx_prev = 0.0
                        self._fy[i]._dx_prev = 0.0
                    return self._last_out.copy()
            else:
                self._rest_frames = 0

        out = np.empty_like(pts)
        for i in range(n):
            out[i, 0] = self._fx[i](float(pts[i, 0]), dt)
            out[i, 1] = self._fy[i](float(pts[i, 1]), dt)

        self._last_out = out.copy()
        return out


class OneEuroBoxFilter:
    """One Euro on ear box center (cx, cy) and side length."""

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ) -> None:
        self._cx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self._cy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self._side = OneEuroFilter1D(min_cutoff, beta * 0.5, d_cutoff)

    def reset(self) -> None:
        self._cx.reset()
        self._cy.reset()
        self._side.reset()

    def update(
        self, cx: float, cy: float, side: float, dt: float
    ) -> tuple[float, float, float]:
        return (
            self._cx(float(cx), dt),
            self._cy(float(cy), dt),
            max(16.0, self._side(float(side), dt)),
        )
