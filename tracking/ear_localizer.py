"""
YOLO tip + side → tip-anchored AudioEar square crop (earlobe crop proportions).

No earlobe.onnx. Box is locked to the YOLO ear tip (trusted). Landmark
feedback may refine size/center only if the tip still sits near the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from detectors.yolo_pose import PoseDetection
from utils.coordinates import (
    ear_square_geometry,
    landmarks_to_earlobe_kpts,
    roi_from_center,
    tip_anchored_ear_square,
)


EarSide = str


@dataclass
class EarLocalization:
    side: EarSide
    roi: Tuple[int, int, int, int]
    center: Tuple[float, float]
    yolo_ear: Tuple[float, float]
    face_height: float  # AudioEar square side (px)
    helix_top: Tuple[float, float]
    tragus: Tuple[float, float]
    lobe: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    source: str


def _medial_dir(
    tip: Tuple[float, float],
    side: EarSide,
    pose: Optional[PoseDetection],
    frame_w: int,
) -> Tuple[float, float]:
    if pose is not None and pose.nose[2] >= 0.2:
        nx, ny, _ = pose.nose
        vx, vy = nx - tip[0], ny - tip[1]
    else:
        vx, vy = (0.5 * frame_w) - tip[0], 0.0
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-3:
        return (-1.0 if side == "LEFT" else 1.0, 0.0)
    return vx / norm, vy / norm


def yolo_to_earlobe_kpts(
    tip: Tuple[float, float],
    side: EarSide,
    pose: Optional[PoseDetection],
    frame_w: int,
    frame_h: int,
    ear_scale: float = 0.55,
    ear_h: Optional[float] = None,
) -> Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float, float, float],
]:
    """Synthetic helix / tragus / lobe for debug + ear_square_geometry fallback."""
    frame_min = float(min(frame_w, frame_h))
    if ear_h is None:
        from utils.coordinates import estimate_ear_height_px

        h = estimate_ear_height_px(tip, pose, frame_min, ear_scale)
    else:
        h = float(ear_h)
    h = max(28.0, min(frame_min * 0.24, h))

    mx, my = _medial_dir(tip, side, pose, frame_w)

    # Helix = YOLO tip (trusted); small margin above for rim
    hx, hy = float(tip[0]), float(tip[1] - 0.05 * h)
    tx = hx + mx * (0.36 * h)
    ty = hy + my * (0.36 * h) + (0.40 * h)
    lx = hx + mx * (0.08 * h)
    ly = hy + my * (0.08 * h) + (1.02 * h)

    helix = (hx, hy)
    tragus = (float(tx), float(ty))
    lobe = (float(lx), float(ly))

    lat = -mx
    xs = [hx, tx, lx, hx + lat * (0.28 * h)]
    ys = [hy, ty, ly, hy + 0.50 * h]
    pad = 0.08 * h
    bbox = (
        float(min(xs) - pad),
        float(min(ys) - pad),
        float(max(xs) + pad),
        float(max(ys) + pad),
    )
    return helix, tragus, lobe, bbox


def _tip_in_box_top(
    tip: Tuple[float, float],
    cx: float,
    cy: float,
    side: float,
) -> bool:
    """YOLO tip must sit in the upper band of the square (AudioEar layout)."""
    half = side * 0.5
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    if not (x0 <= tip[0] <= x1 and y0 <= tip[1] <= y1):
        return False
    # Tip in top ~40% of square
    return tip[1] <= y0 + 0.40 * side


class EarLocalizer:
    """
    YOLO tip is the anchor. Crop square is tip-anchored (earlobe/AudioEar pad).
    Landmark refine accepted only if tip still lies near the top of the box.
    """

    def __init__(
        self,
        ear_scale: float = 0.55,
        ear_keypoint_min_conf: float = 0.25,
        ema_roi: float = 0.50,
        crop_pad: float = 1.45,
    ) -> None:
        self.ear_scale = ear_scale
        self.ear_keypoint_min_conf = ear_keypoint_min_conf
        self.ema_roi = ema_roi
        self.crop_pad = crop_pad
        self._side: Optional[EarSide] = None
        self._smooth: Optional[Tuple[float, float, float]] = None
        self._last_tip: Optional[Tuple[float, float]] = None
        self._lm_geo: Optional[Tuple[float, float, float]] = None
        self._last_kpts: Optional[
            Tuple[
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float, float, float],
            ]
        ] = None

    def reset(self) -> None:
        self._side = None
        self._smooth = None
        self._last_tip = None
        self._lm_geo = None
        self._last_kpts = None

    def _choose_side(self, pose: Optional[PoseDetection]) -> Optional[EarSide]:
        if pose is None:
            return self._side
        _, _, lc = pose.left_ear
        _, _, rc = pose.right_ear
        min_c = self.ear_keypoint_min_conf
        if lc >= min_c or rc >= min_c:
            return "LEFT" if lc >= rc and lc >= min_c else "RIGHT"
        nx, _, nc = pose.nose
        if nc >= 0.2 and (lc >= 0.08 or rc >= 0.08):
            lx, _, _ = pose.left_ear
            rx, _, _ = pose.right_ear
            return "LEFT" if abs(lx - nx) >= abs(rx - nx) else "RIGHT"
        return self._side

    def _tip(
        self, side: EarSide, pose: Optional[PoseDetection]
    ) -> Optional[Tuple[float, float]]:
        if pose is not None:
            x, y, c = pose.left_ear if side == "LEFT" else pose.right_ear
            if c >= 0.1:
                return (float(x), float(y))
        return self._last_tip

    def update_from_landmarks(self, pts: np.ndarray, side: EarSide) -> None:
        """
        Propose landmark-based crop. Applied next frame only if tip still
        sits in the top of that square (YOLO tip stays authoritative).
        """
        if self._last_tip is None:
            return
        helix, tragus, lobe, bbox = landmarks_to_earlobe_kpts(pts, side)
        cx, cy, side_len = ear_square_geometry(
            bbox, helix, tragus, lobe, pad=self.crop_pad
        )
        if not _tip_in_box_top(self._last_tip, cx, cy, side_len):
            # Bad landmarks — keep tip-anchored box
            self._lm_geo = None
            return
        self._last_kpts = (helix, tragus, lobe, bbox)
        self._lm_geo = (cx, cy, side_len)
        self._side = side

    def update(
        self,
        frame_shape: Tuple[int, int],
        pose: Optional[PoseDetection],
        face=None,
    ) -> Optional[EarLocalization]:
        del face
        h, w = frame_shape[:2]
        side = self._choose_side(pose)
        if side is None:
            return None
        if self._side is not None and side != self._side:
            self._smooth = None
            self._last_tip = None
            self._lm_geo = None
            self._last_kpts = None
        self._side = side

        tip = self._tip(side, pose)
        if tip is None:
            return None
        self._last_tip = tip

        # Primary: tip-anchored square (YOLO tip correct → box follows tip)
        cx, cy, side_len, ear_h = tip_anchored_ear_square(
            tip,
            side,
            pose,
            w,
            h,
            pad=self.crop_pad,
            ear_scale=self.ear_scale,
        )
        source = "yolo"

        # Optional landmark refine — only if tip still in top band
        if self._lm_geo is not None:
            lcx, lcy, ls = self._lm_geo
            if _tip_in_box_top(tip, lcx, lcy, ls):
                # Blend: trust landmarks for size, keep tip-aligned center bias
                cx = 0.55 * lcx + 0.45 * cx
                cy = 0.55 * lcy + 0.45 * cy
                side_len = 0.55 * ls + 0.45 * side_len
                # Re-snap so tip stays in top band after blend
                half = side_len * 0.5
                top = cy - half
                if tip[1] < top + 0.08 * side_len or tip[1] > top + 0.38 * side_len:
                    cy = tip[1] + 0.28 * side_len
                source = "landmarks"
            else:
                self._lm_geo = None

        helix, tragus, lobe, bbox = yolo_to_earlobe_kpts(
            tip, side, pose, w, h, ear_scale=self.ear_scale, ear_h=ear_h
        )
        if self._last_kpts is not None and source == "landmarks":
            helix, tragus, lobe, bbox = self._last_kpts
        else:
            self._last_kpts = (helix, tragus, lobe, bbox)

        if self._smooth is None:
            self._smooth = (cx, cy, side_len)
        else:
            a = self.ema_roi
            pcx, pcy, ps = self._smooth
            # Tip moved a lot → snap faster so box stays on ear
            tip_jump = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if tip_jump > side_len * 0.20:
                a = min(0.85, a + 0.30)
            self._smooth = (
                a * cx + (1 - a) * pcx,
                a * cy + (1 - a) * pcy,
                a * side_len + (1 - a) * ps,
            )
        scx, scy, ss = self._smooth

        # Hard constraint: smoothed box must keep tip in upper band
        if not _tip_in_box_top(tip, scx, scy, ss):
            scy = tip[1] + 0.28 * ss
            mx, _ = _medial_dir(tip, side, pose, w)
            scx = tip[0] + mx * (0.12 * (ss / max(self.crop_pad, 1.0)))

        ss = min(ss, float(w), float(h))
        scx = max(ss * 0.5, min(w - ss * 0.5, scx))
        scy = max(ss * 0.5, min(h - ss * 0.5, scy))
        self._smooth = (scx, scy, ss)

        raw = roi_from_center(scx, scy, ss, ss, w, h)
        roi = (
            max(0, min(w - 1, int(round(raw[0])))),
            max(0, min(h - 1, int(round(raw[1])))),
            max(1, min(w, int(round(raw[2])))),
            max(1, min(h, int(round(raw[3])))),
        )
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            return None

        return EarLocalization(
            side=side,
            roi=roi,
            center=(scx, scy),
            yolo_ear=tip,
            face_height=float(ss),
            helix_top=helix,
            tragus=tragus,
            lobe=lobe,
            bbox=bbox,
            source=source,
        )
