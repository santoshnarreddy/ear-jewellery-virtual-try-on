"""
Build earlobe-style dets from YOLO for ear_square_geometry (no earlobe.onnx).

COCO ear ≠ helix top — it sits near mid-pinna / canal. Helix is above the tip,
lobe below, so ear_square_geometry centers on the ear like CP2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from detectors.yolo_pose import PoseDetection


@dataclass
class EarCropDet:
    bbox: Tuple[float, float, float, float]
    helix_top: Tuple[float, float]
    tragus: Tuple[float, float]
    lobe: Tuple[float, float]
    side: str
    tip: Tuple[float, float]
    from_landmarks: bool = False


def _ear_height(
    tip: Tuple[float, float],
    pose: PoseDetection,
    frame_min: float,
) -> float:
    cands: list[float] = []
    eye = pose.eye_distance()
    if eye is not None and eye > frame_min * 0.02:
        # Slightly under IOD → tighter pinna box
        cands.append(float(eye) * 0.95)
    nx, ny, nc = pose.nose
    if nc >= 0.2:
        d = ((tip[0] - nx) ** 2 + (tip[1] - ny) ** 2) ** 0.5
        if d > max(12.0, frame_min * 0.03):
            cands.append(d * 0.52)
    x1, y1, x2, y2 = pose.bbox
    bh = y2 - y1
    if bh > 1:
        cands.append(bh * 0.145)
    if not cands:
        return frame_min * 0.11
    cands.sort()
    if len(cands) == 1:
        return float(cands[0])
    # Prefer tighter estimate
    return float(0.60 * cands[0] + 0.40 * cands[1])


def _medial(
    tip: Tuple[float, float],
    side: str,
    pose: PoseDetection,
    frame_w: int,
) -> Tuple[float, float]:
    nx, ny, nc = pose.nose
    if nc >= 0.2:
        vx, vy = nx - tip[0], ny - tip[1]
    else:
        vx, vy = (0.5 * frame_w) - tip[0], 0.0
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-3:
        return (-1.0 if side == "LEFT" else 1.0, 0.0)
    return vx / norm, vy / norm


def pose_to_ear_crop_det(
    pose: PoseDetection,
    frame_w: int,
    frame_h: int,
    min_conf: float = 0.25,
) -> Optional[EarCropDet]:
    """
    YOLO mid-ear tip → helix (above) / tragus (medial) / lobe (below) / bbox.

    Matches earlobe.onnx roles so ear_square_geometry (pad≈1.45) frames AudioEar.
    """
    _, _, lc = pose.left_ear
    _, _, rc = pose.right_ear
    if lc < 0.1 and rc < 0.1:
        return None
    side = "LEFT" if lc >= rc else "RIGHT"
    if lc < min_conf and rc < min_conf:
        # still allow weak tip
        pass

    tip_x, tip_y, tip_c = pose.left_ear if side == "LEFT" else pose.right_ear
    if tip_c < 0.1:
        return None
    tip = (float(tip_x), float(tip_y))

    frame_min = float(min(frame_w, frame_h))
    h = max(30.0, min(frame_min * 0.22, _ear_height(tip, pose, frame_min)))
    mx, my = _medial(tip, side, pose, frame_w)
    lat = -mx

    # Compact mid-pinna layout → smaller ear_square after pad
    hx = tip[0] + lat * (0.05 * h)
    hy = tip[1] - 0.42 * h
    tx = tip[0] + mx * (0.24 * h)
    ty = tip[1] + my * (0.24 * h)
    lx = tip[0] + mx * (0.04 * h)
    ly = tip[1] + 0.46 * h
    ox = tip[0] + lat * (0.26 * h)
    oy = tip[1]

    helix = (float(hx), float(hy))
    tragus = (float(tx), float(ty))
    lobe = (float(lx), float(ly))

    xs = [hx, tx, lx, ox, tip[0]]
    ys = [hy, ty, ly, oy, tip[1]]
    pad = 0.06 * h
    bbox = (
        float(min(xs) - pad),
        float(min(ys) - pad),
        float(max(xs) + pad),
        float(max(ys) + pad),
    )
    return EarCropDet(
        bbox=bbox,
        helix_top=helix,
        tragus=tragus,
        lobe=lobe,
        side=side,
        tip=tip,
        from_landmarks=False,
    )


def tip_inside_box(
    tip: Tuple[float, float],
    cx: float,
    cy: float,
    side: float,
    margin: float = 0.08,
) -> bool:
    """True if tip is inside the square with a small inner margin."""
    half = side * 0.5 * (1.0 - margin)
    return (abs(tip[0] - cx) <= half) and (abs(tip[1] - cy) <= half)
