#!/usr/bin/env python3
"""
Live ear landmarks — full-ear crop, ~25–30 FPS (no earlobe.onnx):

  webcam
    → YOLO tip+side (sparse, downscaled)
    → tip-centered FULL-EAR square crop (large enough for helix+lobe)
    → 2-SHGNet (LEFT flip; retry other if weak)
    → One Euro
    → green box = crop square (covers entire ear) · 55 landmarks

Never shrink the crop from bad landmarks (that was pushing points off the ear).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from detectors.shgnet_onnx import SHGNetOnnxEarLandmarker
from detectors.yolo_pose import PoseDetection, YoloPoseDetector
from tracking.one_euro import OneEuroLandmarkFilter
from utils.coordinates import extract_square_crop
from utils.visualization import draw_hud, draw_landmarks, draw_roi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-ear YOLO + SHGNet live")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    # Crop side = pinna_h × pad. 1.55 ≈ AudioEar / CP2 full-ear framing.
    p.add_argument("--crop-pad", type=float, default=1.55)
    p.add_argument("--refine-pad", type=float, default=1.35)
    p.add_argument("--yolo-every", type=int, default=3)
    p.add_argument("--yolo-imgsz", type=int, default=320)
    return p.parse_args()


def pick_ear(
    pose: PoseDetection, min_conf: float = 0.25
) -> Optional[Tuple[str, Tuple[float, float]]]:
    lx, ly, lc = pose.left_ear
    rx, ry, rc = pose.right_ear
    if lc < 0.1 and rc < 0.1:
        return None
    if lc >= rc and lc >= min_conf * 0.4:
        return "LEFT", (float(lx), float(ly))
    if rc >= min_conf * 0.4:
        return "RIGHT", (float(rx), float(ry))
    return None


def pinna_height_px(
    pose: PoseDetection, tip: Tuple[float, float], frame_min: float
) -> float:
    """Estimate full pinna height (helix→lobe), slightly generous."""
    cands: list[float] = []
    eye = pose.eye_distance()
    if eye is not None and eye > frame_min * 0.02:
        cands.append(float(eye) * 1.15)  # ear ≈ IOD, slight margin
    nx, ny, nc = pose.nose
    if nc >= 0.2:
        d = float(np.hypot(tip[0] - nx, tip[1] - ny))
        if d > frame_min * 0.03:
            cands.append(d * 0.65)
    bh = pose.bbox[3] - pose.bbox[1]
    if bh > 1:
        cands.append(float(bh) * 0.18)
    if not cands:
        return frame_min * 0.14
    cands.sort()
    # Prefer larger of the two smallest → covers full ear
    if len(cands) == 1:
        h = cands[0]
    else:
        h = 0.35 * cands[0] + 0.65 * cands[1]
    return float(max(48.0, min(frame_min * 0.32, h)))


def medial_offset(
    tip: Tuple[float, float], side: str, pose: PoseDetection, fw: int
) -> Tuple[float, float]:
    nx, ny, nc = pose.nose
    if nc >= 0.2:
        vx, vy = nx - tip[0], ny - tip[1]
    else:
        vx, vy = 0.5 * fw - tip[0], 0.0
    n = (vx * vx + vy * vy) ** 0.5
    if n < 1e-3:
        return (-1.0 if side == "LEFT" else 1.0, 0.0)
    return vx / n, vy / n


def box_xyxy(
    cx: float, cy: float, side: float, fw: int, fh: int
) -> Tuple[int, int, int, int]:
    h = side * 0.5
    return (
        max(0, int(round(cx - h))),
        max(0, int(round(cy - h))),
        min(fw, int(round(cx + h))),
        min(fh, int(round(cy + h))),
    )


def map_pts(
    pts256: np.ndarray, ox: int, oy: int, side_px: int, model_input: int
) -> np.ndarray:
    s = side_px / float(model_input)
    out = np.empty((55, 2), np.float32)
    out[:, 0] = pts256[:, 0] * s + ox
    out[:, 1] = pts256[:, 1] * s + oy
    return out


def landmarks_ok(pts: np.ndarray, tip: Tuple[float, float], side_px: float) -> bool:
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = float(x1 - x0), float(y1 - y0)
    span = max(bw, bh)
    ratio = span / max(1.0, side_px)
    # Full-ear crop → landmarks fill ~50–80% of square
    if not (0.40 <= ratio <= 0.88):
        return False
    if min(bw, bh) < span * 0.28:
        return False
    mid = pts.mean(0)
    if float(np.hypot(mid[0] - tip[0], mid[1] - tip[1])) > side_px * 0.45:
        return False
    # Tip near landmark cloud (allow small margin)
    pad_x, pad_y = 0.08 * bw, 0.08 * bh
    if tip[0] < x0 - pad_x or tip[0] > x1 + pad_x:
        return False
    if tip[1] < y0 - pad_y or tip[1] > y1 + pad_y:
        return False
    return True


def hull_square(pts: np.ndarray, pad: float) -> Tuple[float, float, float]:
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    span = max(x1 - x0, y1 - y0, 20.0)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), span * pad


def resolve_onnx() -> str:
    p = Path(config.SHGNET_ONNX)
    if not p.is_file():
        alt = ROOT / "models" / "shgnet" / "hourglass_2stack_single.onnx"
        if alt.is_file():
            return str(alt)
    return str(p)


def infer_once(
    shg: SHGNetOnnxEarLandmarker,
    crop: np.ndarray,
    need_flip: bool,
    ox: int,
    oy: int,
    side_px: int,
    model_input: int,
) -> Tuple[np.ndarray, float]:
    infer = cv2.flip(crop, 1) if need_flip else crop
    pts256, score = shg.predict_with_score(infer)
    pts256 = np.asarray(pts256, dtype=np.float32).reshape(55, 2)
    if need_flip:
        pts256 = pts256.copy()
        pts256[:, 0] = (model_input - 1) - pts256[:, 0]
    return map_pts(pts256, ox, oy, side_px, model_input), float(score)


def main() -> int:
    args = parse_args()
    model_input = config.SHGNET_INPUT_SIZE

    if not Path(config.YOLO_WEIGHTS).is_file():
        print(f"Missing YOLO: {config.YOLO_WEIGHTS}", file=sys.stderr)
        return 1
    print(f"[YOLO] imgsz={args.yolo_imgsz} every={args.yolo_every}")
    yolo = YoloPoseDetector(
        config.YOLO_WEIGHTS, conf=config.YOLO_CONF, iou=config.YOLO_IOU
    )

    onnx_path = resolve_onnx()
    if not Path(onnx_path).is_file():
        print(f"Missing SHGNet ONNX: {onnx_path}", file=sys.stderr)
        return 1
    try:
        shg = SHGNetOnnxEarLandmarker(onnx_path, input_size=model_input)
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(e, file=sys.stderr)
        return 1

    filt = OneEuroLandmarkFilter(
        num_landmarks=55,
        min_cutoff=0.35,   # somewhat smooth (not as heavy as 0.22)
        beta=0.008,        # damp jitter; still follows head motion
        d_cutoff=0.9,
        rest_speed_px=16.0,
        rest_hold_frames=2,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed camera {args.camera}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print("Full-ear crop · green box = crop · SHGNet ONNX · One Euro. Press q.")

    fps = 0.0
    t_prev = time.perf_counter()
    frame_idx = 0
    lost = 0
    side: Optional[str] = None
    tip: Optional[Tuple[float, float]] = None
    pose: Optional[PoseDetection] = None
    geo: Optional[Tuple[float, float, float]] = None
    last_lm: Optional[np.ndarray] = None
    last_box: Optional[Tuple[int, int, int, int]] = None

    try:
        while True:
            t0 = time.perf_counter()
            dt = max(t0 - t_prev, 1e-6)
            t_prev = t0

            ok, frame = cap.read()
            if not ok:
                break
            fh, fw = frame.shape[:2]
            fmin = float(min(fw, fh))

            if frame_idx % max(1, args.yolo_every) == 0:
                scale = args.yolo_imgsz / float(max(fw, fh))
                nw, nh = max(32, int(fw * scale)), max(32, int(fh * scale))
                small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                dets = yolo.detect(small, imgsz=args.yolo_imgsz)
                if dets:
                    sx, sy = fw / float(nw), fh / float(nh)
                    d0 = dets[0]
                    kps = d0.keypoints.copy()
                    kps[:, 0] *= sx
                    kps[:, 1] *= sy
                    bb = d0.bbox
                    pose = PoseDetection(
                        bbox=(bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy),
                        keypoints=kps,
                        conf=d0.conf,
                    )
                    picked = pick_ear(pose, config.EAR_KEYPOINT_MIN_CONF)
                    if picked is not None:
                        new_side, new_tip = picked
                        if side is not None and new_side != side:
                            filt.reset()
                            last_lm = None
                            geo = None
                        side, tip = new_side, new_tip
                        pinna = pinna_height_px(pose, tip, fmin)
                        side_len = pinna * args.crop_pad
                        mx, my = medial_offset(tip, side, pose, fw)
                        # Center: tip + slight medial/down so full pinna fits
                        ncx = tip[0] + mx * (0.10 * pinna)
                        ncy = tip[1] + 0.06 * pinna
                        a = 0.60 if geo is not None else 1.0
                        if geo is None:
                            geo = (ncx, ncy, side_len)
                        else:
                            geo = (
                                (1 - a) * geo[0] + a * ncx,
                                (1 - a) * geo[1] + a * ncy,
                                (1 - a) * geo[2] + a * side_len,
                            )

            landmarks = None
            draw_box = last_box

            if tip is None or side is None or geo is None:
                lost += 1
                if lost > config.LOST_TRACK_FRAMES:
                    filt.reset()
                    side = tip = pose = geo = None
                    last_lm = last_box = None
                elif last_lm is not None:
                    landmarks = last_lm
            else:
                lost = 0
                cx, cy, side_len = geo
                # Tip must stay well inside the full-ear square
                half = side_len * 0.5
                if abs(tip[0] - cx) > half * 0.55 or abs(tip[1] - cy) > half * 0.55:
                    if pose is not None:
                        pinna = side_len / max(args.crop_pad, 1e-3)
                        mx, my = medial_offset(tip, side, pose, fw)
                        cx = tip[0] + mx * (0.10 * pinna)
                        cy = tip[1] + 0.06 * pinna
                    else:
                        cx, cy = tip[0], tip[1]
                    geo = (cx, cy, side_len)

                crop, ox, oy, side_px = extract_square_crop(frame, cx, cy, side_len)
                # Green box = full-ear crop (always covers the ear)
                draw_box = box_xyxy(cx, cy, float(side_px), fw, fh)
                last_box = draw_box

                if crop.size > 0 and min(crop.shape[:2]) >= 8:
                    prefer_flip = side == "LEFT"
                    # First lock: try both flips so points land on the ear immediately
                    first_lock = last_lm is None
                    pts, score = infer_once(
                        shg, crop, prefer_flip, ox, oy, side_px, model_input
                    )
                    if (
                        first_lock
                        or score < 0.12
                        or not landmarks_ok(pts, tip, float(side_px))
                    ):
                        pts2, sc2 = infer_once(
                            shg, crop, not prefer_flip, ox, oy, side_px, model_input
                        )
                        if sc2 > score or (
                            first_lock
                            and landmarks_ok(pts2, tip, float(side_px))
                            and not landmarks_ok(pts, tip, float(side_px))
                        ):
                            pts, score = pts2, sc2

                    # One-shot refine when points are valid but fill is weak
                    if landmarks_ok(pts, tip, float(side_px)) and score > 0.07:
                        fill = float(np.max(pts.max(0) - pts.min(0))) / max(
                            1.0, float(side_px)
                        )
                        if fill < 0.52 or fill > 0.82:
                            rcx, rcy, rside = hull_square(pts, args.refine_pad)
                            # Keep tip inside refine crop
                            if (
                                abs(tip[0] - rcx) < rside * 0.45
                                and abs(tip[1] - rcy) < rside * 0.45
                            ):
                                crop2, ox2, oy2, s2 = extract_square_crop(
                                    frame, rcx, rcy, rside
                                )
                                if crop2.size > 0 and min(crop2.shape[:2]) >= 8:
                                    p2, sc2 = infer_once(
                                        shg,
                                        crop2,
                                        prefer_flip,
                                        ox2,
                                        oy2,
                                        s2,
                                        model_input,
                                    )
                                    if landmarks_ok(p2, tip, float(s2)) and sc2 >= score * 0.90:
                                        pts, score = p2, sc2

                    if landmarks_ok(pts, tip, float(side_px)) and score > 0.07:
                        # Snap on first good hit so landmarks appear on-ear immediately
                        landmarks = filt.update(
                            pts,
                            dt=dt,
                            side=side,
                            max_step_px=12.0,
                            snap=first_lock,
                        )
                        last_lm = landmarks
                        # Expand crop only if landmarks stick outside (never shrink)
                        x0, y0 = landmarks.min(0)
                        x1, y1 = landmarks.max(0)
                        need = max(
                            cx - half - float(x0),
                            float(x1) - (cx + half),
                            cy - half - float(y0),
                            float(y1) - (cy + half),
                            0.0,
                        )
                        if need > 2.0:
                            side_len = side_len + 2.0 * need + side_len * 0.04
                            geo = (cx, cy, side_len)
                    elif last_lm is not None:
                        landmarks = last_lm

                cv2.circle(
                    frame,
                    (int(tip[0]), int(tip[1])),
                    4,
                    (0, 140, 255),
                    -1,
                    cv2.LINE_AA,
                )

            if draw_box is not None:
                draw_roi(frame, draw_box, color=config.ROI_COLOR, thickness=2)
            if landmarks is not None:
                draw_landmarks(
                    frame,
                    landmarks,
                    color=config.LANDMARK_COLOR,
                    radius=config.LANDMARK_RADIUS,
                    draw_indices=args.debug,
                )

            inst = 1.0 / max(time.perf_counter() - t0, 1e-6)
            fps = fps * 0.85 + inst * 0.15 if fps > 0 else inst
            draw_hud(
                frame,
                fps=fps,
                ear_side=side,
                num_landmarks=55 if landmarks is not None else 0,
                model_name="YOLO+SHGNet-ONNX",
                color=config.HUD_COLOR,
            )
            cv2.imshow(config.WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

