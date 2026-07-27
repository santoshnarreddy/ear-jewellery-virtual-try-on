/**
 * Ear ROI tracker — mirrors desktop `EarLocalizer` for correct pinna framing.
 */

const LEFT_EAR_CORE = [234, 127, 93, 132, 58, 172, 136];
const RIGHT_EAR_CORE = [454, 356, 323, 361, 288, 397, 365];

function outermostXY(lms, indices, midX, w, h) {
  let best = null;
  let bestDist = -1;
  for (const i of indices) {
    const x = lms[i].x * w;
    const y = lms[i].y * h;
    const d = Math.abs(x - midX);
    if (d > bestDist) {
      bestDist = d;
      best = [x, y];
    }
  }
  return best;
}

function emaBox(prev, next, a) {
  if (!prev) return next.slice();
  return [
    a * next[0] + (1 - a) * prev[0],
    a * next[1] + (1 - a) * prev[1],
    a * next[2] + (1 - a) * prev[2],
    a * next[3] + (1 - a) * prev[3],
  ];
}

function clipRoi(roi, w, h) {
  let [x1, y1, x2, y2] = roi;
  x1 = Math.max(0, Math.min(w - 2, x1));
  y1 = Math.max(0, Math.min(h - 2, y1));
  x2 = Math.max(x1 + 2, Math.min(w, x2));
  y2 = Math.max(y1 + 2, Math.min(h, y2));
  return [x1, y1, x2, y2];
}

export function yawFromMatrix(mat) {
  const data = mat?.data || mat;
  if (!data || data.length < 16) return 0;
  const r00 = data[0];
  const r10 = data[4];
  const r20 = data[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(r20, sy) * 180) / Math.PI;
}

export class EarRoiTracker {
  constructor({
    faceHeightRatio = 0.55,
    aspect = 0.78,
    pad = 1.2,
    ema = 0.35,
    earKeypointMinConf = 0.25,
    yawThresholdDeg = 12,
    outwardRatio = 0.08,
  } = {}) {
    this.faceHeightRatio = faceHeightRatio;
    this.aspect = aspect;
    this.pad = pad;
    this.ema = ema;
    this.earKeypointMinConf = earKeypointMinConf;
    this.yawThresholdDeg = yawThresholdDeg;
    this.outwardRatio = outwardRatio;
    this.side = null;
    this.smoothRoi = null;
    this.lastCenter = null;
    this.lastFaceH = null;
  }

  reset() {
    this.side = null;
    this.smoothRoi = null;
    this.lastCenter = null;
    this.lastFaceH = null;
  }

  _chooseSide(yolo, landmarks, w, h, yawDeg) {
    const minC = this.earKeypointMinConf;
    let leftC = 0;
    let rightC = 0;
    if (yolo) {
      leftC = yolo.left?.c || 0;
      rightC = yolo.right?.c || 0;
    }

    let yawPref = null;
    if (yawDeg >= this.yawThresholdDeg) yawPref = "RIGHT";
    else if (yawDeg <= -this.yawThresholdDeg) yawPref = "LEFT";

    if (leftC >= minC || rightC >= minC) {
      let side = leftC >= rightC && leftC >= minC ? "LEFT" : "RIGHT";
      if (yawPref && yawPref !== side) {
        const prefC = yawPref === "RIGHT" ? rightC : leftC;
        if (prefC >= minC * 0.5) side = yawPref;
      }
      return side;
    }

    if (yawPref) return yawPref;

    if (landmarks && landmarks.length >= 468) {
      const midX = 0.5 * (landmarks[234].x + landmarks[454].x) * w;
      const leftHint = outermostXY(landmarks, LEFT_EAR_CORE, midX, w, h);
      const rightHint = outermostXY(landmarks, RIGHT_EAR_CORE, midX, w, h);
      if (leftHint && rightHint) {
        return Math.abs(leftHint[0] - midX) >= Math.abs(rightHint[0] - midX)
          ? "LEFT"
          : "RIGHT";
      }
    }
    return this.side;
  }

  _earCenter(side, yolo, landmarks, w, h) {
    const minC = this.earKeypointMinConf;
    if (yolo) {
      const ear = side === "LEFT" ? yolo.left : yolo.right;
      if (ear && ear.c >= minC) return { x: ear.x, y: ear.y, source: "yolo" };
    }
    if (landmarks && landmarks.length >= 468) {
      const midX = 0.5 * (landmarks[234].x + landmarks[454].x) * w;
      const hint =
        side === "LEFT"
          ? outermostXY(landmarks, LEFT_EAR_CORE, midX, w, h)
          : outermostXY(landmarks, RIGHT_EAR_CORE, midX, w, h);
      if (hint) return { x: hint[0], y: hint[1], source: "mediapipe" };
    }
    if (this.lastCenter) {
      return { x: this.lastCenter[0], y: this.lastCenter[1], source: "tracked" };
    }
    return null;
  }

  _faceMetrics(yolo, landmarks, w, h) {
    let faceH = null;
    let faceW = null;
    let midX = null;
    if (landmarks && landmarks.length >= 468) {
      faceH = Math.abs(landmarks[152].y - landmarks[10].y) * h;
      faceW = Math.abs(landmarks[454].x - landmarks[234].x) * w;
      midX = 0.5 * (landmarks[234].x + landmarks[454].x) * w;
    }
    if ((faceH == null || faceH < 1) && yolo?.bbox) {
      const [, y1, , y2] = yolo.bbox;
      faceH = (y2 - y1) * 0.45;
    }
    if ((faceH == null || faceH < 1) && this.lastFaceH) faceH = this.lastFaceH;
    if (faceH == null || faceH < 1) faceH = 160;
    this.lastFaceH = faceH;
    return { faceH, faceW, midX };
  }

  update({ yolo = null, landmarks = null, w, h, yawDeg = 0 } = {}) {
    const side = this._chooseSide(yolo, landmarks, w, h, yawDeg);
    if (!side) return null;

    if (this.side && side !== this.side) {
      this.smoothRoi = null;
      this.lastCenter = null;
    }
    this.side = side;

    const center = this._earCenter(side, yolo, landmarks, w, h);
    if (!center) return null;

    const { faceH, faceW, midX } = this._faceMetrics(yolo, landmarks, w, h);
    let cx = center.x;
    let cy = center.y;

    // Nudge away from face midline so ROI covers full pinna (desktop 0.08)
    if (midX != null && faceW != null) {
      const outward = this.outwardRatio * Math.max(faceW, 1);
      cx = cx + (cx >= midX ? outward : -outward);
    }

    const boxH = Math.max(48, faceH * this.faceHeightRatio * this.pad);
    const boxW = boxH * this.aspect;
    let raw = [cx - boxW / 2, cy - boxH / 2, cx + boxW / 2, cy + boxH / 2];
    raw = clipRoi(raw, w, h);
    this.smoothRoi = emaBox(this.smoothRoi, raw, this.ema);
    const roi = clipRoi(this.smoothRoi, w, h).map((v) => Math.round(v));
    this.lastCenter = [cx, cy];

    return {
      side,
      roi,
      center: [cx, cy],
      faceH,
      source: center.source,
    };
  }
}
