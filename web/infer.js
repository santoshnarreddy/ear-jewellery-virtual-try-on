/**
 * Browser live ear landmarks — ONNX only (no .pth, no MediaPipe):
 *
 *   webcam → YOLO ONNX tip → tip-centered full-ear crop
 *         → 2-SHGNet ONNX (flip LEFT) → One Euro → overlay
 */
import * as ort from "/vendor/onnxruntime-web/dist/ort.wasm.min.mjs";
import { OneEuroLandmarks } from "./one_euro.js";
import { YoloPoseBrowser } from "./yolo_pose.js";
import { canvasRgbaToBgrChw, heatmapsToPointsSoft } from "./preprocess.js";

const SHGNET_URL = "/models/shgnet/hourglass_2stack.onnx";
const YOLO_URL = "/models/yolo/yolo26n-pose.onnx";
const WASM_PATH = "/vendor/onnxruntime-web/dist/";

const CROP_PAD = 1.55;
const YOLO_EVERY = 3;
const YOLO_IMGSZ = 640; // must match yolo26n-pose.onnx input (fixed 640)

const statusEl = document.getElementById("status");
const sizesEl = document.getElementById("sizes");
const loadBtn = document.getElementById("loadModel");
const startCamBtn = document.getElementById("startCam");
const stopCamBtn = document.getElementById("stopCam");
const video = document.getElementById("video");
const canvas = document.getElementById("out");
const ctx = canvas.getContext("2d", { alpha: false });

let shgSession = null;
let yoloPose = null;
let stream = null;
let live = false;
let inferBusy = false;
let rafId = 0;
let lastTs = 0;
let fpsEma = 0;
let inferMsEma = 0;
let frameIdx = 0;
let lastYolo = null;
let side = null;
let tip = null;
let geo = null; // {cx, cy, side}
let rawPts = null; // unmirrored frame space
let lastBox = null; // unmirrored xyxy
let firstLock = true;

const smoother = new OneEuroLandmarks(55, 0.35, 0.008, 0.9, 16, 2);

const cropCanvas = document.createElement("canvas");
cropCanvas.width = 256;
cropCanvas.height = 256;
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

const padCanvas = document.createElement("canvas");
const padCtx = padCanvas.getContext("2d", { willReadFrequently: true });

function setStatus(msg) {
  statusEl.textContent = msg;
}

function updateButtons() {
  startCamBtn.disabled = !(shgSession && yoloPose && !live);
  stopCamBtn.disabled = !live;
  loadBtn.disabled = live;
}

function reportSizes() {
  sizesEl.innerHTML = `
    <strong>Browser assets (no .pth)</strong>
    <table>
      <thead><tr><th>Asset</th><th>Size</th><th>Role</th></tr></thead>
      <tbody>
        <tr><td>hourglass_2stack.onnx</td><td>~26 MB</td><td>55 ear landmarks</td></tr>
        <tr><td>yolo26n-pose.onnx</td><td>~12 MB</td><td>ear tip + side</td></tr>
        <tr><td>onnxruntime-web WASM</td><td>~5–15 MB</td><td>1st visit, then cached</td></tr>
        <tr><td><strong>Total first visit</strong></td><td><strong>~43–53 MB</strong></td><td>vs ~83 MB .pth on desktop</td></tr>
      </tbody>
    </table>
  `;
}

let ortReady = null;
async function ensureOrt() {
  if (ortReady) return ortReady;
  ortReady = (async () => {
    ort.env.wasm.wasmPaths = WASM_PATH;
    const canSAB =
      typeof SharedArrayBuffer !== "undefined" &&
      (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
    ort.env.wasm.numThreads = canSAB
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;
    ort.env.wasm.proxy = false;
  })();
  return ortReady;
}

async function createSession(url, label) {
  await ensureOrt();
  setStatus(`Loading ${label}…`);
  return ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

async function loadModels() {
  loadBtn.disabled = true;
  setStatus("Loading ONNX models (YOLO + SHGNet)…");
  try {
    const shg = await createSession(SHGNET_URL, "SHGNet ONNX (~26 MB)");
    const yolo = await createSession(YOLO_URL, "YOLO pose ONNX (~12 MB)");
    shgSession = shg;
    yoloPose = new YoloPoseBrowser(
      yolo,
      (data, dims) => new ort.Tensor("float32", data, dims),
      YOLO_IMGSZ,
      0.28
    );
    setStatus(
      "Ready · YOLO + SHGNet ONNX (no .pth)\n" +
        "Click Start live cam — allow camera when prompted."
    );
    updateButtons();
    reportSizes();
  } catch (e) {
    console.error(e);
    setStatus(`Load failed: ${e?.message || e}\nRetry Load models in Chrome/Edge.`);
    loadBtn.disabled = false;
    updateButtons();
  }
}

function pinnaHeight(yolo, vw, vh) {
  const fmin = Math.min(vw, vh);
  const tip = yolo.tip;
  const cands = [];
  if (yolo.eyeDist && yolo.eyeDist > fmin * 0.02) cands.push(yolo.eyeDist * 1.15);
  if (yolo.nose?.c >= 0.2) {
    const d = Math.hypot(tip.x - yolo.nose.x, tip.y - yolo.nose.y);
    if (d > fmin * 0.03) cands.push(d * 0.65);
  }
  const [, y1, , y2] = yolo.bbox;
  const bh = y2 - y1;
  if (bh > 1) cands.push(bh * 0.18);
  if (!cands.length) return fmin * 0.14;
  cands.sort((a, b) => a - b);
  const h =
    cands.length === 1 ? cands[0] : 0.35 * cands[0] + 0.65 * cands[1];
  return Math.max(48, Math.min(fmin * 0.32, h));
}

function medial(yolo, tip, side, vw) {
  if (yolo.nose?.c >= 0.2) {
    const vx = yolo.nose.x - tip.x;
    const vy = yolo.nose.y - tip.y;
    const n = Math.hypot(vx, vy);
    if (n > 1e-3) return [vx / n, vy / n];
  }
  return [side === "LEFT" ? -1 : 1, 0];
}

function updateGeoFromYolo(yolo, vw, vh) {
  const tipPt = yolo.tip;
  const pinna = pinnaHeight(yolo, vw, vh);
  const sideLen = pinna * CROP_PAD;
  const [mx] = medial(yolo, tipPt, yolo.side, vw);
  const ncx = tipPt.x + mx * (0.1 * pinna);
  const ncy = tipPt.y + 0.06 * pinna;
  if (!geo) {
    geo = { cx: ncx, cy: ncy, side: sideLen };
  } else {
    const a = 0.6;
    geo = {
      cx: (1 - a) * geo.cx + a * ncx,
      cy: (1 - a) * geo.cy + a * ncy,
      side: (1 - a) * geo.side + a * sideLen,
    };
  }
  side = yolo.side;
  tip = tipPt;
}

/** Square crop with gray pad (matches Python extract_square_crop). */
function drawSquareCrop(cx, cy, sidePx, needFlip) {
  const s = Math.max(32, Math.round(sidePx));
  const ox = Math.round(cx - s * 0.5);
  const oy = Math.round(cy - s * 0.5);
  padCanvas.width = s;
  padCanvas.height = s;
  padCtx.fillStyle = "rgb(114,114,114)";
  padCtx.fillRect(0, 0, s, s);
  padCtx.drawImage(video, ox, oy, s, s, 0, 0, s, s);

  cropCtx.save();
  if (needFlip) {
    cropCtx.translate(256, 0);
    cropCtx.scale(-1, 1);
  }
  cropCtx.drawImage(padCanvas, 0, 0, s, s, 0, 0, 256, 256);
  cropCtx.restore();
  return { ox, oy, sidePx: s };
}

function cropToTensor() {
  const img = cropCtx.getImageData(0, 0, 256, 256);
  return new ort.Tensor("float32", canvasRgbaToBgrChw(img), [1, 3, 256, 256]);
}

function landmarksOk(pts, tipPt, sidePx) {
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  const bw = x1 - x0;
  const bh = y1 - y0;
  const span = Math.max(bw, bh);
  const ratio = span / Math.max(1, sidePx);
  if (ratio < 0.4 || ratio > 0.88) return false;
  if (Math.min(bw, bh) < span * 0.28) return false;
  let mx = 0,
    my = 0;
  for (const [x, y] of pts) {
    mx += x;
    my += y;
  }
  mx /= pts.length;
  my /= pts.length;
  if (Math.hypot(mx - tipPt.x, my - tipPt.y) > sidePx * 0.45) return false;
  return true;
}

function mirrorX(x, w) {
  return w - 1 - x;
}

function paintVideoMirrored() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw;
    canvas.height = vh;
  }
  ctx.save();
  ctx.translate(vw, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, vw, vh);
  ctx.restore();
  return { vw, vh };
}

function drawHud(smoothedDisp, boxDisp, tipDisp, info) {
  if (boxDisp) {
    const [x1, y1, x2, y2] = boxDisp;
    ctx.strokeStyle = "rgba(80, 200, 120, 0.95)";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (tipDisp) {
    ctx.fillStyle = "rgb(0,140,255)";
    ctx.beginPath();
    ctx.arc(tipDisp[0], tipDisp[1], 4, 0, Math.PI * 2);
    ctx.fill();
  }
  if (smoothedDisp) {
    ctx.fillStyle = "rgb(0,220,255)";
    for (const [x, y] of smoothedDisp) {
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (info) {
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(8, 8, 280, 40);
    ctx.fillStyle = "#f0f0f0";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(info, 16, 26);
    ctx.fillText("YOLO+SHGNet ONNX", 16, 42);
  }
}

async function runShg(needFlip, ox, oy, sidePx) {
  const t0 = performance.now();
  const out = await shgSession.run({
    [shgSession.inputNames[0]]: cropToTensor(),
  });
  const ms = performance.now() - t0;
  inferMsEma = inferMsEma ? inferMsEma * 0.85 + ms * 0.15 : ms;
  let pts256 = heatmapsToPointsSoft(out[shgSession.outputNames[0]], 256);
  if (needFlip) {
    pts256 = pts256.map(([x, y]) => [255 - x, y]);
  }
  const scale = sidePx / 256;
  return pts256.map(([x, y]) => [ox + x * scale, oy + y * scale]);
}

async function inferFrame(vw, vh) {
  if (!geo || !tip || !side || !shgSession) return;
  let { cx, cy, side: sideLen } = geo;
  const half = sideLen * 0.5;
  if (Math.abs(tip.x - cx) > half * 0.55 || Math.abs(tip.y - cy) > half * 0.55) {
    cx = tip.x;
    cy = tip.y;
    geo = { cx, cy, side: sideLen };
  }

  const needFlip = side === "LEFT";
  const { ox, oy, sidePx } = drawSquareCrop(cx, cy, sideLen, needFlip);
  lastBox = [
    Math.max(0, Math.round(cx - sidePx * 0.5)),
    Math.max(0, Math.round(cy - sidePx * 0.5)),
    Math.min(vw, Math.round(cx + sidePx * 0.5)),
    Math.min(vh, Math.round(cy + sidePx * 0.5)),
  ];

  let pts = await runShg(needFlip, ox, oy, sidePx);
  let ok = landmarksOk(pts, tip, sidePx);

  if (!ok || firstLock) {
    // try opposite flip
    drawSquareCrop(cx, cy, sideLen, !needFlip);
    const pts2 = await runShg(!needFlip, ox, oy, sidePx);
    if (landmarksOk(pts2, tip, sidePx) && (!ok || firstLock)) {
      pts = pts2;
      ok = true;
    }
  }

  if (ok) {
    rawPts = pts;
    // expand crop if landmarks stick out
    let x0 = Infinity,
      y0 = Infinity,
      x1 = -Infinity,
      y1 = -Infinity;
    for (const [x, y] of pts) {
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
    const need = Math.max(
      cx - half - x0,
      x1 - (cx + half),
      cy - half - y0,
      y1 - (cy + half),
      0
    );
    if (need > 2) {
      geo = { cx, cy, side: sideLen + 2 * need + sideLen * 0.04 };
    }
  }
}

async function loopLive(ts) {
  if (!live) return;
  rafId = requestAnimationFrame(loopLive);

  const frame = paintVideoMirrored();
  if (!frame) return;
  const { vw, vh } = frame;
  const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : 1 / 30;
  lastTs = ts;
  frameIdx++;

  if (yoloPose && frameIdx % YOLO_EVERY === 1) {
    yoloPose
      .detect(video)
      .then((y) => {
        if (!y) return;
        if (side && y.side !== side) {
          smoother.reset();
          rawPts = null;
          firstLock = true;
          geo = null;
        }
        lastYolo = y;
        updateGeoFromYolo(y, vw, vh);
      })
      .catch((e) => {
        console.error(e);
        setStatus(`YOLO error: ${e?.message || e}`);
      });
  }

  if (!inferBusy && geo && tip && shgSession) {
    inferBusy = true;
    inferFrame(vw, vh)
      .catch((e) => setStatus(`Infer error: ${e?.message || e}`))
      .finally(() => {
        inferBusy = false;
      });
  }

  let smoothedDisp = null;
  if (rawPts) {
    const snap = firstLock;
    const sm = smoother.update(rawPts, dt, side, {
      maxStepPx: 12,
      snap,
    });
    if (snap) firstLock = false;
    smoothedDisp = sm.map(([x, y]) => [mirrorX(x, vw), y]);
  }

  let boxDisp = null;
  if (lastBox) {
    const [x1, y1, x2, y2] = lastBox;
    boxDisp = [mirrorX(x2, vw), y1, mirrorX(x1, vw), y2];
  }
  const tipDisp = tip ? [mirrorX(tip.x, vw), tip.y] : null;

  const instFps = 1 / Math.max(dt, 1e-3);
  fpsEma = fpsEma ? fpsEma * 0.9 + instFps * 0.1 : instFps;
  drawHud(
    smoothedDisp,
    boxDisp,
    tipDisp,
    `LIVE ${fpsEma.toFixed(0)} fps · SHGNet ${inferMsEma.toFixed(0)} ms · ${side || "?"}`
  );
  setStatus(
    `LIVE ${fpsEma.toFixed(0)} FPS · SHGNet ${inferMsEma.toFixed(0)} ms (wasm)\n` +
      `Ear: ${side || "—"} · 55 pts ${smoothedDisp ? "on" : "…"} · ONNX only`
  );
}

async function startCamera() {
  if (live) return;
  if (!shgSession || !yoloPose) {
    setStatus("Models not ready — wait for Ready or click Load models.");
    return;
  }
  try {
    setStatus("Requesting camera…");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 540 },
        frameRate: { ideal: 30 },
      },
    });
  } catch (e) {
    setStatus(`Camera error: ${e?.name || e}`);
    return;
  }
  video.srcObject = stream;
  video.style.display = "none";
  await video.play();
  live = true;
  rawPts = null;
  lastYolo = null;
  lastBox = null;
  geo = null;
  tip = null;
  side = null;
  firstLock = true;
  frameIdx = 0;
  smoother.reset();
  lastTs = 0;
  updateButtons();
  rafId = requestAnimationFrame(loopLive);
}

function stopCamera() {
  live = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = 0;
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
  video.srcObject = null;
  updateButtons();
  setStatus("Camera stopped.");
}

loadBtn.addEventListener("click", () => loadModels());
startCamBtn.addEventListener("click", () =>
  startCamera().catch((e) => setStatus(String(e)))
);
stopCamBtn.addEventListener("click", () => stopCamera());
window.addEventListener("beforeunload", () => stopCamera());

reportSizes();
updateButtons();
loadModels();
