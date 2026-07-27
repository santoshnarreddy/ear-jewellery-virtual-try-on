/** One Euro landmarks — moderate smooth + rest freeze + snap. */
export class OneEuro1D {
  constructor(minCutoff = 0.35, beta = 0.008, dCutoff = 0.9) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = null;
  }

  static alpha(cutoff, dt) {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / Math.max(dt, 1e-6));
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = null;
  }

  filter(x, dt) {
    if (this.xPrev == null) {
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dx = (x - this.xPrev) / Math.max(dt, 1e-6);
    const aD = OneEuro1D.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = OneEuro1D.alpha(cutoff, dt);
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat;
    this.dxPrev = dxHat;
    return xHat;
  }
}

export class OneEuroLandmarks {
  constructor(
    n = 55,
    minCutoff = 0.35,
    beta = 0.008,
    dCutoff = 0.9,
    restSpeedPx = 16,
    restHoldFrames = 2
  ) {
    this.n = n;
    this.fx = Array.from({ length: n }, () => new OneEuro1D(minCutoff, beta, dCutoff));
    this.fy = Array.from({ length: n }, () => new OneEuro1D(minCutoff, beta, dCutoff));
    this.side = null;
    this.lastOut = null;
    this.restFrames = 0;
    this.restSpeedPx = restSpeedPx;
    this.restHoldFrames = restHoldFrames;
  }

  reset() {
    for (const f of this.fx) f.reset();
    for (const f of this.fy) f.reset();
    this.side = null;
    this.lastOut = null;
    this.restFrames = 0;
  }

  update(pts, dt, side = null, { maxStepPx = 12, snap = false } = {}) {
    if (side && this.side && side !== this.side) this.reset();
    if (side) this.side = side;

    const n = Math.min(pts.length, this.n);
    let cur = pts.map((p) => [p[0], p[1]]);

    if (snap || this.lastOut == null) {
      for (let i = 0; i < n; i++) {
        this.fx[i].xPrev = cur[i][0];
        this.fy[i].xPrev = cur[i][1];
        this.fx[i].dxPrev = 0;
        this.fy[i].dxPrev = 0;
      }
      this.lastOut = cur.slice(0, n).map((p) => [p[0], p[1]]);
      this.restFrames = 0;
      return this.lastOut.map((p) => [p[0], p[1]]);
    }

    if (maxStepPx > 0) {
      for (let i = 0; i < n; i++) {
        const dx = cur[i][0] - this.lastOut[i][0];
        const dy = cur[i][1] - this.lastOut[i][1];
        const dist = Math.hypot(dx, dy);
        if (dist > maxStepPx) {
          const s = maxStepPx / dist;
          cur[i][0] = this.lastOut[i][0] + dx * s;
          cur[i][1] = this.lastOut[i][1] + dy * s;
        }
      }
    }

    if (this.restSpeedPx > 0) {
      const speeds = [];
      for (let i = 0; i < n; i++) {
        speeds.push(
          Math.hypot(cur[i][0] - this.lastOut[i][0], cur[i][1] - this.lastOut[i][1]) /
            Math.max(dt, 1e-6)
        );
      }
      speeds.sort((a, b) => a - b);
      const med = speeds[Math.floor(speeds.length / 2)];
      if (med < this.restSpeedPx) {
        this.restFrames++;
        if (this.restFrames >= this.restHoldFrames) {
          for (let i = 0; i < n; i++) {
            this.fx[i].xPrev = this.lastOut[i][0];
            this.fy[i].xPrev = this.lastOut[i][1];
            this.fx[i].dxPrev = 0;
            this.fy[i].dxPrev = 0;
          }
          return this.lastOut.map((p) => [p[0], p[1]]);
        }
      } else {
        this.restFrames = 0;
      }
    }

    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = [
        this.fx[i].filter(cur[i][0], dt),
        this.fy[i].filter(cur[i][1], dt),
      ];
    }
    this.lastOut = out;
    return out;
  }
}
