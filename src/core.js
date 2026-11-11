// core.js — TV extension of Paganin-method phase retrieval (Miqueles & Paganin)
// Pure-JS port of the reference Python implementation (pagtv.ipynb).
// Conventions follow the Python code exactly:
//   * images are row-major Float64Array, index i = y*W + x  (axis 0 = y, axis 1 = x)
//   * forward-difference gradient with Neumann BC, divergence = -adjoint
//   * _paganin_: edge padding of 100 px, kernel 1/(1 + L k^2), k from fftfreq (cycles/pixel)
//   * Chambolle: p_code with u = r - ell div(p_code);  returned p = -p_code (paper: u = r + ell div p)
// Works both in the browser (Web Worker) and in Node (module.exports).

"use strict";

const Core = (function () {
  const VERSION = "1.1.0";

  // ------------------------------------------------------------------ FFT
  const _twiddleCache = new Map();
  function _twiddles(n) {
    let t = _twiddleCache.get(n);
    if (!t) {
      t = { c: new Float64Array(n / 2), s: new Float64Array(n / 2) };
      for (let k = 0; k < n / 2; k++) {
        const a = -2 * Math.PI * k / n;
        t.c[k] = Math.cos(a); t.s[k] = Math.sin(a);
      }
      _twiddleCache.set(n, t);
    }
    return t;
  }

  // in-place radix-2 forward FFT (n power of two)
  function fftPow2(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const tw = _twiddles(n);
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const wr = tw.c[k * step], wi = tw.s[k * step];
          const a = i + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  const _bluesteinCache = new Map();
  function _bluestein(n) {
    let b = _bluesteinCache.get(n);
    if (b) return b;
    let m = 1; while (m < 2 * n - 1) m <<= 1;
    const wr = new Float64Array(n), wi = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const a = Math.PI * ((k * k) % (2 * n)) / n;   // exact reduction of k^2
      wr[k] = Math.cos(a); wi[k] = -Math.sin(a);       // w_k = exp(-i pi k^2/n)
    }
    const br = new Float64Array(m), bi = new Float64Array(m);
    br[0] = wr[0]; bi[0] = -wi[0];
    for (let k = 1; k < n; k++) {
      br[k] = br[m - k] = wr[k];
      bi[k] = bi[m - k] = -wi[k];                       // conj(w_k)
    }
    fftPow2(br, bi);
    b = { m, wr, wi, br, bi, tr: new Float64Array(m), ti: new Float64Array(m) };
    _bluesteinCache.set(n, b);
    return b;
  }

  // forward FFT of length-n complex vector (any n), in place
  function fft1(re, im) {
    const n = re.length;
    if ((n & (n - 1)) === 0) { fftPow2(re, im); return; }
    const B = _bluestein(n), m = B.m, tr = B.tr, ti = B.ti;
    tr.fill(0); ti.fill(0);
    for (let k = 0; k < n; k++) {
      tr[k] = re[k] * B.wr[k] - im[k] * B.wi[k];
      ti[k] = re[k] * B.wi[k] + im[k] * B.wr[k];
    }
    fftPow2(tr, ti);
    for (let k = 0; k < m; k++) {
      const a = tr[k] * B.br[k] - ti[k] * B.bi[k];
      const c = tr[k] * B.bi[k] + ti[k] * B.br[k];
      tr[k] = a; ti[k] = -c;                            // conj for inverse via forward
    }
    fftPow2(tr, ti);
    for (let k = 0; k < n; k++) {
      const cr = tr[k] / m, ci = -ti[k] / m;            // conj back, scale
      re[k] = cr * B.wr[k] - ci * B.wi[k];
      im[k] = cr * B.wi[k] + ci * B.wr[k];
    }
  }

  // 2D FFT (forward or inverse) of H x W complex array, in place
  function fft2(re, im, H, W, inverse) {
    if (inverse) for (let i = 0; i < im.length; i++) im[i] = -im[i];
    const rr = new Float64Array(W), ri = new Float64Array(W);
    for (let y = 0; y < H; y++) {
      const o = y * W;
      for (let x = 0; x < W; x++) { rr[x] = re[o + x]; ri[x] = im[o + x]; }
      fft1(rr, ri);
      for (let x = 0; x < W; x++) { re[o + x] = rr[x]; im[o + x] = ri[x]; }
    }
    const cr = new Float64Array(H), ci = new Float64Array(H);
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) { cr[y] = re[y * W + x]; ci[y] = im[y * W + x]; }
      fft1(cr, ci);
      for (let y = 0; y < H; y++) { re[y * W + x] = cr[y]; im[y * W + x] = ci[y]; }
    }
    if (inverse) {
      const s = 1 / (H * W);
      for (let i = 0; i < re.length; i++) { re[i] *= s; im[i] = -im[i] * s; }
    }
  }

  function fftfreq(n) {
    const f = new Float64Array(n), h = Math.floor((n - 1) / 2);
    for (let i = 0; i < n; i++) f[i] = (i <= h ? i : i - n) / n;
    return f;
  }

  // ------------------------------------------------------------- Paganin (L2 prox)
  // L in the convention of _paganin_ (fftfreq in cycles/pixel): L = 4 pi^2 * ell_PAG[px^2]
  function paganin(r, H, W, L, PAD) {
    if (PAD === undefined) PAD = 100;
    const Hp = H + 2 * PAD, Wp = W + 2 * PAD;
    const re = new Float64Array(Hp * Wp), im = new Float64Array(Hp * Wp);
    for (let y = 0; y < Hp; y++) {
      const ys = Math.min(H - 1, Math.max(0, y - PAD));
      for (let x = 0; x < Wp; x++) {
        const xs = Math.min(W - 1, Math.max(0, x - PAD));
        re[y * Wp + x] = r[ys * W + xs];                  // np.pad(mode='edge')
      }
    }
    fft2(re, im, Hp, Wp, false);
    const ky = fftfreq(Hp), kx = fftfreq(Wp);
    for (let y = 0; y < Hp; y++) {
      const ky2 = ky[y] * ky[y];
      for (let x = 0; x < Wp; x++) {
        const k = 1.0 / (1.0 + L * (ky2 + kx[x] * kx[x]));
        re[y * Wp + x] *= k; im[y * Wp + x] *= k;
      }
    }
    fft2(re, im, Hp, Wp, true);
    const u = new Float64Array(H * W);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) u[y * W + x] = re[(y + PAD) * Wp + (x + PAD)];
    return u;
  }

  // ------------------------------------------------------------- grad / div (Neumann)
  function grad(u, H, W, g0, g1) {
    for (let y = 0; y < H; y++) {
      const o = y * W, last = (y === H - 1);
      for (let x = 0; x < W - 1; x++) {
        const i = o + x;
        g0[i] = last ? 0 : u[i + W] - u[i];
        g1[i] = u[i + 1] - u[i];
      }
      const i = o + W - 1;
      g0[i] = last ? 0 : u[i + W] - u[i];
      g1[i] = 0;
    }
  }
  function div(p0, p1, H, W, out) {
    for (let y = 0; y < H; y++) {
      const o = y * W;
      const first = (y === 0), last = (y === H - 1);
      for (let x = 0; x < W; x++) {
        const i = o + x;
        const a = first ? p0[i] : (last ? -p0[i - W] : p0[i] - p0[i - W]);
        const b = (x === 0) ? p1[i] : ((x === W - 1) ? -p1[i - 1] : p1[i] - p1[i - 1]);
        out[i] = a + b;
      }
    }
  }
  // chunked summation: plain sums over blocks of 1024, blocks combined with Neumaier compensation
  function Acc() { this.s = 0; this.c = 0; this.b = 0; this.n = 0; }
  Acc.prototype.add = function (v) {
    this.b += v;
    if (++this.n === 1024) { this._flush(); }
  };
  Acc.prototype._flush = function () {
    const v = this.b, s = this.s, t = s + v;
    this.c += (Math.abs(s) >= Math.abs(v)) ? (s - t) + v : (v - t) + s; this.s = t; this.b = 0; this.n = 0;
  };
  Acc.prototype.val = function () { this._flush(); return this.s + this.c; };
  function ksum() { return new Acc(); }
  function tvIso(u, H, W) {
    const acc = new Acc();
    for (let y = 0; y < H; y++) {
      const o = y * W, last = (y === H - 1);
      for (let x = 0; x < W; x++) {
        const i = o + x;
        const a = last ? 0 : u[i + W] - u[i];
        const b = (x < W - 1) ? u[i + 1] - u[i] : 0;
        acc.add(Math.sqrt(a * a + b * b));
      }
    }
    return acc.val();
  }

  // ------------------------------------------------------------- TV proximal map (Algorithm 1 of the paper)
  // Dual projected-gradient iteration (Chambolle's projection algorithm in its projected-gradient
  // form, cf. A. Chambolle, EMMCVPR 2005; J.-F. Aujol, JMIV 2009):
  //     q = p + tau * grad( div p - r/ell ),     p <- q / max(1, |q|)
  // which is Eq. (13) / Algorithm 1 of the paper (code sign convention p_c = -p, u = r - ell div p_c).
  // It is NOT the semi-implicit fixed-point update of Chambolle (JMIV 2004), p <- (p + tau g)/(1 + tau|g|).
  // Convergence of the projected gradient is guaranteed for 0 < tau < 1/4 (||div||^2 <= 8).
  //
  // Stopping: relative primal-dual gap. With v = u - r = -ell div p_c, the gap is evaluated in the
  // pointwise-nonnegative form
  //     gap = E_p(u) - E_d(p) = ell * sum_x ( |grad u| + grad u . p_c )  >= 0     (|p_c| <= 1)
  // which is algebraically identical to E_p - E_d but free of the cancellation in 0.5|r|^2 - 0.5|u|^2.
  //     E_p = 0.5 |u - r|^2 + ell TV(u) >= 0.
  // Converged when  gap <= eps * E_p   (relative),  or  E_p == 0  (r constant: u = r is optimal),
  // or gap <= 64 * machine_eps * E_p  (gap below rounding level).
  // onProgress(k, G, sat) every progressEvery iterations; return false to abort.
  const MACH_EPS = 2.220446049250313e-16;
  function chambolle(r, H, W, ell, tau, eps, maxIter, onProgress, progressEvery) {
    const N = H * W;
    const px = new Float64Array(N), py = new Float64Array(N);
    const d = new Float64Array(N), g = new Float64Array(N);
    const gx = new Float64Array(N), gy = new Float64Array(N);
    const u = new Float64Array(N), ux = new Float64Array(N), uy = new Float64Array(N);
    const Ghist = [], Shist = [];
    let k = 0, G = NaN, gap = NaN, Ep = NaN, sat = 0, aborted = false, stop = "max_iter";
    let firstActive = 0, lastActive = 0, maxSat = 0, negGap = 0;
    div(px, py, H, W, d);                       // d = div(p^(k)); reused from the previous iteration
    for (k = 1; k <= maxIter; k++) {
      for (let i = 0; i < N; i++) g[i] = d[i] - r[i] / ell;
      grad(g, H, W, gx, gy);
      let nsat = 0;
      for (let i = 0; i < N; i++) {
        const qx = px[i] + tau * gx[i], qy = py[i] + tau * gy[i];
        const nq = Math.sqrt(qx * qx + qy * qy);
        if (nq > 1.0) nsat++;
        const n = Math.max(1.0, nq);
        px[i] = qx / n; py[i] = qy / n;
      }
      div(px, py, H, W, d);
      const sv = ksum();
      for (let i = 0; i < N; i++) { u[i] = r[i] - ell * d[i]; const e = u[i] - r[i]; sv.add(e * e); }
      grad(u, H, W, ux, uy);
      const st = ksum(), sg = ksum();
      for (let i = 0; i < N; i++) {
        const m = Math.sqrt(ux[i] * ux[i] + uy[i] * uy[i]);
        st.add(m); sg.add(m + ux[i] * px[i] + uy[i] * py[i]);
      }
      Ep = 0.5 * sv.val() + ell * st.val();
      gap = ell * sg.val();
      if (gap < 0) { negGap++; gap = 0; }                 // rounding only (|p| <= 1 => terms >= 0)
      G = Ep > 0 ? gap / Ep : 0;
      sat = nsat / N;
      if (nsat > 0) { if (!firstActive) firstActive = k; lastActive = k; }
      if (sat > maxSat) maxSat = sat;
      Ghist.push(G); Shist.push(sat);
      if (onProgress && (k % progressEvery === 0)) {
        if (onProgress(k, G, sat) === false) { aborted = true; stop = "aborted"; break; }
      }
      if (Ep === 0) { stop = "zero_energy"; break; }
      if (gap <= eps * Ep) { stop = "relative_gap"; break; }
      if (gap <= 64 * MACH_EPS * Ep) { stop = "rounding_level"; break; }
    }
    if (k > maxIter) k = maxIter;
    for (let i = 0; i < N; i++) { px[i] = -px[i]; py[i] = -py[i]; }   // paper convention: u = r + ell div p
    return { u, p0: px, p1: py, nit: k, G, gap, Ep, sat, stop, aborted,
             regime: { firstActive, lastActive, maxSat, everActive: firstActive > 0, negGapClamped: negGap },
             Ghist, Shist };
  }

  // explicit heat flow u <- u + tau * div grad u, k steps (for the linear-regime check)
  function heatFlow(r, H, W, tau, k) {
    const N = H * W, u = Float64Array.from(r), a = new Float64Array(N), b = new Float64Array(N), d = new Float64Array(N);
    for (let j = 0; j < k; j++) { grad(u, H, W, a, b); div(a, b, H, W, d); for (let i = 0; i < N; i++) u[i] += tau * d[i]; }
    return u;
  }

  // ------------------------------------------------------------- effective diffusivity D
  // min_{D>=0} 0.5 mean|ellTV p - lam grad u + ellDark grad(D u)|^2 + 0.5 alpha mean|grad D|^2 + 0.5 beta mean D^2
  // (same functional and scaling as estimate_positive_D_ridge), discretized with the same grad/div.
  // For beta > 0 this discrete functional is strictly convex, so its minimizer is unique; any convergent
  // solver (here projected FISTA with gradient-based adaptive restart; L-BFGS-B in the paper) reaches it.
  // This says nothing about D being a physical diffusivity. Convergence is monitored by the norm of the
  // projected-gradient mapping, relative to its initial value. Returns also the flux-fit residual
  // R = ellTV p - lam grad u + ellDark grad(D u), its relative norm |R|/|R(D=0)| and the map |R|.
  function estimateD(u, p0, p1, H, W, opt, onProgress) {
    const N = H * W;
    const lam = opt.lambda, ed = opt.ellDark, et = opt.ellTV;
    const alpha = opt.alpha || 0, beta = opt.beta || 0;
    const maxIter = opt.maxIter || 5000, tol = opt.tol || 1e-6, every = opt.checkEvery || 10;
    const t0 = new Float64Array(N), t1 = new Float64Array(N);
    const a0 = new Float64Array(N), a1 = new Float64Array(N), tmp = new Float64Array(N);
    grad(u, H, W, a0, a1);
    let umax = 0;
    for (let i = 0; i < N; i++) {
      t0[i] = et * p0[i] - lam * a0[i]; t1[i] = et * p1[i] - lam * a1[i];
      umax = Math.max(umax, Math.abs(u[i]));
    }
    const Lip = (8 * ed * ed * umax * umax + 8 * alpha + beta) / N;
    if (!(Lip > 0) || !isFinite(Lip)) throw new Error("degenerate D problem (u = 0 and alpha = beta = 0, or non-finite input)");
    const w = new Float64Array(N), w0 = new Float64Array(N), w1 = new Float64Array(N);
    function lossGrad(D, gD) {
      for (let i = 0; i < N; i++) w[i] = D[i] * u[i];
      grad(w, H, W, w0, w1);
      const s = ksum();
      for (let i = 0; i < N; i++) {
        w0[i] = t0[i] + ed * w0[i]; w1[i] = t1[i] + ed * w1[i];
        s.add(w0[i] * w0[i] + w1[i] * w1[i]);
      }
      let loss = 0.5 * s.val() / N;
      div(w0, w1, H, W, tmp);
      for (let i = 0; i < N; i++) gD[i] = -ed * tmp[i] * u[i] / N;
      if (alpha !== 0) {
        grad(D, H, W, w0, w1);
        const sa = ksum();
        for (let i = 0; i < N; i++) sa.add(w0[i] * w0[i] + w1[i] * w1[i]);
        loss += 0.5 * alpha * sa.val() / N;
        div(w0, w1, H, W, tmp);
        for (let i = 0; i < N; i++) gD[i] += -alpha * tmp[i] / N;
      }
      if (beta !== 0) {
        const sb = ksum();
        for (let i = 0; i < N; i++) { sb.add(D[i] * D[i]); gD[i] += beta * D[i] / N; }
        loss += 0.5 * beta * sb.val() / N;
      }
      return loss;
    }
    const D = new Float64Array(N).fill(opt.D0 === undefined ? 1 : opt.D0);
    const Dp = new Float64Array(D), Y = new Float64Array(D), gY = new Float64Array(N);
    // projected-gradient-mapping norm, relative to its value at the start
    function pgNorm(X) {
      lossGrad(X, gY);
      let s = 0;
      for (let i = 0; i < N; i++) { const z = X[i] - Math.max(0, X[i] - gY[i] / Lip); s += z * z; }
      return Math.sqrt(s) * Lip;
    }
    const pg0 = pgNorm(D);
    let t = 1, k = 0, rel = 1, loss = NaN, reason = "max_iter", restarts = 0;
    for (k = 1; k <= maxIter; k++) {
      lossGrad(Y, gY);
      let dot = 0;
      for (let i = 0; i < N; i++) {
        const dn = Math.max(0, Y[i] - gY[i] / Lip);
        dot += (Y[i] - dn) * (dn - Dp[i]);
        D[i] = dn;
      }
      let tn = 0.5 * (1 + Math.sqrt(1 + 4 * t * t));
      if (dot > 0) { t = 1; tn = 1; restarts++; }       // adaptive restart
      const mom = (t - 1) / tn;
      for (let i = 0; i < N; i++) { Y[i] = D[i] + mom * (D[i] - Dp[i]); Dp[i] = D[i]; }
      t = tn;
      if (k % every === 0) {
        rel = pg0 > 0 ? pgNorm(D) / pg0 : 0;
        if (onProgress && onProgress(k, rel) === false) { reason = "aborted"; break; }
        if (rel < tol) { reason = "converged"; break; }
      }
    }
    if (k > maxIter) k = maxIter;
    if (pg0 === 0) reason = "optimal_at_start";
    loss = lossGrad(D, gY);
    // flux-fit residual
    for (let i = 0; i < N; i++) w[i] = D[i] * u[i];
    grad(w, H, W, w0, w1);
    const Rmag = new Float64Array(N), sR = ksum(), sT = ksum();
    for (let i = 0; i < N; i++) {
      const R0 = t0[i] + ed * w0[i], R1 = t1[i] + ed * w1[i];
      Rmag[i] = Math.sqrt(R0 * R0 + R1 * R1);
      sR.add(R0 * R0 + R1 * R1); sT.add(t0[i] * t0[i] + t1[i] * t1[i]);
    }
    const relResidual = sT.val() > 0 ? Math.sqrt(sR.val() / sT.val()) : 0;
    return { D, nit: k, rel, loss, reason, restarts, Rmag, relResidual, strictlyConvex: beta > 0 };
  }

  // ------------------------------------------------------------- .npy I/O
  function parseNpy(buf) {
    const b = new Uint8Array(buf);
    if (b[0] !== 0x93 || String.fromCharCode(b[1], b[2], b[3], b[4], b[5]) !== "NUMPY") throw new Error("not a .npy file");
    const major = b[6];
    const hlen = major === 1 ? (b[8] | (b[9] << 8)) : (b[8] | (b[9] << 8) | (b[10] << 16) | (b[11] << 24));
    const off = major === 1 ? 10 : 12;
    let hdr = ""; for (let i = 0; i < hlen; i++) hdr += String.fromCharCode(b[off + i]);
    const descr = /'descr':\s*'([^']+)'/.exec(hdr)[1];
    const fortran = /'fortran_order':\s*True/.test(hdr);
    const shape = /'shape':\s*\(([^)]*)\)/.exec(hdr)[1].split(",").map(s => s.trim()).filter(s => s.length).map(Number);
    if (shape.length !== 2) throw new Error("expected a 2D array, got shape (" + shape.join(",") + ")");
    if (descr[0] === ">") throw new Error("big-endian .npy not supported");
    const T = { "<f8": Float64Array, "<f4": Float32Array, "|u1": Uint8Array, "<u2": Uint16Array,
                "<i2": Int16Array, "<i4": Int32Array, "<u4": Uint32Array, "|i1": Int8Array }[descr];
    if (!T) throw new Error("unsupported dtype " + descr);
    const start = off + hlen, n = shape[0] * shape[1];
    const raw = new T(buf.slice(start, start + n * T.BYTES_PER_ELEMENT));
    const [H, W] = shape, out = new Float64Array(n);
    if (!fortran) for (let i = 0; i < n; i++) out[i] = raw[i];
    else for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = raw[x * H + y];
    return { data: out, H, W };
  }
  function writeNpy(arr, shape) {   // float64, C order
    let hdr = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + shape.join(", ") + (shape.length === 1 ? "," : "") + "), }";
    const total = 10 + hdr.length + 1;
    hdr += " ".repeat((64 - (total % 64)) % 64) + "\n";
    const out = new Uint8Array(10 + hdr.length + arr.length * 8);
    out.set([0x93, 78, 85, 77, 80, 89, 1, 0, hdr.length & 255, hdr.length >> 8]);
    for (let i = 0; i < hdr.length; i++) out[10 + i] = hdr.charCodeAt(i);
    out.set(new Uint8Array(new Float64Array(arr).buffer), 10 + hdr.length);
    return out;
  }

  // physical Paganin parameter: ell = z * (delta/beta) * lambda / (4 pi), returned in pixel^2
  function ellPagPx(z_m, energy_keV, pixel_m, deltaOverBeta) {
    const lam = 1.239841984e-9 / energy_keV;
    return z_m * deltaOverBeta * lam / (4 * Math.PI) / (pixel_m * pixel_m);
  }

  return { VERSION, fft1, fft2, fftfreq, paganin, grad, div, tvIso, chambolle, heatFlow, estimateD, parseNpy, writeNpy, ellPagPx };
})();

if (typeof module !== "undefined" && module.exports) module.exports = Core;
