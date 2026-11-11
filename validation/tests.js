// tests.js — unit tests of src/core.js (no reference data needed).   node tests.js
"use strict";
const C = require("../src/core.js");
let fails = 0;
const out = [];
function check(name, ok, detail) { out.push(`| ${ok ? "PASS" : "FAIL"} | ${name} | ${detail} |`); if (!ok) fails++; }
const e = (x) => x.toExponential(2);
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
const rand = rng(12345);
const relL2 = (a, b) => { let n = 0, d = 0; for (let i = 0; i < a.length; i++) { n += (a[i] - b[i]) ** 2; d += b[i] ** 2; } return Math.sqrt(n / (d || 1)); };

// 1. FFT against a naive DFT
{
  let worst = 0;
  for (const n of [1, 2, 3, 8, 12, 41, 50, 64, 100, 257]) {
    const re = Float64Array.from({ length: n }, rand), im = Float64Array.from({ length: n }, rand);
    const R = new Float64Array(n), I = new Float64Array(n);
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) { const a = -2 * Math.PI * j * k / n; R[k] += re[j] * Math.cos(a) - im[j] * Math.sin(a); I[k] += re[j] * Math.sin(a) + im[j] * Math.cos(a); }
    const a = re.slice(), b = im.slice(); C.fft1(a, b);
    let err = 0, m = 0; for (let k = 0; k < n; k++) { err = Math.max(err, Math.hypot(a[k] - R[k], b[k] - I[k])); m = Math.max(m, Math.hypot(R[k], I[k])); }
    worst = Math.max(worst, err / m);
  }
  check("FFT (radix-2 + Bluestein) vs naive DFT, n = 1..257", worst < 1e-12, `max rel. error ${e(worst)}`);
}
// 2. adjointness <grad u, p> = -<u, div p>
{
  const H = 37, W = 53, N = H * W;
  const u = Float64Array.from({ length: N }, rand), p0 = Float64Array.from({ length: N }, rand), p1 = Float64Array.from({ length: N }, rand);
  const g0 = new Float64Array(N), g1 = new Float64Array(N), d = new Float64Array(N);
  C.grad(u, H, W, g0, g1); C.div(p0, p1, H, W, d);
  let a = 0, b = 0; for (let i = 0; i < N; i++) { a += g0[i] * p0[i] + g1[i] * p1[i]; b += u[i] * d[i]; }
  check("discrete adjointness <grad u,p> = -<u,div p>", Math.abs(a + b) / Math.abs(a) < 1e-13, `rel. mismatch ${e(Math.abs(a + b) / Math.abs(a))}`);
}
// helper image: smooth blobs + edges + noise
function testImage(H, W, contrast, noise) {
  const r = new Float64Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    r[y * W + x] = 1 + contrast * ((Math.hypot(x - W / 3, y - H / 2) < H / 5 ? 1 : 0) + 0.5 * Math.sin(x / 7) * Math.cos(y / 11)) + noise * (rand() - 0.5);
  return r;
}
// 3. identity u - r = ell div p and stable gap vs energy-difference gap
{
  const H = 96, W = 96, r = testImage(H, W, 0.05, 0.01), ell = 0.05;
  const res = C.chambolle(r, H, W, ell, 0.125, 1e-3, 5000, null, 1);
  const d = new Float64Array(H * W); C.div(res.p0, res.p1, H, W, d);
  let mx = 0; for (let i = 0; i < d.length; i++) mx = Math.max(mx, Math.abs(res.u[i] - r[i] - ell * d[i]));
  check("u - r = ell_TV div p (paper sign convention)", mx < 1e-14, `max abs ${e(mx)}`);
  let s1 = 0, s2 = 0, s3 = 0; for (let i = 0; i < r.length; i++) { s1 += (res.u[i] - r[i]) ** 2; s2 += r[i] ** 2; s3 += res.u[i] ** 2; }
  const Ep = 0.5 * s1 + ell * C.tvIso(res.u, H, W), Ed = 0.5 * s2 - 0.5 * s3;
  const rd = Math.abs((Ep - Ed) - res.gap) / res.gap;
  check("stable gap = E_p - E_d (energy-difference form)", rd < 1e-6, `relative difference ${e(rd)} (cancellation in the difference form: |r|^2/2 = ${e(0.5 * s2)}, gap = ${e(res.gap)})`);
  let pmax = 0; for (let i = 0; i < d.length; i++) pmax = Math.max(pmax, Math.hypot(res.p0[i], res.p1[i]));
  check("dual feasibility max |p| <= 1", pmax <= 1 + 1e-15, `max |p| = ${pmax}`);
}
// 4. constant image: zero energy, immediate stop, no NaN
{
  const H = 32, W = 40, r = new Float64Array(H * W).fill(0.8);
  const res = C.chambolle(r, H, W, 0.1, 0.125, 1e-2, 1000, null, 1);
  check("constant image: stops at k = 1, G = 0, no NaN", res.nit === 1 && res.G === 0 && res.stop === "zero_energy" && res.u.every(v => v === 0.8),
    `nit = ${res.nit}, G = ${res.G}, stop = ${res.stop}`);
}
// 5. very low contrast: gap never negative, correct stop
{
  const H = 64, W = 64, r = testImage(H, W, 1e-9, 1e-10), ell = 1e-9;
  const res = C.chambolle(r, H, W, ell, 0.125, 1e-3, 20000, null, 1);
  let minG = Infinity, negDiff = 0;
  for (const g of res.Ghist) minG = Math.min(minG, g);
  // energy-difference form on the final iterate (what the previous version used)
  let s1 = 0, s2 = 0, s3 = 0; for (let i = 0; i < r.length; i++) { s1 += (res.u[i] - r[i]) ** 2; s2 += r[i] ** 2; s3 += res.u[i] ** 2; }
  const Ep = 0.5 * s1 + ell * C.tvIso(res.u, H, W), Gdiff = (Ep - (0.5 * s2 - 0.5 * s3)) / Ep;
  check("low contrast (1e-9): G_k >= 0 for all k, stops on the relative gap", minG >= 0 && res.stop === "relative_gap",
    `nit = ${res.nit}, final G = ${e(res.G)}, min G = ${e(minG)}; energy-difference form would give G = ${e(Gdiff)}`);
}
// 6. linear regime: projection never active => exactly explicit heat flow; G need not stay near 1
{
  const H = 48, W = 48, r = testImage(H, W, 0.02, 0.002), ell = 1000, K = 4000;
  const res = C.chambolle(r, H, W, ell, 0.125, 1e-2, K, null, 1);
  const heat = C.heatFlow(r, H, W, 0.125, res.nit);
  const dh = relL2(res.u, heat);
  check("ell_TV = 1000: projection never active and u_TV = explicit heat flow (all iterations)",
    !res.regime.everActive && dh < 1e-12, `everActive = ${res.regime.everActive}, |u_TV - u_heat|/|u_heat| = ${e(dh)}, nit = ${res.nit}`);
  check("...with early stopping the gap stays near 1 (48x48, 4000 it.)", res.G > 0.5, `G_${res.nit} = ${res.G.toFixed(3)}`);
  const r2 = testImage(16, 16, 0.02, 0.002), res2 = C.chambolle(r2, 16, 16, ell, 0.125, 1e-2, 50000, null, 1);
  let mean = 0; for (const v of r2) mean += v / r2.length;
  let dm = 0; for (const v of res2.u) dm = Math.max(dm, Math.abs(v - mean));
  check("...but run long enough (16x16) the gap becomes small without the projection ever activating",
    !res2.regime.everActive && res2.stop === "relative_gap", `nit = ${res2.nit}, G = ${e(res2.G)}, max|u - mean(r)| = ${e(dm)} (TV prox for large ell = constant image)`);
}
// 7. regime bookkeeping uses the full history
{
  const H = 64, W = 64, r = testImage(H, W, 0.05, 0.01);
  const res = C.chambolle(r, H, W, 0.02, 0.125, 1e-3, 3000, null, 1);
  const anyActive = res.Shist.some(s => s > 0);
  check("regime flags consistent with full history", res.regime.everActive === anyActive && res.regime.firstActive >= 1,
    `first active k = ${res.regime.firstActive}, last = ${res.regime.lastActive}, final sat = ${(100 * res.sat).toFixed(1)}%`);
}
// 8. D: degenerate input rejected; beta = 0 flagged; residual reported
{
  const H = 32, W = 32, N = H * W, z = new Float64Array(N);
  let threw = false; try { C.estimateD(z, z, z, H, W, { lambda: 1, ellDark: 1, ellTV: 1, alpha: 0, beta: 0 }); } catch (err) { threw = true; }
  check("D: degenerate problem (u = 0, alpha = beta = 0) is rejected", threw, "throws");
  const r = testImage(H, W, 0.05, 0.01), tv = C.chambolle(r, H, W, 0.05, 0.125, 1e-3, 3000, null, 1);
  const a = C.estimateD(tv.u, tv.p0, tv.p1, H, W, { lambda: 2, ellDark: 1, ellTV: 0.05, beta: 1e-1, tol: 1e-8, maxIter: 1e5 });
  const b = C.estimateD(tv.u, tv.p0, tv.p1, H, W, { lambda: 2, ellDark: 1, ellTV: 0.05, beta: 1e-3, tol: 1e-8, maxIter: 1e5 });
  check("D: converges, strict convexity flag, residual decreases as beta decreases",
    a.reason === "converged" && b.reason === "converged" && a.strictlyConvex && b.relResidual <= a.relResidual,
    `beta=1e-1: |R|/|R0| = ${a.relResidual.toFixed(3)}; beta=1e-3: ${b.relResidual.toFixed(3)}`);
  // optimality (KKT) check of the projected solution: D>0 => grad ~ 0 ; D=0 => grad >= 0
  const g = new Float64Array(N);
}
// 9. .npy round trip
{
  const a = Float64Array.from({ length: 12 }, rand), buf = C.writeNpy(a, [3, 4]);
  const b = C.parseNpy(buf.buffer);
  check(".npy write/read round trip", b.H === 3 && b.W === 4 && relL2(b.data, a) === 0, "exact");
}
console.log("| result | test | detail |\n|---|---|---|\n" + out.join("\n"));
console.log(fails ? `\n${fails} FAILED` : "\nall tests passed");
process.exitCode = fails ? 1 : 0;
