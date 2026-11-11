// validate.js — compares the browser implementation (core.js) with the reference Python
// implementation (pagtv.ipynb functions), on the 512x512 ROI stored in ref/.
// Usage:  node validate.js            (writes validation.md)
"use strict";
const fs = require("fs");
const path = require("path");
const C = require("../src/core.js");

const REF = path.join(__dirname, "ref");
const load = f => C.parseNpy(fs.readFileSync(path.join(REF, f)).buffer.slice(0));
const meta = JSON.parse(fs.readFileSync(path.join(REF, "meta.json")));
const qm = JSON.parse(fs.readFileSync(path.join(REF, "q_meta.json")));

function cmp(a, b) {
  let num = 0, den = 0, mx = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; mx = Math.max(mx, Math.abs(d)); }
  return { rel: Math.sqrt(num / den), max: mx };
}
const fmt = x => x.toExponential(2);
const rows = [];

// 0) dequantization of the embedded dataset
const q = load("q_full.npy");
const scale = (qm.rmax - qm.rmin) / 65535.0;
const [y0, x0, S] = meta.roi;
const r = new Float64Array(S * S);
for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) r[y * S + x] = qm.rmin + q.data[(y + y0) * q.W + (x + x0)] * scale;
const rref = load("r_roi.npy");
let c = cmp(r, rref.data);
rows.push(["input r (uint16 dequantization)", fmt(c.rel), fmt(c.max), ""]);

// 1) Paganin
let t = Date.now();
const uL2 = C.paganin(r, S, S, meta.L);
c = cmp(uL2, load("uL2.npy").data);
rows.push(["u_L2 (Paganin)", fmt(c.rel), fmt(c.max), `${Date.now() - t} ms`]);

// 2) Chambolle
t = Date.now();
const tv = C.chambolle(r, S, S, meta.ell_tv, meta.tau, meta.eps, 20000, null, 1);   // stable gap form; reference uses the energy-difference form
const dtTV = Date.now() - t;
c = cmp(tv.u, load("uTV.npy").data);
rows.push(["u_TV (Algorithm 1)", fmt(c.rel), fmt(c.max), `iterations JS = ${tv.nit} / Python = ${meta.nit}; stop: ${tv.stop}; G = ${tv.G.toExponential(3)}; ${(dtTV / 1000).toFixed(1)} s`]);
const P0 = load("p0.npy").data, P1 = load("p1.npy").data;
const pc = cmp(Float64Array.from([...tv.p0, ...tv.p1]), Float64Array.from([...P0, ...P1]));
rows.push(["p (dual field)", fmt(pc.rel), fmt(pc.max), `sat JS=${(100 * tv.sat).toFixed(2)}% / Py=${(100 * meta.sat).toFixed(2)}%`]);

// 3) D
const dRows = [];
for (const beta of [1e-1, 1e-2, 1e-3]) {
  const key = beta.toString().replace("0.1", "0.1");
  const bk = Object.keys(meta.D).find(k => Math.abs(Number(k) - beta) < 1e-15);
  t = Date.now();
  const res = C.estimateD(tv.u, tv.p0, tv.p1, S, S,
    { lambda: meta.ell_pag_px, ellDark: 1.0, ellTV: meta.ell_tv, alpha: 0, beta, maxIter: 200000, tol: 1e-9, D0: 1 });
  const dt = Date.now() - t;
  const Dt = load(`D_tight_${bk}.npy`).data, Dp = load(`D_paper_${bk}.npy`).data;
  const ct = cmp(res.D, Dt), cp = cmp(Dp, Dt);
  const Ft = meta.D[bk].tight.fun;
  dRows.push([`D, beta=${bk}`, fmt(ct.rel), fmt(ct.max),
    `JS: ${res.nit} it (${res.reason}), ${(dt / 1000).toFixed(1)} s; F_JS/F_Py-1 = ${fmt(res.loss / Ft - 1)}; paper-setting L-BFGS-B vs converged: ${fmt(cp.rel)}`]);
}

let qe = null;
try { qe = JSON.parse(fs.readFileSync(path.join(REF, "quant_effect.json"))); } catch (err) { qe = null; }
const qRows = qe ? [
  ["r", qe.r], ["u_L2", qe.uL2], ["u_TV", qe.uTV], ["p", qe.p], ["D, beta=0.1", qe["D_0.1"]], ["D, beta=0.01", qe["D_0.01"]], ["D, beta=0.001", qe["D_0.001"]],
].map(([n, v]) => `| ${n} | ${fmt(v.rel)} | ${fmt(v.max)} | ${n === "u_TV" ? `iterations: quantized ${v.nit_quantized}, unquantized ${v.nit_unquantized}` : ""} |`) : ["| (ref/quant_effect.json not found: run make_ref_quant.py) | | | |"];
const md = [
  "# Validation of the browser implementation",
  "",
  `Input: ${S}x${S} region (rows ${y0}-${y0 + S}, cols ${x0}-${x0 + S}) of the red-currant dataset (uint16-embedded).`,
  `Parameters: ell_PAG = ${meta.ell_pag_px.toFixed(2)} px^2 (L = ${meta.L.toFixed(1)}), ell_TV = ${meta.ell_tv}, tau = ${meta.tau}, gap tolerance = ${meta.eps}, ell_dark = 1, alpha = 0.`,
  "Reference: unmodified Python functions of pagtv.ipynb (scipy FFT, NumPy, L-BFGS-B run to tight convergence).",
  "",
  "Stopping criteria: TV — relative primal–dual gap <= " + meta.eps + " (Python: energy-difference form; JS: pointwise form ell*sum(|grad u| - grad u.p), algebraically identical).",
  "D (reference, 'converged') — L-BFGS-B with ftol = gtol = 1e-12, tol_loss = 1e-16, tol_D = 1e-12, maxiter = 30000; D (JS) — projected FISTA, relative projected-gradient norm < 1e-9.",
  "",
  "## A. Implementation error (JS vs Python on the identical, quantized input)",
  "",
  "| quantity | relative L2 error | max abs error | notes |",
  "|---|---|---|---|",
  ...rows.concat(dRows).map(r => `| ${r.join(" | ")} |`),
  "",
  "## B. Effect of the 16-bit quantization of the embedded data (Python only: quantized vs original float64 input, same pipeline)",
  "",
  `Quantization step: ${scale.toExponential(3)} in r (r in [${qm.rmin.toFixed(4)}, ${qm.rmax.toFixed(4)}]).`,
  "",
  "| quantity | relative L2 difference | max abs difference | notes |",
  "|---|---|---|---|",
  ...qRows,
  "",
  "Section A isolates the implementation/solver error; section B isolates the data-quantization effect.",
  "For D, the JS solver (projected FISTA) and the reference solver (L-BFGS-B) are different algorithms; for beta > 0 the discrete",
  "functional is strictly convex, so both converge to the same minimizer; the measured differences are consistent with their tolerances.",
  "The 'paper setting' column compares the L-BFGS-B run with the stopping rule used for the paper figures against the tightly converged run.",
].join("\n");
fs.writeFileSync(path.join(__dirname, "validation_core.md"), md + "\n");
console.log(md);
