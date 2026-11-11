# Validation report — TV–Paganin explorer v1.1.0

All results below were produced by executing the scripts in this folder (Node 22, Python 3 with NumPy/SciPy,
headless Chromium via Playwright). They were run by the developer (with Claude, Anthropic); the authors should
re-run them (`node validate.js`, `node tests.js`, `python browser_tests.py ../index.html <inputs>`).

## 1. Comparison with the reference Python implementation (validate.js)

Input: 512x512 region (rows 700-1212, cols 600-1112) of the red-currant dataset (uint16-embedded).
Parameters: ell_PAG = 197.71 px^2 (L = 7805.4), ell_TV = 0.1, tau = 0.125, gap tolerance = 0.001, ell_dark = 1, alpha = 0.
Reference: unmodified Python functions of pagtv.ipynb (scipy FFT, NumPy, L-BFGS-B run to tight convergence).

Stopping criteria: TV — relative primal–dual gap <= 0.001 (Python: energy-difference form; JS: pointwise form ell*sum(|grad u| - grad u.p), algebraically identical).
D (reference, 'converged') — L-BFGS-B with ftol = gtol = 1e-12, tol_loss = 1e-16, tol_D = 1e-12, maxiter = 30000; D (JS) — projected FISTA, relative projected-gradient norm < 1e-9.

## A. Implementation error (JS vs Python on the identical, quantized input)

| quantity | relative L2 error | max abs error | notes |
|---|---|---|---|
| input r (uint16 dequantization) | 0.00e+0 | 0.00e+0 |  |
| u_L2 (Paganin) | 5.05e-16 | 8.88e-16 | 458 ms |
| u_TV (Algorithm 1) | 0.00e+0 | 0.00e+0 | iterations JS = 8046 / Python = 8046; stop: relative_gap; G = 1.000e-3; 55.3 s |
| p (dual field) | 0.00e+0 | 0.00e+0 | sat JS=40.44% / Py=40.44% |
| D, beta=0.1 | 2.08e-6 | 4.09e-5 | JS: 120 it (converged), 0.8 s; F_JS/F_Py-1 = -1.58e-12; paper-setting L-BFGS-B vs converged: 0.00e+0 |
| D, beta=0.01 | 4.12e-6 | 2.37e-4 | JS: 450 it (converged), 3.3 s; F_JS/F_Py-1 = -1.38e-11; paper-setting L-BFGS-B vs converged: 1.19e-4 |
| D, beta=0.001 | 1.45e-5 | 6.90e-4 | JS: 1160 it (converged), 8.0 s; F_JS/F_Py-1 = -1.99e-10; paper-setting L-BFGS-B vs converged: 6.10e-4 |

## B. Effect of the 16-bit quantization of the embedded data (Python only: quantized vs original float64 input, same pipeline)

Quantization step: 2.992e-5 in r (r in [0.1616, 2.1222]).

| quantity | relative L2 difference | max abs difference | notes |
|---|---|---|---|
| r | 2.24e-5 | 1.50e-5 |  |
| u_L2 | 5.64e-7 | 4.26e-6 |  |
| u_TV | 2.16e-6 | 2.09e-5 | iterations: quantized 8046, unquantized 8046 |
| p | 1.31e-4 | 1.64e-3 |  |
| D, beta=0.1 | 1.68e-4 | 9.99e-3 |  |
| D, beta=0.01 | 8.22e-5 | 1.17e-2 |  |
| D, beta=0.001 | 4.74e-5 | 1.19e-2 |  |

Section A isolates the implementation/solver error; section B isolates the data-quantization effect.
For D, the JS solver (projected FISTA) and the reference solver (L-BFGS-B) are different algorithms; for beta > 0 the discrete
functional is strictly convex, so both converge to the same minimizer; the measured differences are consistent with their tolerances.
The 'paper setting' column compares the L-BFGS-B run with the stopping rule used for the paper figures against the tightly converged run.

## 2. Unit tests of the numerical core (tests.js)

| result | test | detail |
|---|---|---|
| PASS | FFT (radix-2 + Bluestein) vs naive DFT, n = 1..257 | max rel. error 1.33e-14 |
| PASS | discrete adjointness <grad u,p> = -<u,div p> | rel. mismatch 1.85e-14 |
| PASS | u - r = ell_TV div p (paper sign convention) | max abs 1.11e-16 |
| PASS | stable gap = E_p - E_d (energy-difference form) | relative difference 2.06e-8 (cancellation in the difference form: |r|^2/2 = 4.67e+3, gap = 8.45e-4) |
| PASS | dual feasibility max |p| <= 1 | max |p| = 1.0000000000000004 |
| PASS | constant image: stops at k = 1, G = 0, no NaN | nit = 1, G = 0, stop = zero_energy |
| PASS | low contrast (1e-9): G_k >= 0 for all k, stops on the relative gap | nit = 1896, final G = 9.99e-4, min G = 9.99e-4; energy-difference form would give G = 8.36e+3 |
| PASS | ell_TV = 1000: projection never active and u_TV = explicit heat flow (all iterations) | everActive = false, |u_TV - u_heat|/|u_heat| = 1.77e-16, nit = 4000 |
| PASS | ...with early stopping the gap stays near 1 (48x48, 4000 it.) | G_4000 = 0.995 |
| PASS | ...but run long enough (16x16) the gap becomes small without the projection ever activating | nit = 2868, G = 9.96e-3, max|u - mean(r)| = 2.69e-9 (TV prox for large ell = constant image) |
| PASS | regime flags consistent with full history | first active k = 4, last = 555, final sat = 80.5% |
| PASS | D: degenerate problem (u = 0, alpha = beta = 0) is rejected | throws |
| PASS | D: converges, strict convexity flag, residual decreases as beta decreases | beta=1e-1: |R|/|R0| = 0.878; beta=1e-3: 0.425 |
| PASS | .npy write/read round trip | exact |

all tests passed

## 3. End-to-end browser tests (browser_tests.py)

| result | browser test (Chromium, headless) | detail |
|---|---|---|
| PASS | Run all (512² preset): TV and D converge, TV regime reported | D done · 330 iterations · status converged · ‖R(D)‖/‖R(0)‖ = 0.460 · 2.2 s / TV regime: flux limit active on 25.8% of pixels at the final |
| PASS | run log lists parameters and convergence |  |
| PASS | item 1: changing ℓ_TV marks u_TV/p and D as STALE | ℓ_TV = 0.1 · 1477 it · STALE: parameters changed |
| PASS | item 1: "D only" refuses to mix u,p from ℓ_TV = 0.1 with ℓ_TV = 0.05 | error: the TV controls differ from the last TV run (ℓ_TV = 0.1, τ = 0.125, ε = 0.01, max i |
| PASS | item 1: restoring ℓ_TV clears the STALE mark | ℓ_TV = 0.1 · 1477 it |
| PASS | item 1: changing ℓ_PAG marks u_L2 and D as STALE | ℓ_PAG = 197.71 px² · STALE: parameters changed |
| PASS | item 2: data, region, source, upload and run controls disabled while computing | [True, True, True, True, True, True, True, True, True, True] |
| PASS | item 2: programmatic region change ignored while busy | {"x":600,"y":700,"w":512,"h":512} |
| PASS | item 2: Stop discards the computation (version bumped) and re-enables controls | stopped — the interrupted computation was discarded |
| PASS | item 5: preset restores all TV, Paganin, D and region parameters | ['0.1', '0.125', '0.01', '5000', '197.71', '0.01', '0', '1', '0.000001', '20000', '600', '700', '512', '512'] |
| PASS | item 9: τ ≥ 1/4 rejected before running | cannot run: • τ must satisfy 0 < τ < 1/4 |
| PASS | item 9: ℓ_TV ≤ 0 rejected | cannot run: • ℓ_TV must be a finite number > 0 |
| PASS | item 9: non-integer iteration limit rejected | cannot run: • D max iterations must be an integer in [1, 10⁷] |
| PASS | item 9: upload with NaN values rejected | could not use file: 40 non-finite values (NaN/Inf) in the input; please clean th |
| PASS | item 9: values ≤ 0 are counted and flagged (magenta), not clipped silently | x=0, y=0, 128×128 · 100 px ≤ 0 (magenta) |
| PASS | item 9: a negative pixel is painted magenta | [217, 70, 239] |
| PASS | item 3: constant image stops at k = 1 with G = 0 (no NaN) | TV done · 1 iterations · stop: E_p = 0 (constant input; u = r is optimal) · G = 0.00e+0 ·  |
| PASS | item 4: ℓ_TV = 1000 → "linear regime throughout" (history-based) | Linear regime throughout (k = 1…300): the projection was never active, |
| PASS | item 10: metadata JSON has version, region, parameters, stopping rule and status | file tvpag_20261110-203331_x600_y700_512x512_metadata.json |
| PASS | item 10: array file names share the run id with the metadata | tvpag_20261110-203331_x600_y700_512x512_uTV.npy |
| PASS | no JavaScript errors during the session |  |

## 4. Not verified

* Browsers other than headless Chromium (Safari, Firefox, mobile). The page needs `DecompressionStream`
  (Safari >= 16.4, Firefox >= 113, Chrome >= 80).
* Numerical comparison on the full 1850x1850 image (only the 512x512 region was compared; the full-image
  run was only timed, ~90 ms per TV iteration in Chromium). Agreement is expected because the same code path
  is used, but it was not measured.
* D for beta = 1e-4, and D with alpha > 0, against the Python implementation (alpha > 0 was only checked by an
  independent reviewer against a coordinate-descent solver on a small problem).
* Upload of 8-bit images (PNG/JPEG); only .npy uploads were tested.
* Any physical meaning of D: the tests concern the discrete optimization problem only.

## 5. Notes on the algorithm

* TV: dual projected gradient q = p + tau grad(div p + r/ell), p <- q / max(1, |q|) (Algorithm 1 / Eq. 13 of the
  paper; Chambolle 2005 projected form, not the semi-implicit 2004 fixed point). Converges for 0 < tau < 1/4.
* Gap: evaluated as ell * sum(|grad u| - grad u . p) >= 0, algebraically equal to E_p - E_d; the energy-difference
  form loses all accuracy at low contrast (test 5 of section 2: G = 8.4e3 instead of 1e-3).
* Linear regime: while no pixel reaches |q| > 1 the iterates equal the explicit heat flow exactly (tests 6);
  the relative gap then is close to 1 only for early stopping — run long enough, the iterates converge to the
  constant image and G becomes small without the projection ever activating (test 6, 16x16 case).
