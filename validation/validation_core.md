# Validation of the browser implementation

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
