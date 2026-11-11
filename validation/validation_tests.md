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
