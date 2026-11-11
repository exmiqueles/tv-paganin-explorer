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
