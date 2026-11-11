# TV–Paganin explorer

Interactive companion to

> E. X. Miqueles and D. M. Paganin, *A total-variation extension of Paganin-method phase retrieval and its flux-field interpretation* (submitted).

**Live version:** `https://<your-user>.github.io/tv-paganin-explorer/` *(enable GitHub Pages: Settings → Pages → branch `main`, folder `/`)*

`index.html` is a single, self-contained page (no external dependencies, works offline). It embeds the red-currant dataset used in the paper and computes, in the browser:

* the Paganin-method solution $u_{L_2}$ (proximal map of $\tfrac{\ell}{2}\|\nabla u\|^2$),
* the TV solution $u_{\mathrm{TV}}$ and the dual flux field $\mathbf p$ (Algorithm 1 of the paper: dual projected gradient $p \leftarrow \Pi_{\|p\|\le1}(p + \tau\nabla(\mathrm{div}\,p + r/\ell))$, i.e. Chambolle's projection algorithm in its projected-gradient form, stopped on the relative primal–dual gap),
* the effective diffusivity $D$.

You can also upload your own normalized ratio $r = I/I_0$ as a 2D `.npy` file (or an 8-bit image), choose a region, and vary $\ell_{\mathrm{PAG}}$, $\ell_{\mathrm{TV}}$, $\beta$, etc. Results can be exported as `.npy`.

Try $\ell_{\mathrm{TV}} = 10^3$ and then $\ell_{\mathrm{TV}} = 10^{-1}$: in the first case the projection onto $\|\mathbf p\|\le 1$ is never active and the iteration is exactly the explicit heat flow (Remark 2 of the paper) — a blur when stopped early, the constant image if run to convergence; in the second, the flux limit is reached along edges and fibers and the edge-preserving TV behaviour appears.

Every result records the parameters it was computed with; results are marked STALE when the controls change, and D refuses to combine $u$, $\mathbf p$ from one $\ell_{\mathrm{TV}}$ with another. Exports include a metadata JSON (data, region, parameters, stopping rules, convergence status).

## Validation

See `validation/validation.md` (also embedded in the page under *Validation report*). It contains:

1. comparison with the reference Python implementation on a 512×512 region (`validate.js`): $u_{L_2}$ to 5e-16, $u_{\mathrm{TV}}$ and $\mathbf p$ bit-identical (same iteration count), $D$ to 2e-6 – 1.5e-5 against tightly converged L-BFGS-B, with the 16-bit quantization effect of the embedded data reported separately (`make_ref_quant.py`);
2. unit tests of the numerical core (`tests.js`): FFT, adjointness, sign conventions, gap on constant and low-contrast images, linear regime, D degenerate cases;
3. end-to-end browser tests (`browser_tests.py`): parameter provenance/staleness, locking during computations, presets, input validation, metadata export;
4. a list of what was **not** verified.

```
cd validation
node validate.js && node tests.js                       # Node >= 18
python browser_tests.py ../index.html inputs # Playwright + Chromium
python make_ref.py ratio.npy                 # regenerate the Python reference (ref/)
python make_ref_quant.py ratio_samantha.npy
```

## Building

`index.html` is generated from `src/template.html`, `src/core.js` and the quantized dataset:

```
python build.py
```

## Data

Red-currant propagation-based X-ray image from S. Berujon and E. Ziegler, *Phys. Rev. Applied* 5, 044014 (2016), ESRF; provided by S. J. Alloo and E. Brun.

## Use of AI

The browser implementation (`src/`), the validation scripts and this README were developed with the assistance of Claude (Anthropic), from the authors' reference Python implementation, and checked against it as described in `validation/validation.md`.
