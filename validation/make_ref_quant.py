# Effect of the 16-bit quantization of the embedded dataset, measured with the reference Python code only:
# same pipeline (same parameters and stopping rules) on the unquantized and on the quantized region.
# Usage: python make_ref_quant.py path/to/ratio_samantha.npy      (writes ref/quant_effect.json)
import numpy, numpy as np, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'pagtv_functions.py')).read())
src = open(os.path.join(HERE, 'make_ref.py')).read()
exec(src[src.index('def chambolle_gap'):src.index('ELL_TV = 0.1')])        # same chambolle_gap as make_ref.py
meta = json.load(open(os.path.join(HERE, 'ref', 'meta.json')))
r_full = np.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'ratio_samantha.npy')).astype(float)
y0, x0, S = meta['roi']
ru = r_full[y0:y0+S, x0:x0+S].copy()
rq = np.load(os.path.join(HERE, 'ref', 'r_roi.npy'))
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
mx = lambda a, b: float(np.abs(a - b).max())
res = dict(r=dict(rel=rel(rq, ru), max=mx(rq, ru)))
uL2u = _paganin_(ru, meta['L']); uL2q = np.load(os.path.join(HERE, 'ref', 'uL2.npy'))
res['uL2'] = dict(rel=rel(uL2q, uL2u), max=mx(uL2q, uL2u))
uu, p0u, p1u, nitu, Gu, satu = chambolle_gap(ru, meta['ell_tv'], eps=meta['eps'])
uq = np.load(os.path.join(HERE, 'ref', 'uTV.npy')); p0q = np.load(os.path.join(HERE, 'ref', 'p0.npy')); p1q = np.load(os.path.join(HERE, 'ref', 'p1.npy'))
res['uTV'] = dict(rel=rel(uq, uu), max=mx(uq, uu), nit_unquantized=int(nitu), nit_quantized=int(meta['nit']))
res['p'] = dict(rel=rel(np.stack([p0q, p1q]), np.stack([p0u, p1u])), max=mx(np.stack([p0q, p1q]), np.stack([p0u, p1u])))
for beta in [1e-1, 1e-2, 1e-3]:
    Du, _, _ = estimate_positive_D_ridge(uu, p0u, p1u, lambda_phys=meta['ell_pag_px'], ell_dark=1.0, ell_tv=meta['ell_tv'],
                                         alpha=0, beta=beta, maxiter=30000, D0=1, tol_loss=1e-16, tol_D=1e-12)
    Dq = np.load(os.path.join(HERE, 'ref', f'D_tight_{beta:g}.npy'))
    res[f'D_{beta:g}'] = dict(rel=rel(Dq, Du), max=mx(Dq, Du))
    print(beta, res[f'D_{beta:g}'], flush=True)
json.dump(res, open(os.path.join(HERE, 'ref', 'quant_effect.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
