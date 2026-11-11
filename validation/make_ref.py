# Reference outputs from the original Python implementation (pagtv.ipynb functions, unmodified)
# Usage: python make_ref.py path/to/ratio_samantha.npy   (writes ref/)
import numpy, numpy as np, json, time, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'pagtv_functions.py')).read())

OUT = os.path.join(HERE, 'ref') + os.sep
r_full = np.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'ratio_samantha.npy')).astype(float)

# --- uint16 quantization used for the embedded dataset (same arithmetic as the widget)
rmin, rmax = float(r_full.min()), float(r_full.max())
scale = (rmax - rmin) / 65535.0
q = np.round((r_full - rmin) / scale).astype(np.uint16)
rq = rmin + q.astype(float) * scale
np.save(OUT + 'q_full.npy', q)
json.dump({'rmin': rmin, 'rmax': rmax}, open(OUT + 'q_meta.json', 'w'))

y0, x0, S = 700, 600, 512
r = rq[y0:y0+S, x0:x0+S].copy()
np.save(OUT + 'r_roi.npy', r)

ell_pag_px = 1.0 * 1146 * (1.239841984e-9/17) / (4*np.pi) / (5.8e-6)**2
L = 4*np.pi**2 * ell_pag_px
uL2 = _paganin_(r, L); np.save(OUT + 'uL2.npy', uL2)

def chambolle_gap(r, ell, tau=0.125, eps=1e-3, max_iter=20000):
    px = np.zeros_like(r); py = np.zeros_like(r)
    for k in range(1, max_iter+1):
        gx, gy = _grad(_div(px, py) - r/float(ell))
        qx = px + tau*gx; qy = py + tau*gy
        nq = np.sqrt(qx*qx + qy*qy); n = np.maximum(1.0, nq)
        px = qx/n; py = qy/n
        u = r - ell*_div(px, py)
        Ep = 0.5*np.sum((u-r)**2) + ell*_tv_isotropic(u); Ed = 0.5*np.sum(r**2) - 0.5*np.sum(u**2)
        G = (Ep-Ed)/Ep
        if G < eps: break
    return u, -px, -py, k, G, np.mean(nq > 1)

ELL_TV = 0.1
t = time.time(); u, p0, p1, nit, G, sat = chambolle_gap(r, ELL_TV)
print('chambolle', nit, G, sat, time.time()-t)
np.save(OUT + 'uTV.npy', u); np.save(OUT + 'p0.npy', p0); np.save(OUT + 'p1.npy', p1)
meta = dict(ell_pag_px=ell_pag_px, L=L, ell_tv=ELL_TV, tau=0.125, eps=1e-3, nit=nit, G=G, sat=sat, roi=[y0, x0, S])

meta['D'] = {}
for beta in [1e-1, 1e-2, 1e-3]:
    t = time.time()
    Dp, _, rp = estimate_positive_D_ridge(u, p0, p1, lambda_phys=ell_pag_px, ell_dark=1.0, ell_tv=ELL_TV,
                                          alpha=0, beta=beta, maxiter=1000, D0=1)
    Dt, _, rt = estimate_positive_D_ridge(u, p0, p1, lambda_phys=ell_pag_px, ell_dark=1.0, ell_tv=ELL_TV,
                                          alpha=0, beta=beta, maxiter=30000, D0=1, tol_loss=1e-16, tol_D=1e-12)
    np.save(OUT + f'D_paper_{beta:g}.npy', Dp); np.save(OUT + f'D_tight_{beta:g}.npy', Dt)
    meta['D'][f'{beta:g}'] = dict(paper=dict(msg=str(rp.message), nit=int(rp.nit), fun=float(rp.fun)),
                                  tight=dict(msg=str(rt.message), nit=int(rt.nit), fun=float(rt.fun)))
    print(beta, rp.message, rp.nit, rt.message, rt.nit, rt.fun, 'rel diff paper-vs-tight',
          np.linalg.norm(Dp-Dt)/np.linalg.norm(Dt), f'{time.time()-t:.0f}s', flush=True)
json.dump(meta, open(OUT + 'meta.json', 'w'), indent=1, default=float)
