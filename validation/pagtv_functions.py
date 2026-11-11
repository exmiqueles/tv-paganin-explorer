from scipy.optimize import minimize
from scipy.fft import fft2, ifft2

def pad(I, pad):
    return np.pad(
        I,
        pad_width=((pad, pad), (pad, pad)),
        mode='edge'
    )


def unpad(I_padded, pad):
    if pad == 0:
        return I_padded
    return I_padded[pad:-pad, pad:-pad]


def _paganin_(r, L):

    _PAD_ = 100

    r = pad(r, _PAD_)

    Ny, Nx = r.shape

    ky = np.fft.fftfreq(Ny, d=1)
    kx = np.fft.fftfreq(Nx, d=1)

    k2 = np.add.outer(
        ky**2,
        kx**2
    )

    kernel = 1.0 / (1.0 + L * k2)

    r_hat = fft2(r)

    filtered_hat = kernel * r_hat

    u = ifft2(filtered_hat)

    p = unpad(
        u.real,
        _PAD_
    )

    return p


def _grad(u):
    """
    Gradiente (forward) com BC de Neumann (derivada = 0 na borda).
    Retorna p = (px, py) com mesmo shape de u.
    """
    px = np.zeros_like(u)
    py = np.zeros_like(u)
    px[:-1, :] = u[1:, :] - u[:-1, :]
    py[:, :-1] = u[:, 1:] - u[:, :-1]
    return px, py


def _div(px, py):
    """
    Divergente (adjunto do gradiente acima): backward com Neumann.
    """
    vx = np.zeros_like(px)
    vy = np.zeros_like(py)

    vx[0, :] = px[0, :]
    vx[1:-1, :] = px[1:-1, :] - px[0:-2, :]
    vx[-1, :] = -px[-2, :]

    vy[:, 0] = py[:, 0]
    vy[:, 1:-1] = py[:, 1:-1] - py[:, 0:-2]
    vy[:, -1] = -py[:, -2]

    return vx + vy


def _rot(px, py):
    """
    Rotacional
    """
    vx = np.zeros_like(py)
    vy = np.zeros_like(px)

    vx[0, :] = px[0, :]
    vx[1:-1, :] = px[1:-1, :] - px[0:-2, :]
    vx[-1, :] = -px[-2, :]

    vy[:, 0] = py[:, 0]
    vy[:, 1:-1] = py[:, 1:-1] - py[:, 0:-2]
    vy[:, -1] = -py[:, -2]

    return vx - vy


def _tv_isotropic(u):
    px, py = _grad(u)
    return np.sum(np.sqrt(px**2 + py**2))


class EarlyStop(Exception):
    pass


def estimate_positive_D_ridge(
    u, px, py,
    lambda_phys,
    ell_dark,
    ell_tv,
    alpha=0.0,      # penaliza grad(D)
    beta=0.0,       # penaliza amplitude D
    maxiter=1000,
    D0=0.1,
    eps=1e-12,
    D_init=None,
    verbose=False,
    verbose_every=10,
    check_every=10,
    tol_loss=1e-6,
    tol_D=1e-4,
    miniter=50,
):
    """
    Resolve

        min_{D>=0}
            0.5 || ell_tv*p - lambda*grad(u)
                   + ell_dark*grad(Du) ||²
          + 0.5 alpha ||grad(D)||²
          + 0.5 beta  ||D||²

    usando L-BFGS-B.

    Requer:
        normalize99(u)
        _grad(f)
        _div(fx, fy)

    Observação:
        beta > 0 penaliza o modo volumétrico/macro de D.
    """

    u = np.asarray(u, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)

    # ----------------------------------------------------
    # normalizações
    # ----------------------------------------------------

    
    #u, uscale = normalize99(u, eps=eps)
    #
    #pmag = np.sqrt(px**2 + py**2)
    #pscale = np.percentile(pmag, 99) + eps
    #
    #px = px / pscale
    #py = py / pscale

    ux, uy = _grad(u)

    tx = ell_tv * px - lambda_phys * ux
    ty = ell_tv * py - lambda_phys * uy

    H, W = u.shape
    HW = H * W

    def unpack(x):
        return x.reshape(H, W)

    # ----------------------------------------------------
    # funcional
    # ----------------------------------------------------

    def loss_grad(x):

        D = unpack(x)

        w = D * u
        wx, wy = _grad(w)

        rx = tx + ell_dark * wx
        ry = ty + ell_dark * wy

        loss_data = 0.5 * np.mean(rx**2 + ry**2)

        # gradiente do termo data
        gw = -ell_dark * _div(rx, ry) / HW
        gD = gw * u

        loss_reg_grad = 0.0
        loss_reg_amp = 0.0

        # suavidade: 0.5 alpha ||grad D||²
        if alpha != 0.0:
            Dx, Dy = _grad(D)

            loss_reg_grad = 0.5 * alpha * np.mean(
                Dx**2 + Dy**2
            )

            gD += -alpha * _div(Dx, Dy) / HW

        # amplitude/ridge: 0.5 beta ||D||²
        if beta != 0.0:
            loss_reg_amp = 0.5 * beta * np.mean(D**2)

            gD += beta * D / HW

        loss = loss_data + loss_reg_grad + loss_reg_amp

        return loss, gD.ravel()

    # ----------------------------------------------------
    # chute inicial
    # ----------------------------------------------------

    if D_init is None:
        x0 = np.full((H, W), D0, dtype=float)
    else:
        x0 = np.asarray(D_init, dtype=float)

        if x0.shape != (H, W):
            raise ValueError(
                f"D_init shape {x0.shape} != {(H, W)}"
            )

        x0 = np.maximum(x0, 0.0)

    # ----------------------------------------------------
    # callback
    # ----------------------------------------------------

    state = {
        "iter": 0,
        "prev_loss": None,
        "prev_D": None,
        "last_x": x0.ravel().copy(),
        "stop_reason": None,
    }

    def callback(xk):

        state["iter"] += 1
        state["last_x"] = xk.copy()

        do_print = (
            verbose
            and state["iter"] % verbose_every == 0
        )

        do_check = (
            state["iter"] % check_every == 0
        )

        if not (do_print or do_check):
            return

        loss, _ = loss_grad(xk)
        Dcur = unpack(xk)

        rel_loss = np.nan
        rel_D = np.nan

        if state["prev_loss"] is not None:
            rel_loss = (
                abs(loss - state["prev_loss"])
                /
                (abs(state["prev_loss"]) + eps)
            )

        if state["prev_D"] is not None:
            rel_D = (
                np.linalg.norm(Dcur - state["prev_D"])
                /
                (np.linalg.norm(state["prev_D"]) + eps)
            )

        state["prev_loss"] = loss
        state["prev_D"] = Dcur.copy()

        if do_print:
            print(
                f"[{state['iter']:5d}] "
                f"loss={loss:.6e} "
                f"rel_loss={rel_loss:.3e} "
                f"rel_D={rel_D:.3e} "
                f"Dmean={Dcur.mean():.4e} "
                f"Dmax={Dcur.max():.4e} "
                f"Dp95={np.percentile(Dcur,95):.4e}"
            )

        if (
            state["iter"] >= miniter
            and np.isfinite(rel_D)
            and np.isfinite(rel_loss)
        ):
            if rel_D < tol_D:
                state["stop_reason"] = f"rel_D={rel_D:.3e}"
                raise EarlyStop

            if rel_loss < tol_loss:
                state["stop_reason"] = f"rel_loss={rel_loss:.3e}"
                raise EarlyStop

    # ----------------------------------------------------
    # otimização
    # ----------------------------------------------------

    try:
        res = minimize(
            fun=lambda x: loss_grad(x),
            x0=x0.ravel(),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * (H * W),
            callback=callback,
            options={
                "maxiter": maxiter,
                "ftol": 1e-12,
                "gtol": 1e-12,
                "maxls": 50,
            }
        )

        x_final = res.x

    except EarlyStop:
        x_final = state["last_x"]

        class Result:
            pass

        res = Result()
        res.x = x_final
        res.success = True
        res.message = "early stop: " + state["stop_reason"]
        res.nit = state["iter"]
        res.fun = loss_grad(x_final)[0]

    # ----------------------------------------------------
    # saída
    # ----------------------------------------------------

    D = unpack(x_final)
    w = D * u

    if verbose:
        print()
        print("success :", res.success)
        print("message :", res.message)
        print("nit     :", res.nit)
        print("final F :", res.fun)

    return D, w, res
