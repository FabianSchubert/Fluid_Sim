import numpy as xp


def solve_incompr(
    u: xp.ndarray,
    v: xp.ndarray,
    d: xp.ndarray,
    mask_u: xp.ndarray,
    mask_v: xp.ndarray,
    inv_sum_mask: xp.ndarray,
    n_it: int,
    g: xp.float32,
    width: int,
    height: int,
):

    for i in range(n_it):

        _g = g if i < (n_it - 1) else xp.float32(1.0)

        xp.add(u, v, out=d)
        xp.subtract(d[:, :-1], u[:, 1:], out=d[:, :-1])
        xp.subtract(d[:-1, :], v[1:, :], out=d[:-1, :])
        xp.subtract(d[:, -1], u[:, 0], out=d[:, -1])
        xp.subtract(d[-1, :], v[0, :], out=d[-1, :])

        xp.multiply(d, inv_sum_mask, out=d)
        xp.multiply(d, _g, out=d)

        _ch_ind = i % 2
        d[_ch_ind::2, ::2] = 0.0
        d[(1 - _ch_ind) :: 2, 1::2] = 0.0

        xp.add(u[:, 1:], d[:, :-1], out=u[:, 1:])
        xp.add(u[:, 0], d[:, -1], out=u[:, 0])

        xp.add(v[1:, :], d[:-1, :], out=v[1:, :])
        xp.add(v[0, :], d[-1, :], out=v[0, :])

        xp.subtract(u, d, out=u)
        xp.subtract(v, d, out=v)

        xp.multiply(u, mask_u, out=u)
        xp.multiply(v, mask_v, out=v)


def lapl(f: xp.ndarray, lp: xp.ndarray) -> None:

    xp.add(xp.roll(f, 1, axis=0), xp.roll(f, -1, axis=0), out=lp)
    xp.add(lp, xp.roll(f, 1, axis=1), out=lp)
    xp.add(lp, xp.roll(f, -1, axis=1), out=lp)
    xp.subtract(lp, 4.0 * f, out=lp)


def visc_step(
    f: xp.ndarray, f_l: xp.ndarray, f_n: xp.ndarray, dt: xp.float32, visc: xp.float32
) -> None:
    lapl(f, f_l)
    xp.multiply(f_l, dt * visc, out=f_l)
    xp.add(f, f_l, out=f_n)


def bil_filt(
    f: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    f_n: xp.ndarray,
    width: int,
    height: int,
) -> None:
    x0 = x.astype(xp.int32)
    x1 = (x + 1).astype(xp.int32)
    y0 = y.astype(xp.int32)
    y1 = (y + 1).astype(xp.int32)

    dx = x - x0
    dy = y - y0

    f_00 = f[y0 % height, x0 % width]
    f_10 = f[y0 % height, x1 % width]
    f_01 = f[y1 % height, x0 % width]
    f_11 = f[y1 % height, x1 % width]

    f_n[:] = ((1.0 - dx) * f_00 + dx * f_10) * (1.0 - dy) + (
        (1.0 - dx) * f_01 + dx * f_11
    ) * dy


def semi_lag_step(
    f: xp.ndarray,
    u: xp.ndarray,
    v: xp.ndarray,
    f_n: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    dt: xp.float32,
    width: int,
    height: int,
) -> None:

    bil_filt(f, x - dt * u, y - dt * v, f_n, width, height)


def calc_vel_center(u: xp.ndarray, v: xp.ndarray, uc: xp.ndarray, vc: xp.ndarray):

    xp.add(u[:, :-1], u[:, 1:], out=uc[:, :-1])
    xp.add(u[:, -1], u[:, 0], out=uc[:, -1])
    xp.add(v[:-1, :], v[1:, :], out=vc[:-1, :])
    xp.add(v[-1, :], v[0, :], out=vc[-1, :])

    xp.multiply(uc, 0.5, out=uc)
    xp.multiply(vc, 0.5, out=vc)


def calc_uv(u: xp.ndarray, uv: xp.ndarray):
    xp.add(u[1:, :-1], u[1:, 1:], out=uv[1:, :-1])
    xp.add(uv[1:, :-1], u[:-1, :-1], out=uv[1:, :-1])
    xp.add(uv[1:, :-1], u[:-1, 1:], out=uv[1:, :-1])

    xp.add(u[0, :-1], u[0, 1:], out=uv[0, :-1])
    xp.add(uv[0, :-1], u[-1, :-1], out=uv[0, :-1])
    xp.add(uv[0, :-1], u[-1, 1:], out=uv[0, :-1])

    xp.add(u[1:, -1], u[1:, 0], out=uv[1:, -1])
    xp.add(uv[1:, -1], u[:-1, -1], out=uv[1:, -1])
    xp.add(uv[1:, -1], u[:-1, 0], out=uv[1:, -1])

    xp.add(u[0, -1], u[0, 0], out=uv[:1, -1:])
    xp.add(uv[0, -1], u[-1, -1], out=uv[:1, -1:])
    xp.add(uv[0, -1], u[-1, 0], out=uv[:1, -1:])

    xp.multiply(uv, 0.25, out=uv)


def calc_vu(v: xp.ndarray, vu: xp.ndarray):
    xp.add(v[:-1, 1:], v[1:, 1:], out=vu[:-1, 1:])
    xp.add(vu[:-1, 1:], v[:-1, :-1], out=vu[:-1, 1:])
    xp.add(vu[:-1, 1:], v[1:, :-1], out=vu[:-1, 1:])

    xp.add(v[:-1, 0], v[1:, 0], out=vu[:-1, 0])
    xp.add(vu[:-1, 0], v[:-1, -1], out=vu[:-1, 0])
    xp.add(vu[:-1, 0], v[:-1, -1], out=vu[:-1, 0])

    xp.add(v[-1, 0], v[0, 0], out=vu[-1:, :1])
    xp.add(vu[-1:, :1], v[-1, -1], out=vu[-1:, :1])
    xp.add(vu[-1:, :1], v[0, -1], out=vu[-1:, :1])

    xp.multiply(vu, 0.25, out=vu)
