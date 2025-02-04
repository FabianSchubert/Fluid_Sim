import numpy as np


def solve_incompr(u, v, mask_u, mask_v, inv_sum_mask, n_it=20):

    for i in range(n_it):

        g = 1.95 if i < (n_it - 1) else 1.0

        D = (u - np.roll(u, -1, axis=1) + v - np.roll(v, -1, axis=0)) * inv_sum_mask * g

        offs = i % 2

        np.subtract(u[offs::2, ::2], D[offs::2, ::2], out=u[offs::2, ::2])
        np.subtract(
            u[(1 - offs) :: 2, 1::2],
            D[(1 - offs) :: 2, 1::2],
            out=u[(1 - offs) :: 2, 1::2],
        )

        np.subtract(v[offs::2, ::2], D[offs::2, ::2], out=v[offs::2, ::2])
        np.subtract(
            v[(1 - offs) :: 2, 1::2],
            D[(1 - offs) :: 2, 1::2],
            out=v[(1 - offs) :: 2, 1::2],
        )

        np.subtract(u, (D - np.roll(D, 1, axis=1)) * g, out=u)
        np.subtract(v, (D - np.roll(D, 1, axis=0)) * g, out=v)

        np.multiply(u, mask_u, out=u)
        np.multiply(v, mask_v, out=v)


def lapl(f, lp):

    np.add(np.roll(f, 1, axis=0), np.roll(f, -1, axis=0), out=lp)
    np.add(lp, np.roll(f, 1, axis=1), out=lp)
    np.add(lp, np.roll(f, -1, axis=1), out=lp)
    np.subtract(lp, 4 * f, out=lp)
