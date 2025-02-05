import cupy as xp
from numba import cuda

BLOCK_SIZE_X, BLOCK_SIZE_Y = 8, 8

incompr_it_kern_code = r"""
extern "C" __global__
void incompr_it(
    const unsigned int i, 
    float* u, float* v,
    const float* div,
    const float* mask_u, const float* mask_v,
    const float* inv_sum_mask,
    float g, const unsigned int n_it, const int width, const int height){

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const unsigned int idx_00 = y * width + x;
    const unsigned int idx_n0 = ((y - 1 + height) % height) * width + x;
    const unsigned int idx_0n = y * width + ((x - 1 + width) % width);

    const float _g = (i < (n_it - 1)) ? g : 1.0;
    const float _mask = (x + y + i) % 2;
    const float _mask_inv = 1.0 - _mask;

    u[idx_00] += (_mask_inv * div[idx_0n] * inv_sum_mask[idx_0n] - _mask * div[idx_00] * inv_sum_mask[idx_00]) * _g;
    v[idx_00] += (_mask_inv * div[idx_n0] * inv_sum_mask[idx_0n] - _mask * div[idx_00] * inv_sum_mask[idx_00]) * _g;

    u[idx_00] *= mask_u[idx_00];
    v[idx_00] *= mask_v[idx_00];
    
}
"""

calc_div_kern_code = r"""
extern "C" __global__
void calc_div(
    const float* u, const float* v,
    float* div,
    const int width, const int height){

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const unsigned int idx_00 = y * width + x;
    const unsigned int idx_p0 = ((y + 1) % height) * width + x;
    const unsigned int idx_0p = y * width + ((x + 1) % width);
    
    div[idx_00] = (u[idx_00] + v[idx_00] - u[idx_0p] - v[idx_p0]);
}
"""

incompr_it_kernel = xp.RawKernel(incompr_it_kern_code, "incompr_it")
calc_div_kernel = xp.RawKernel(calc_div_kern_code, "calc_div")


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
    grid_size = (
        (width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X,
        (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y,
    )

    block_size = (BLOCK_SIZE_X, BLOCK_SIZE_Y)

    for i in range(n_it):
        calc_div_kernel(grid_size, block_size, (u, v, d, width, height))
        incompr_it_kernel(
            grid_size,
            block_size,
            (i, u, v, d, mask_u, mask_v, inv_sum_mask, g, n_it, width, height),
        )


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
