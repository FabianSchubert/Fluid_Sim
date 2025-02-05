from importlib.util import find_spec

from enum import Enum

import numpy as np

from .backends.numpy import operators as numpy_ops

CUPY_INSTALLED = find_spec("cupy") is not None
if CUPY_INSTALLED:
    import cupy as cp

    CUDA_GPU_AVAILABLE = cp.is_available()
else:
    CUDA_GPU_AVAILABLE = False

if CUDA_GPU_AVAILABLE:
    from .backends.cupy import operators as cupy_ops
else:
    print("Cupy not installed or no CUDA GPU available. Falling back to CPU.")


BACKEND = Enum("Backend", {"CPU": np, "GPU": (cp if CUDA_GPU_AVAILABLE else None)})

DEFAULT_PARAMS = {
    "dt": 0.3,
    "gs_it": 20,
    "gs_overrelax": 1.95,
    "visc": 0.0,
    "n_part": 5000,
}


class Simulator:

    def __init__(
        self,
        dimensions: tuple[int, int],
        sim_params: dict | None = None,
        backend: str = "GPU",
    ) -> None:
        assert len(dimensions) == 2, "dimensions parameter expects length 2 tuple."
        self.WIDTH = dimensions[0]
        self.HEIGHT = dimensions[1]

        backend = backend if CUDA_GPU_AVAILABLE else "CPU"

        self.backend = BACKEND[backend]

        self.prec = self.xp.float32

        if self.backend == BACKEND.GPU:
            self.calc_uv = cupy_ops.calc_uv
            self.calc_vu = cupy_ops.calc_vu
            self.calc_vel_center = cupy_ops.calc_vel_center
            self.semi_lag_step = cupy_ops.semi_lag_step
            self.solve_incompr = cupy_ops.solve_incompr  # cupy_ops.solve_incompr
            self.visc_step = cupy_ops.visc_step
            self.bil_filt = cupy_ops.bil_filt
        else:
            self.calc_uv = numpy_ops.calc_uv
            self.calc_vu = numpy_ops.calc_vu
            self.calc_vel_center = numpy_ops.calc_vel_center
            self.semi_lag_step = numpy_ops.semi_lag_step
            self.solve_incompr = numpy_ops.solve_incompr
            self.visc_step = numpy_ops.visc_step
            self.bil_filt = numpy_ops.bil_filt

        self.SIM_PARAMS = DEFAULT_PARAMS | (
            sim_params if isinstance(sim_params, dict) else {}
        )

        _param_excl = set(self.SIM_PARAMS.keys()) - set(DEFAULT_PARAMS.keys())

        assert len(_param_excl) == 0, f"Invalid sim. parameters {_param_excl}."

        self.DT = self.prec(self.SIM_PARAMS["dt"])
        self.GS_IT = int(self.SIM_PARAMS["gs_it"])
        self.GS_OVERRELAX = self.prec(self.SIM_PARAMS["gs_overrelax"])
        self.VISC = self.prec(self.SIM_PARAMS["visc"])
        self.N_PART = int(self.SIM_PARAMS["n_part"])

        self.X, self.Y = self.xp.meshgrid(
            self.xp.arange(self.WIDTH), self.xp.arange(self.HEIGHT)
        )

        self._u = self.xp.zeros((2, self.HEIGHT, self.WIDTH), dtype=self.prec)
        self._v = self.xp.zeros((2, self.HEIGHT, self.WIDTH), dtype=self.prec)

        self.d = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)

        self.ul = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)
        self.vl = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)

        self.uc = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)
        self.vc = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)

        self.uv = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)
        self.vu = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)

        self.D = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)

        # only needed for cpu backend
        self.chb_mask = self.xp.zeros((2, self.HEIGHT, self.WIDTH), dtype=self.prec)
        self.chb_mask[0] = ((self.X + self.Y) % 2).astype(self.prec)
        self.chb_mask[1] = ((self.X + self.Y + 1) % 2).astype(self.prec)

        self.solid_mask = self.xp.zeros((self.HEIGHT, self.WIDTH), dtype=self.prec)

        self.mask_u, self.mask_v = self.xp.ones(
            (self.HEIGHT, self.WIDTH), dtype=self.prec
        ), self.xp.ones((self.HEIGHT, self.WIDTH), dtype=self.prec)

        self.inv_sum_mask = 4.0 * self.xp.ones(
            (self.HEIGHT, self.WIDTH), dtype=self.prec
        )

        self.update_masks()

        self.flop = 0

        self._pos_part = self.xp.random.rand(self.N_PART, 2)
        self._pos_part[:, 0] *= self.WIDTH
        self._pos_part[:, 1] *= self.HEIGHT
        self._u_part = self.xp.zeros((self.N_PART))
        self._v_part = self.xp.zeros((self.N_PART))

    def step(self):
        i = self.flop
        i_n = 1 - self.flop

        self.calc_uv(self._u[i], self.uv)
        self.calc_vu(self._v[i], self.vu)

        self.semi_lag_step(
            self._u[i],
            self._u[i],
            self.vu,
            self._u[i_n],
            self.X,
            self.Y,
            self.DT,
            self.WIDTH,
            self.HEIGHT,
        )
        self.semi_lag_step(
            self._v[i],
            self.uv,
            self._v[i],
            self._v[i_n],
            self.X,
            self.Y,
            self.DT,
            self.WIDTH,
            self.HEIGHT,
        )

        self.visc_step(self._u[i_n], self.ul, self._u[i_n], self.DT, self.VISC)
        self.visc_step(self._v[i_n], self.vl, self._v[i_n], self.DT, self.VISC)

        self.solve_incompr(
            self._u[i_n],
            self._v[i_n],
            self.d,
            self.mask_u,
            self.mask_v,
            self.inv_sum_mask,
            # self.chb_mask,
            self.GS_IT,
            self.GS_OVERRELAX,
            self.WIDTH,
            self.HEIGHT,
        )

        self.calc_vel_center(self._u[i_n], self._v[i_n], self.uc, self.vc)

        self.update_particles(self._u[i_n], self._v[i_n])

        self.flop = 1 - self.flop

    def update_particles(self, u, v):
        self.bil_filt(
            u,
            self._pos_part[:, 0],
            self._pos_part[:, 1],
            self._u_part,
            self.WIDTH,
            self.HEIGHT,
        )
        self.bil_filt(
            v,
            self._pos_part[:, 0],
            self._pos_part[:, 1],
            self._v_part,
            self.WIDTH,
            self.HEIGHT,
        )

        self._pos_part[:, 0] += self.DT * self._u_part
        self._pos_part[:, 1] += self.DT * self._v_part

        self._pos_part[:, 0] = self._pos_part[:, 0] % self.WIDTH
        self._pos_part[:, 1] = self._pos_part[:, 1] % self.HEIGHT

    def update_masks(self):

        self.mask_u = (
            1.0
            - self.xp.clip(
                self.solid_mask + self.xp.roll(self.solid_mask, 1, axis=1), 0.0, 1.0
            )
        ).astype(self.prec)

        self.mask_v = (
            1.0
            - self.xp.clip(
                self.solid_mask + self.xp.roll(self.solid_mask, 1, axis=0), 0.0, 1.0
            )
        ).astype(self.prec)

        _nb_sum = (
            self.mask_u
            + self.xp.roll(self.mask_u, -1, axis=1)
            + self.mask_v
            + self.xp.roll(self.mask_v, -1, axis=0)
        ).astype(self.prec)

        self.inv_sum_mask = self.xp.divide(
            1.0, _nb_sum  # , out=self.xp.zeros_like(_nb_sum), where=_nb_sum != 0
        ).astype(self.prec)

        self.inv_sum_mask[_nb_sum == 0] = 0.0

        if self.backend == BACKEND.CPU:

            def to_host(x):
                return x.copy()

            def to_device(x):
                return x.copy()

        else:

            def to_host(x):
                return self.xp.asnumpy(x)

            def to_device(x):
                return self.xp.asarray(x)

        self.to_host = to_host
        self.to_device = to_device

    @property
    def u(self):
        return self.to_host(self._u[self.flop])

    @u.setter
    def u(self, x):
        self._u[self.flop] = self.to_device(x)

    @property
    def v(self):
        return self.to_host(self._v[self.flop])

    @v.setter
    def v(self, x):
        self._v[self.flop] = self.to_device(x)

    @property
    def pos_part(self):
        return self.to_host(self._pos_part)

    @pos_part.setter
    def pos_part(self, x):
        self._pos_part[:] = self.to_device(x)

    @property
    def u_part(self):
        return self.to_host(self._u_part)

    @u_part.setter
    def u_part(self, x):
        self._u_part[:] = self.to_device(x)

    @property
    def backend(self):
        return BACKEND["CPU"] if self.xp is np else BACKEND["GPU"]

    @backend.setter
    def backend(self, be: BACKEND):
        assert isinstance(be, BACKEND)
        self.xp = be.value

    """
    def solve_incompr(self, u, v):  # , mask_u, mask_v, inv_sum_mask):

        for i in range(self.GS_IT):

            g = self.prec(1.95 if i < (self.GS_IT - 1) else 1.0)

            self.xp.add(u, v, out=self.D)
            self.xp.subtract(self.D[:, :-1], u[:, 1:], out=self.D[:, :-1])
            self.xp.subtract(self.D[:-1, :], v[1:, :], out=self.D[:-1, :])
            self.xp.subtract(self.D[:, -1], u[:, 0], out=self.D[:, -1])
            self.xp.subtract(self.D[-1, :], v[0, :], out=self.D[-1, :])

            self.xp.multiply(self.D, self.inv_sum_mask, out=self.D)
            self.xp.multiply(self.D, self.chb_mask[i % 2], out=self.D)
            self.xp.multiply(self.D, g, out=self.D)

            self.xp.add(u[:, 1:], self.D[:, :-1], out=u[:, 1:])
            self.xp.add(u[:, 0], self.D[:, -1], out=u[:, 0])

            self.xp.add(v[1:, :], self.D[:-1, :], out=v[1:, :])
            self.xp.add(v[0, :], self.D[-1, :], out=v[0, :])

            self.xp.subtract(u, self.D, out=u)
            self.xp.subtract(v, self.D, out=v)

            self.xp.multiply(u, self.mask_u, out=u)
            self.xp.multiply(v, self.mask_v, out=v)

    def visc_step(self, f, f_l, f_n):
        self.lapl(f, f_l)
        self.xp.multiply(f_l, self.DT * self.VISC, out=f_l)
        self.xp.add(f, f_l, out=f_n)

    def lapl(self, f, lp):

        self.xp.add(
            self.xp.roll(f, 1, axis=0), self.xp.roll(f, -1, axis=0), out=lp
        )
        self.xp.add(lp, self.xp.roll(f, 1, axis=1), out=lp)
        self.xp.add(lp, self.xp.roll(f, -1, axis=1), out=lp)
        self.xp.subtract(lp, 4.0 * f, out=lp)

    def bil_filt(self, f, x, y, f_n):
        x0 = x.astype(int)
        x1 = (x + 1).astype(int)
        y0 = y.astype(int)
        y1 = (y + 1).astype(int)

        dx = x - x0
        dy = y - y0

        f_00 = f[y0 % self.HEIGHT, x0 % self.WIDTH]
        f_10 = f[y0 % self.HEIGHT, x1 % self.WIDTH]
        f_01 = f[y1 % self.HEIGHT, x0 % self.WIDTH]
        f_11 = f[y1 % self.HEIGHT, x1 % self.WIDTH]

        f_n[:] = ((1.0 - dx) * f_00 + dx * f_10) * (1.0 - dy) + (
            (1.0 - dx) * f_01 + dx * f_11
        ) * dy

    def semi_lag_step(self, f, u, v, f_n):

        self.bil_filt(f, self.X - self.DT * u, self.Y - self.DT * v, f_n)

    def calc_vel_center(self, u, v, uc, vc):

        self.xp.add(u[:, :-1], u[:, 1:], out=uc[:, :-1])
        self.xp.add(u[:, -1], u[:, 0], out=uc[:, -1])
        self.xp.add(v[:-1, :], v[1:, :], out=vc[:-1, :])
        self.xp.add(v[-1, :], v[0, :], out=vc[-1, :])

        self.xp.multiply(uc, 0.5, out=uc)
        self.xp.multiply(vc, 0.5, out=vc)

    def calc_uv(self, u, uv):
        self.xp.add(u[1:, :-1], u[1:, 1:], out=uv[1:, :-1])
        self.xp.add(uv[1:, :-1], u[:-1, :-1], out=uv[1:, :-1])
        self.xp.add(uv[1:, :-1], u[:-1, 1:], out=uv[1:, :-1])

        self.xp.add(u[0, :-1], u[0, 1:], out=uv[0, :-1])
        self.xp.add(uv[0, :-1], u[-1, :-1], out=uv[0, :-1])
        self.xp.add(uv[0, :-1], u[-1, 1:], out=uv[0, :-1])

        self.xp.add(u[1:, -1], u[1:, 0], out=uv[1:, -1])
        self.xp.add(uv[1:, -1], u[:-1, -1], out=uv[1:, -1])
        self.xp.add(uv[1:, -1], u[:-1, 0], out=uv[1:, -1])

        self.xp.add(u[0, -1], u[0, 0], out=uv[:1, -1:])
        self.xp.add(uv[0, -1], u[-1, -1], out=uv[:1, -1:])
        self.xp.add(uv[0, -1], u[-1, 0], out=uv[:1, -1:])

        self.xp.multiply(uv, 0.25, out=uv)

    def calc_vu(self, v, vu):
        self.xp.add(v[:-1, 1:], v[1:, 1:], out=vu[:-1, 1:])
        self.xp.add(vu[:-1, 1:], v[:-1, :-1], out=vu[:-1, 1:])
        self.xp.add(vu[:-1, 1:], v[1:, :-1], out=vu[:-1, 1:])

        self.xp.add(v[:-1, 0], v[1:, 0], out=vu[:-1, 0])
        self.xp.add(vu[:-1, 0], v[:-1, -1], out=vu[:-1, 0])
        self.xp.add(vu[:-1, 0], v[:-1, -1], out=vu[:-1, 0])

        self.xp.add(v[-1, 0], v[0, 0], out=vu[-1:, :1])
        self.xp.add(vu[-1:, :1], v[-1, -1], out=vu[-1:, :1])
        self.xp.add(vu[-1:, :1], v[0, -1], out=vu[-1:, :1])

        self.xp.multiply(vu, 0.25, out=vu)
    """
