import numpy as np
import cupy as cp

from enum import Enum

BACKEND = Enum("Backend", {"CPU": np, "GPU": cp})

PREC = Enum("Prec", ["f64", "f32", "f16"])

DEFAULT_PARAMS = {
    "dt": 0.3,
    "gs_it": 20,
    "visc": 0.0,
    "n_part": 5000,
}


class Simulator:

    def __init__(
        self,
        dimensions: tuple[int, int],
        sim_params: dict | None = None,
        backend: str = "CPU",
        precision: str = "f32",
    ) -> None:
        assert len(dimensions) == 2, "dimensions parameter expects length 2 tuple."
        self.WIDTH = dimensions[0]
        self.HEIGHT = dimensions[1]

        self.backend = BACKEND[backend]

        self.prec = PREC[precision]

        self.SIM_PARAMS = DEFAULT_PARAMS | (
            sim_params if isinstance(sim_params, dict) else {}
        )

        _param_excl = set(self.SIM_PARAMS.keys()) - set(DEFAULT_PARAMS.keys())

        assert len(_param_excl) == 0, f"Invalid sim. parameters {_param_excl}."

        self.DT = self._prec(self.SIM_PARAMS["dt"])
        self.GS_IT = int(self.SIM_PARAMS["gs_it"])
        self.VISC = self._prec(self.SIM_PARAMS["visc"])
        self.N_PART = int(self.SIM_PARAMS["n_part"])

        self.X, self.Y = self._num.meshgrid(
            self._num.arange(self.WIDTH), self._num.arange(self.HEIGHT)
        )

        self._u = self._num.zeros((2, self.HEIGHT, self.WIDTH), dtype=self._prec)
        self._v = self._num.zeros((2, self.HEIGHT, self.WIDTH), dtype=self._prec)

        self.ul = self._num.zeros((self.HEIGHT, self.WIDTH), dtype=self._prec)
        self.vl = self._num.zeros((self.HEIGHT, self.WIDTH), dtype=self._prec)

        self.uc = self._num.zeros((self.HEIGHT, self.WIDTH), dtype=self._prec)
        self.vc = self._num.zeros((self.HEIGHT, self.WIDTH), dtype=self._prec)

        self.uv = self._num.zeros((self.HEIGHT, self.WIDTH), dtype=self._prec)
        self.vu = self._num.zeros((self.HEIGHT, self.WIDTH), dtype=self._prec)

        self.D = self._num.zeros((self.HEIGHT, self.WIDTH), dtype=self._prec)

        self.chb_mask = self._num.zeros((2, self.HEIGHT, self.WIDTH), dtype=self._prec)
        self.chb_mask[0] = ((self.X + self.Y) % 2).astype(self._prec)
        self.chb_mask[1] = ((self.X + self.Y + 1) % 2).astype(self._prec)

        self.solid_mask = self._num.zeros((self.HEIGHT, self.WIDTH), dtype=self._prec)

        self.mask_u, self.mask_v = self._num.ones(
            (self.HEIGHT, self.WIDTH), dtype=self._prec
        ), self._num.ones((self.HEIGHT, self.WIDTH), dtype=self._prec)

        self.inv_sum_mask = 4.0 * self._num.ones(
            (self.HEIGHT, self.WIDTH), dtype=self._prec
        )

        self.update_masks()

        self.flop = 0

        self._pos_part = self._num.random.rand(self.N_PART, 2)
        self._pos_part[:, 0] *= self.WIDTH
        self._pos_part[:, 1] *= self.HEIGHT
        self._u_part = self._num.zeros((self.N_PART))
        self._v_part = self._num.zeros((self.N_PART))

    def step(self):
        i = self.flop
        i_n = 1 - self.flop

        self.calc_uv(self._u[i], self.uv)
        self.calc_vu(self._v[i], self.vu)

        self.semi_lag_step(self._u[i], self._u[i], self.vu, self._u[i_n])
        self.semi_lag_step(self._v[i], self.uv, self._v[i], self._v[i_n])

        self.visc_step(self._u[i_n], self.ul, self._u[i_n])
        self.visc_step(self._v[i_n], self.vl, self._v[i_n])

        self.solve_incompr(self._u[i_n], self._v[i_n])

        self.calc_vel_center(self._u[i_n], self._v[i_n], self.uc, self.uv)

        self.update_particles(self._u[i_n], self._v[i_n])

        self.flop = 1 - self.flop

    def update_particles(self, u, v):
        self.bil_filt(u, self._pos_part[:, 0], self._pos_part[:, 1], self._u_part)
        self.bil_filt(v, self._pos_part[:, 0], self._pos_part[:, 1], self._v_part)

        self._pos_part[:, 0] += self.DT * self._u_part
        self._pos_part[:, 1] += self.DT * self._v_part

        self._pos_part[:, 0] = self._pos_part[:, 0] % self.WIDTH
        self._pos_part[:, 1] = self._pos_part[:, 1] % self.HEIGHT

    def update_masks(self):

        self.mask_u = (
            1.0
            - self._num.clip(
                self.solid_mask + self._num.roll(self.solid_mask, 1, axis=1), 0.0, 1.0
            )
        ).astype(self._prec)

        self.mask_v = (
            1.0
            - self._num.clip(
                self.solid_mask + self._num.roll(self.solid_mask, 1, axis=0), 0.0, 1.0
            )
        ).astype(self._prec)

        _nb_sum = (
            self.mask_u
            + self._num.roll(self.mask_u, -1, axis=1)
            + self.mask_v
            + self._num.roll(self.mask_v, -1, axis=0)
        ).astype(self._prec)

        self.inv_sum_mask = self._num.divide(
            1.0, _nb_sum  # , out=self._num.zeros_like(_nb_sum), where=_nb_sum != 0
        ).astype(self._prec)

        self.inv_sum_mask[_nb_sum == 0] = 0.0

        if self.backend == BACKEND.CPU:

            def to_host(x):
                return x.copy()

            def to_device(x):
                return x.copy()

        else:

            def to_host(x):
                return self._num.asnumpy(x)

            def to_device(x):
                return self._num.asarray(x)

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
        return BACKEND["CPU"] if self._num is np else BACKEND["GPU"]

    @backend.setter
    def backend(self, be: BACKEND):
        assert isinstance(be, BACKEND)
        self._num = be.value

    @property
    def prec(self):
        if self._prec == self._num.float64:
            return PREC["f64"]
        elif self._prec == self._num.float32:
            return PREC["f32"]
        return PREC["f16"]

    @prec.setter
    def prec(self, p: PREC):
        assert isinstance(p, PREC)
        if p == PREC.f64:
            self._prec = self._num.float64
        elif p == PREC.f32:
            self._prec = self._num.float32
        else:
            self._prec = self._num.float16

    def solve_incompr(self, u, v):  # , mask_u, mask_v, inv_sum_mask):

        for i in range(self.GS_IT):

            g = self._prec(1.95 if i < (self.GS_IT - 1) else 1.0)

            self._num.add(u, v, out=self.D)
            self._num.subtract(self.D[:, :-1], u[:, 1:], out=self.D[:, :-1])
            self._num.subtract(self.D[:-1, :], v[1:, :], out=self.D[:-1, :])
            self._num.subtract(self.D[:, -1], u[:, 0], out=self.D[:, -1])
            self._num.subtract(self.D[-1, :], v[0, :], out=self.D[-1, :])

            self._num.multiply(self.D, self.inv_sum_mask, out=self.D)
            self._num.multiply(self.D, self.chb_mask[i % 2], out=self.D)
            self._num.multiply(self.D, g, out=self.D)

            self._num.add(u[:, 1:], self.D[:, :-1], out=u[:, 1:])
            self._num.add(u[:, 0], self.D[:, -1], out=u[:, 0])

            self._num.add(v[1:, :], self.D[:-1, :], out=v[1:, :])
            self._num.add(v[0, :], self.D[-1, :], out=v[0, :])

            self._num.subtract(u, self.D, out=u)
            self._num.subtract(v, self.D, out=v)

            self._num.multiply(u, self.mask_u, out=u)
            self._num.multiply(v, self.mask_v, out=v)

    def visc_step(self, f, f_l, f_n):
        self.lapl(f, f_l)
        self._num.multiply(f_l, self.DT * self.VISC, out=f_l)
        self._num.add(f, f_l, out=f_n)

    def lapl(self, f, lp):

        self._num.add(
            self._num.roll(f, 1, axis=0), self._num.roll(f, -1, axis=0), out=lp
        )
        self._num.add(lp, self._num.roll(f, 1, axis=1), out=lp)
        self._num.add(lp, self._num.roll(f, -1, axis=1), out=lp)
        self._num.subtract(lp, 4.0 * f, out=lp)

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

        self._num.add(u[:, :-1], u[:, 1:], out=uc[:, :-1])
        self._num.add(u[:, -1], u[:, 0], out=uc[:, -1])
        self._num.add(v[:-1, :], v[1:, :], out=vc[:-1, :])
        self._num.add(v[-1, :], v[0, :], out=vc[-1, :])

        self._num.multiply(uc, 0.5, out=uc)
        self._num.multiply(vc, 0.5, out=vc)

    def calc_uv(self, u, uv):
        self._num.add(u[1:, :-1], u[1:, 1:], out=uv[1:, :-1])
        self._num.add(uv[1:, :-1], u[:-1, :-1], out=uv[1:, :-1])
        self._num.add(uv[1:, :-1], u[:-1, 1:], out=uv[1:, :-1])

        self._num.add(u[0, :-1], u[0, 1:], out=uv[0, :-1])
        self._num.add(uv[0, :-1], u[-1, :-1], out=uv[0, :-1])
        self._num.add(uv[0, :-1], u[-1, 1:], out=uv[0, :-1])

        self._num.add(u[1:, -1], u[1:, 0], out=uv[1:, -1])
        self._num.add(uv[1:, -1], u[:-1, -1], out=uv[1:, -1])
        self._num.add(uv[1:, -1], u[:-1, 0], out=uv[1:, -1])

        self._num.add(u[0, -1], u[0, 0], out=uv[:1, -1:])
        self._num.add(uv[0, -1], u[-1, -1], out=uv[:1, -1:])
        self._num.add(uv[0, -1], u[-1, 0], out=uv[:1, -1:])

        self._num.multiply(uv, 0.25, out=uv)

    def calc_vu(self, v, vu):
        self._num.add(v[:-1, 1:], v[1:, 1:], out=vu[:-1, 1:])
        self._num.add(vu[:-1, 1:], v[:-1, :-1], out=vu[:-1, 1:])
        self._num.add(vu[:-1, 1:], v[1:, :-1], out=vu[:-1, 1:])

        self._num.add(v[:-1, 0], v[1:, 0], out=vu[:-1, 0])
        self._num.add(vu[:-1, 0], v[:-1, -1], out=vu[:-1, 0])
        self._num.add(vu[:-1, 0], v[:-1, -1], out=vu[:-1, 0])

        self._num.add(v[-1, 0], v[0, 0], out=vu[-1:, :1])
        self._num.add(vu[-1:, :1], v[-1, -1], out=vu[-1:, :1])
        self._num.add(vu[-1:, :1], v[0, -1], out=vu[-1:, :1])

        self._num.multiply(vu, 0.25, out=vu)
