import numpy as np
import pygame as pg

import cProfile, pstats, io
from pstats import SortKey

from simulator import Simulator

WIDTH, HEIGHT = 500, 350

SCR_WIDTH, SCR_HEIGHT = 500, 350

STRETCH_WIDTH = SCR_WIDTH / WIDTH
STRETCH_HEIGHT = SCR_HEIGHT / HEIGHT

sim = Simulator((WIDTH, HEIGHT), backend="GPU")

pg.init()

clock = pg.time.Clock()

screen = pg.display.set_mode([SCR_WIDTH, SCR_HEIGHT])

plot_surf = pg.Surface((WIDTH, HEIGHT))

mp = None

col_arr = np.zeros((WIDTH, HEIGHT, 3)).astype(np.uint8)

running = True


def hsv_to_rgb(x):
    pass


def col_map(f, c, gain=1.0):
    _gr = (1.0 - np.exp(-f * gain)).T

    c[..., 0] = (255 * _gr * 0.5).astype(
        np.uint8
    )  # (255 * (1.0 - np.exp(-f * gain))).astype(np.uint8).T
    # c[..., 0] = (255 * (1.0 + np.tanh(f)) * 0.5).astype(np.uint8).T
    c[..., 1] = (255 * _gr * 0.75).astype(np.uint8)
    c[..., 2] = (255 * _gr * 1.0).astype(np.uint8)


pr = cProfile.Profile()
pr.enable()

while running:

    for evt in pg.event.get():
        if evt.type == pg.QUIT:
            running = False

    if pg.mouse.get_pressed()[0]:
        if pg.key.get_pressed()[pg.K_LCTRL]:
            _mp = np.array(pg.mouse.get_pos())
            _mp[0] /= STRETCH_WIDTH
            _mp[1] /= STRETCH_HEIGHT

            _mp = _mp.astype(int)

            """
            mask_x[_mp[1], _mp[0]] = 0.0
            mask_x[_mp[1], (_mp[0] + 1) % WIDTH] = 0.0
            mask_y[_mp[1], _mp[0]] = 0.0
            mask_y[(_mp[1] + 1) % HEIGHT, _mp[0]] = 0.0

            sum_mask = (
                mask_x
                + np.roll(mask_x, -1, axis=1)
                + mask_y
                + np.roll(mask_y, -1, axis=0)
            )
            inv_sum_mask = np.divide(
                1.0, sum_mask, out=np.zeros_like(sum_mask), where=sum_mask != 0
            )
            """

        else:
            _mp = pg.mouse.get_pos()
            if mp is not None:
                vel = np.array(_mp) - mp
                vel[0] /= STRETCH_WIDTH
                vel[1] /= STRETCH_HEIGHT

                idx = int(mp[0] / STRETCH_WIDTH)
                idy = int(mp[1] / STRETCH_HEIGHT)

                _u = sim.u
                _v = sim.v

                _u[idy - 5 : idy + 5, idx - 5 : idx + 5] += vel[0] * 1.0
                _v[idy - 5 : idy + 5, idx - 5 : idx + 5] += vel[1] * 1.0

                sim.u = _u
                sim.v = _v

            mp = np.array(_mp)
    else:
        mp = None

    sim.step()

    screen.fill((0, 0, 0))

    # """
    pxarr = pg.surfarray.pixels3d(plot_surf)

    u = sim.u
    v = sim.v
    col_map(u**2 + v**2, pxarr, gain=0.2)

    # pxarr[..., 2] = np.clip(pxarr[..., 2] + (1 - sum_mask / 4).T * 255, 0, 255)

    del pxarr

    screen.blit(
        pg.transform.scale(plot_surf, (SCR_WIDTH, SCR_HEIGHT)),
        (0, 0, SCR_WIDTH, SCR_HEIGHT),
    )
    # """

    pos_part = sim.pos_part

    for i in range(sim.N_PART):
        pg.draw.circle(
            screen,
            (255, 255, 255),
            (pos_part[i, 0] * STRETCH_WIDTH, pos_part[i, 1] * STRETCH_HEIGHT),
            1,
        )

    pg.display.flip()
    clock.tick()
    print(f"{clock.get_fps():.2f} FPS", end="\r")


pg.quit()

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
