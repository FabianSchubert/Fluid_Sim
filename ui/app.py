import pygame as pg
import numpy as np

from simulator import Simulator

from .utils import col_map, hsv_to_rgb, gen_brush_stencil

import os

DIR = os.path.dirname(os.path.abspath(__file__))


class App:
    def __init__(
        self,
        simulator: Simulator,
        size: tuple[int, int] = (640, 480),
        brush_radius: int = 5,
    ) -> None:
        self._running = False

        self._display_surf = None
        self._sim_surf = None

        self.screen_size = size

        self.simulator = simulator

        self.stretch = tuple(
            self.screen_size[i] / self.simulator.size[i] for i in range(2)
        )

        self.brush_radius = brush_radius
        self.mouse_stencil, self.mouse_stencil_x, self.mouse_stencil_y = (
            gen_brush_stencil(brush_radius)
        )
        self.dragging = False
        self.drawing_obstacle = False
        self.drawing_rho = 0

        self.mouse_pos = np.zeros((2, 2))

        self.prev_draw_pos = None

        self.sim_running = True

    def on_init(self):
        _, n_fail = pg.init()
        self._display_surf = pg.display.set_mode(
            self.screen_size, pg.HWSURFACE | pg.DOUBLEBUF
        )
        pg.display.set_caption("Python Fluid Simulation")

        pg.display.set_icon(pg.image.load(os.path.join(DIR, "logo.png")))

        self._sim_surf = pg.Surface(self.simulator.size)

        self._part_surf = pg.Surface(self.screen_size, pg.SRCALPHA)

        self._running = True

        return n_fail == 0

    def on_event(self, event):
        if event.type == pg.QUIT:
            self._running = False

        if event.type == pg.MOUSEBUTTONDOWN:

            if event.button == 1:
                _mp = np.array(pg.mouse.get_pos()) / self.stretch
                if pg.key.get_pressed()[pg.K_1]:
                    self.drawing_rho = 1
                elif pg.key.get_pressed()[pg.K_2]:
                    self.drawing_rho = 2
                elif pg.key.get_pressed()[pg.K_3]:
                    self.drawing_rho = 3

                if self.drawing_rho:
                    self.simulator.add_rho(
                        self.drawing_rho - 1,
                        self.mouse_stencil_x + int(_mp[0]),
                        self.mouse_stencil_y + int(_mp[1]),
                        1.0,
                    )

                else:
                    self.mouse_pos[0] = _mp
                    self.dragging = True
                    self.mouse_pos[1] = self.mouse_pos[0]
            if event.button == 3:
                self.drawing_obstacle = True
                _mp = np.array(pg.mouse.get_pos()) / self.stretch
                self.simulator.add_obstacle(
                    self.mouse_stencil_x + int(_mp[0]),
                    self.mouse_stencil_y + int(_mp[1]),
                )

        if event.type == pg.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
                self.drawing_rho = 0
            if event.button == 3:
                self.drawing_obstacle = False
                self.prev_draw_pos = None

        if event.type == pg.MOUSEMOTION:
            if self.drawing_obstacle:
                _mp = (np.array(pg.mouse.get_pos()) / self.stretch).astype(int)
                if self.prev_draw_pos is not None:
                    _delta = _mp - self.prev_draw_pos
                    _dist = np.linalg.norm(_delta)
                    _steps = 1 + int(_dist / 3)
                    for i in range(_steps + 1):
                        _pos = (self.prev_draw_pos + i * _delta / _steps).astype(int)
                        self.simulator.add_obstacle(
                            self.mouse_stencil_x + _pos[0],
                            self.mouse_stencil_y + _pos[1],
                        )
                else:
                    self.simulator.add_obstacle(
                        self.mouse_stencil_x + _mp[0],
                        self.mouse_stencil_y + _mp[1],
                    )
                self.prev_draw_pos = _mp
            if self.drawing_rho:
                _mp = (np.array(pg.mouse.get_pos()) / self.stretch).astype(int)
                self.simulator.add_rho(
                    self.drawing_rho-1,
                    self.mouse_stencil_x + _mp[0],
                    self.mouse_stencil_y + _mp[1],
                    1.0,
                )

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                self.sim_running = not self.sim_running

    def on_loop(self):
        if self.dragging:
            _mp = pg.mouse.get_pos()
            self.mouse_pos[1] = self.mouse_pos[0]
            self.mouse_pos[0] = (
                self.mouse_pos[0] * 0.8 + 0.2 * np.array(_mp) / self.stretch
            )
            _v = self.mouse_pos[0] - self.mouse_pos[1]

            self.simulator.add_force(
                (self.mouse_stencil_x + self.mouse_pos[0, 0]).astype(int),
                (self.mouse_stencil_y + self.mouse_pos[0, 1]).astype(int),
                _v[0] * self.simulator.to_device(self.mouse_stencil),
                _v[1] * self.simulator.to_device(self.mouse_stencil),
                mode="direct",
            )

        if self.sim_running:
            self.simulator.step()

    def on_render(self):
        self._display_surf.fill((0, 0, 0))

        pxarr = pg.surfarray.pixels3d(self._sim_surf)

        #u = self.simulator.to_host(self.simulator.uc)
        #v = self.simulator.to_host(self.simulator.vc)
        rho = self.simulator.to_host(self.simulator.rho)

        solid_mask = self.simulator.to_host(1.0 - self.simulator.mask_v)

        # hsv = np.zeros((self.simulator.size[1], self.simulator.size[0], 3))
        # hsv[..., 0] = 0.0#(np.arctan2(u.T, v.T) + np.pi) / (2.0 * np.pi)
        # hsv[..., 2] = 1.0 - np.exp(-rho.T, hsv[..., 2] * 0.02)
        rgb = np.zeros((self.simulator.WIDTH, self.simulator.HEIGHT, 3))
        for i in range(self.simulator.N_SUBST):
            rgb[..., i] = (1.0 - np.exp(-rho[i].T * 0.5)) * 255
        # hsv_to_rgb(hsv, rgb)

        # col_map(u**2 + v**2, pxarr, gain=0.2)
        pxarr[:] = rgb.astype(np.uint8)

        obst_coord = np.where(solid_mask)
        pxarr[obst_coord[1], obst_coord[0], :] = np.array([0, 0, 255])

        del pxarr

        self._display_surf.blit(
            pg.transform.scale(self._sim_surf, self.screen_size),
            (0, 0, self.screen_size[0], self.screen_size[1]),
        )

        """
        pos_part = self.simulator.pos_part

        self._part_surf.fill((0, 0, 0, 0))
        for i in range(self.simulator.N_PART):
            pg.draw.circle(
                self._part_surf,
                (255, 255, 255, 100),
                (pos_part[i, 0] * self.stretch[0], pos_part[i, 1] * self.stretch[1]),
                1,
            )
            # pg.gfxdraw.filled_circle(
            #    self._display_surf,
            #    pos_part[i, 0] * self.stretch[0],
            #    pos_part[i, 1] * self.stretch[1],
            #    1,
            #    (255, 255, 255, 30),
            # )
        

        self._display_surf.blit(
            self._part_surf, (0, 0, self.screen_size[0], self.screen_size[1])
        )
        # """

        pg.display.flip()

    def on_cleanup(self):
        pg.quit()

    def run(self):
        if not self.on_init():
            self._running = False

        while self._running:
            for event in pg.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()
