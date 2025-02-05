import numpy as np


def col_map(f, c, gain=1.0, base_color=(1.0, 1.0, 1.0)):
    _gr = (1.0 - np.exp(-f * gain)).T

    for i in range(3):
        c[..., i] = (255 * _gr * base_color[i]).astype(np.uint8)


def stepwise(x):
    return np.maximum(
        0.0, np.minimum(1.0, 2.0 - 6.0 * x) * np.minimum(1.0, 2.0 + 6.0 * x)
    ) + np.clip(6.0 * x - 4.0, 0.0, 1.0)


def hsv_to_rgb(hsv, rgb):
    rgb[..., 0] = stepwise(hsv[..., 0]) * hsv[..., 2]
    rgb[..., 1] = stepwise(hsv[..., 0] - 1.0 / 3.0) * hsv[..., 2]
    rgb[..., 2] = stepwise(hsv[..., 0] - 2.0 / 3.0) * hsv[..., 2]


def gen_brush_stencil(radius: int) -> np.ndarray:

    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))

    stencil = 1.0 * ((x**2 + y**2) <= radius**2)

    x = x[stencil == 1]
    y = y[stencil == 1]
    stencil = stencil[stencil == 1]

    return stencil.flatten(), x.flatten(), y.flatten()
