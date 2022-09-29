from typing import List

import numpy as np

from objects import Object, Plane, Sphere
from phong import Light, Material

LIGHTS = [
    Light(np.array([20, 20, 20]), 1.5),
    Light(np.array([30, 50, -25]), 1.8),
    Light(np.array([30, 20, 30]), 1.7),
]


RUBY = Material(
    np.array([0.1745, 0.01175, 0.01175]),
    np.array([0.61424, 0.04136, 0.04136]),
    np.array([0.727811, 0.296648, 0.296648]),
    10,
    0.1,
)

OBSIDIAN = Material(
    np.array([0.05375, 0.05, 0.06625]),
    np.array([0.18275, 0.17, 0.22525]),
    np.array([0.332741, 0.32863, 0.346435]),
    50,
    0.4,
)

EMERLAD = Material(
    np.array([0.0215, 0.1745, 0.0215]),
    np.array([0.07568, 0.61424, 0.07568]),
    np.array([0.633, 0.727811, 0.633]),
    120,
    0.6,
)

GRAY = Material(
    np.array([0.1, 0.18725, 0.1745]),
    np.array([0.396, 0.74151, 0.69102]),
    np.array([0.297254, 0.30829, 0.306678]),
    50,
    0.05,
)


MIRROR = Material(np.array([0.3, 0.3, 0.3]), np.array([0.0, 0.0, 0.0]), np.array([1, 1, 1]), 1500, 1)

shift_y, shift_z = -2, -3

a1 = np.zeros(3) + (0, -3.5 + shift_y, 0 + shift_z)
b1 = np.zeros(3) + (1, -3.5 + shift_y, -1 + shift_z)
c1 = np.zeros(3) + (0, -3.5 + shift_y, -1 + shift_z)

a2 = np.zeros(3) + (9, -3.5 + shift_y, -19 + shift_z)
b2 = np.zeros(3) + (-9, -3.5 + shift_y, -19 + shift_z)
c2 = np.zeros(3) + (9, 0 + shift_y, -19 + shift_z)

a3 = np.zeros(3) + (-9, -3.5 + shift_y, -19 + shift_z)
b3 = np.zeros(3) + (-9, -3.5 + shift_y, 0 + shift_z)
c3 = np.zeros(3) + (-9, 0 + shift_y, -19 + shift_z)

a4 = np.zeros(3) + (9, -3.5 + shift_y, -19 + shift_z)
b4 = np.zeros(3) + (9, -3.5 + shift_y, 0 + shift_z)
c4 = np.zeros(3) + (9, 0 + shift_y, -19 + shift_z)

SPHERES = [
    Sphere(np.array((-4.5, 0 + shift_y, -16 + shift_z)), 2, EMERLAD),
    Sphere(np.array((-2.5, -1.5 + shift_y, -12 + shift_z)), 2, RUBY),
    Sphere(np.array((0, -0.5 + shift_y, -18 + shift_z)), 3, RUBY),
    Sphere(np.array((5.6, 5 + shift_y, -18 + shift_z)), 4, MIRROR),
]

OBJECTS: List[Object] = SPHERES + [
    Plane(
        [a1, b1, c1],
        OBSIDIAN,
        bounds=np.array([[-9.2, 9.2], [-5 + shift_y, 5 + shift_y], [-20 + shift_z, -7 + shift_z]]),
    ),
    Plane(
        [a2, b2, c2],
        GRAY,
        bounds=np.array([[-9.1, 9.1], [-3.6 + shift_y, 10 + shift_y], [-20 + shift_z, -7 + shift_z]]),
    ),
    Plane(
        [a3, b3, c3],
        GRAY,
        bounds=np.array([[-9.1, 9.1], [-3.6 + shift_y, 10 + shift_y], [-20.1 + shift_z, -7 + shift_z]]),
    ),
    Plane(
        [a4, b4, c4],
        GRAY,
        bounds=np.array([[-9.1, 9.1], [-3.6 + shift_y, 10 + shift_y], [-20.1 + shift_z, -7 + shift_z]]),
    ),
]
