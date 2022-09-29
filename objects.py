from dataclasses import dataclass
from typing import List, Protocol

import numpy as np
from numpy.linalg import norm

from phong import Material


class Object(Protocol):
    material: Material
    reflection: float

    def ray_intersect(self) -> List[int]:
        ...

    def normal(self, point) -> np.ndarray:
        ...


@dataclass
class Sphere:
    position: np.ndarray
    radius: float
    material: Material

    def ray_intersect(self, orig, dir):
        """https://www.ctralie.com/PrincetonUGRAD/Projects/COS426/Assignment3/part1.html#raysphere"""
        L = self.position - orig

        a = np.dot(L, dir)
        c2 = np.dot(L, L)
        h2 = c2 - a**2

        if h2 > self.radius**2:
            return False, np.infty

        half_a_inside_sphere = np.sqrt(self.radius**2 - h2)

        if a - half_a_inside_sphere > 0:
            return True, a - half_a_inside_sphere

        if a + half_a_inside_sphere > 0:
            return True, a + half_a_inside_sphere

        return False, np.infty

    def normal(self, point):
        "point must belong object"
        N = point - self.position
        N /= norm(N)

        return N


@dataclass
class Plane:
    def __init__(
        self, points: List[np.ndarray], material: Material, bounds=np.array([[-5, 5], [-5, 5], [-20, -5]])
    ) -> None:
        self.points = points
        self.material = material
        self.bounds = bounds
        self.lin_coeff, self.constant = self.get_coeffitient(points)

    def ray_intersect(self, orig, dir):
        """hand made algorithm:
        trace = orig + t * dir,
        place: a * x + b * y + c * z + d = 0
        """

        coef_t = np.dot(self.lin_coeff, dir)
        constant_t = -self.constant - np.dot(self.lin_coeff, orig)

        if abs(coef_t) < 1e-7 and abs(constant_t) < 1e-7:
            return True, 0

        elif abs(coef_t) > 1e-7 and abs(constant_t) < 1e-7:
            return False, np.infty

        t = constant_t / coef_t

        if t < 0:
            return False, np.infty

        if np.all(orig + t * dir > self.bounds[:, 0]) and np.all(orig + t * dir < self.bounds[:, 1]):
            return True, t
        else:
            return False, np.infty

    def normal(self, point):
        "point must belong object"

        return self.lin_coeff

    @staticmethod
    def get_coeffitient(points):
        # 2 vectors belong b
        x1, y1, z1 = points[0]
        x2, y2, z2 = points[1]
        x3, y3, z3 = points[2]

        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1

        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = -a * x1 - b * y1 - c * z1

        lin_coeff = np.array([a, b, c])
        constant = d / norm(lin_coeff)
        lin_coeff /= norm(lin_coeff)

        return lin_coeff, constant
