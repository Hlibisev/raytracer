from typing import List

import cv2
import numpy as np
from numpy.linalg import norm

from data import LIGHTS, OBJECTS
from objects import Object


def reflect(N, L):
    return L - 2 * np.dot(L, N) * N


def simple_raycast(orig, dir, objects: List[Object], depth):
    if depth == 5:
        return 0

    neirest_object = None
    min_dist = np.infty

    for i, object in enumerate(objects):
        _, dist = object.ray_intersect(orig, dir)

        if dist < min_dist:
            index = i
            min_dist = dist
            neirest_object = object

    if neirest_object is not None:
        point = orig + dir * min_dist

        N = neirest_object.normal(point)

        intensity = 0
        specular_intensity = 0

        for light in LIGHTS:
            light_dir = light.position - point
            light_dir /= norm(light_dir)

            shadow_orig = point - 1e-3 * N if np.dot(light_dir, N) < 0 else point + 1e-3 * N

            if any(object.ray_intersect(shadow_orig, light_dir)[0] for k, object in enumerate(objects) if k != index):
                continue

            intensity += max(0, np.dot(light_dir, N)) * light.intensity
            specular_intensity += (
                max(0, np.dot(reflect(N, light_dir), dir)) ** neirest_object.material.specular_exponent
                * light.intensity
            )

        color = (
            neirest_object.material.albedo_color * light.intensity * 0.2
            + neirest_object.material.diffuse_color * intensity * 0.7
            + neirest_object.material.specular_color * specular_intensity * 0.9
        )

        reflect_dir = reflect(N, dir)
        reflect_orig = point - N * 1e-4 if np.dot(reflect_dir, N) < 0 else point + N * 1e-4
        color += (
            simple_raycast(reflect_orig, reflect_dir, objects, depth=depth + 1) * neirest_object.material.reflection
        )

        return color

    return np.zeros(3) + [0.1, 0.5, 0.9]


def main(fov=np.pi / 2):
    n = 1000
    image = np.zeros((n, n, 3))

    widght_screen = np.tan(fov / 2) * 2
    height_screen = np.tan(fov / 2) * 2

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            orig = np.zeros(3)

            x = (j / image.shape[1] - 0.5) * widght_screen
            y = -(i / image.shape[0] - 0.5) * height_screen

            dir = np.array([x, y, -1])
            dir /= norm(dir)

            image[i, j] = simple_raycast(orig, dir, OBJECTS, 0) * 255

    cv2.imwrite("img.png", image[..., ::-1])


if __name__ == "__main__":
    main()
