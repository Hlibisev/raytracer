from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    albedo_color: np.ndarray
    diffuse_color: np.ndarray
    specular_color: np.ndarray
    specular_exponent: float
    reflection: float


@dataclass
class Light:
    position: np.ndarray
    intensity: float
