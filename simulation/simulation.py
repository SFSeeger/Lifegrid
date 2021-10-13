import numpy as np


class Simulation:
    def __init__(self) -> None:
        pass

    def create_field(self, size_x: int, size_y: int):
        return np.zeros((size_x, size_y))

    def apply_kernel(self, kernel, world):
        pass

    def render(self):
        pass
