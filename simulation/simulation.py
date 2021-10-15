from re import M
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal


class Simulation_Helper:
    def __init__(self) -> None:
        pass

    def generate_bell(self, x, m, s):
        """
        m: x-axis shift
        s: y-axis spread
        """
        bell = lambda x, m, s: np.exp(-(((x - m) / s) ** 2) / 2)
        return bell(x, m, s)

    def generate_conway_kernel(self):
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def trim(self, K):
        mask = K == 0
        rows = np.flatnonzero((~mask).sum(axis=1))
        cols = np.flatnonzero((~mask).sum(axis=0))
        return K[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


class Simulation:
    """
    Parameter:
    K: Kernel
    T: dt
    m: growth function x-axis shift
    s: growth function y-axis spread
    """

    def __init__(self, K, T, m, s) -> None:
        self.K = K
        self.T = T
        self.m = m
        self.s = s
        self.sh = Simulation_Helper()

    def create_field(self, size_x: int, size_y: int):
        return np.zeros((size_x, size_y))

    def apply_kernel(self, A):
        return signal.convolve2d(A, self.K, mode='same', boundary='wrap')

    def apply_growth(self, A, U):
        return self.growth_function(A, U)

    def growth_function(self, A, U):
        return np.logical_or(np.logical_and(A, (U == 2)), (U == 3))

    def step(self, A):
        return np.clip(
            self.apply_growth(A, self.apply_kernel(A, self.K)), 0, 1
        )

    # smooth
    def smooth_growth_function(self, U):
        return self.sh.generate_bell(U, self.m, self.s) * 2 - 1

    def smooth_step(self, A):
        return np.clip(
            A + 1 / self.T * self.smooth_growth_function(self.apply_kernel(A)),
            0,
            1,
        )

    def show_growth_func(self):
        plt.plot(
            np.arange(0.0, 1.0, 0.005),
            self.smooth_growth_function(np.arange(0.0, 1.0, 0.005)),
        )
        plt.show()
