from re import M
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from random import randint


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

    def generate_smooth_kernel(self, beta, dx, sizeX, sizeY):
        midX = sizeX // 2
        midY = sizeY // 2
        radius = (
            np.linalg.norm(
                np.asarray(np.ogrid[-midX:midX, -midY:midY], dtype=object) + 1
            )
            / dx
        )
        print(beta)
        Br = len(beta) * radius
        kernel = (
            (Br < len(beta))
            * beta[np.minimum(Br.astype(int), len(beta) - 1)]
            * self.generate_bell(Br % 1, 0.5, 0.15)
        )
        kernel = kernel / np.sum(kernel)
        kernel_FFT = np.fft.fft2(kernel)
        return kernel, kernel_FFT

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

    def __init__(self) -> None:
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
    def calculate_growth(self, A, K_FFT, m, s):
        world_fft = np.fft.fft2(A)
        potential = world_fft * K_FFT
        potential = np.fft.fftshift(np.real(np.fft.ifft2(potential)))

        growth = self.smooth_growth_function(potential, m, s)

        return growth, potential

    """
    def run_automation(self, A, dt, growth):

        h: sum of hk

        new_world = np.clip((A + dt * growth), 0, 1)
        return new_world
    """

    def run_layer_automation(self, growth, hk, h):
        new_world = hk / h * growth

    def run_complex_automation(self, A, dt, layers):
        layer_sum = layers[0]
        for _ in range(1, len(layers)):
            layer_sum += layers[_]
        new_world = np.clip((A + dt * layer_sum), 0, 1)

    def apply_growth(self, potential, m, s):
        growth = self.smooth_growth_function(potential, m, s)

    def smooth_growth_function(self, U, m, s):
        return self.sh.generate_bell(U, m, s) * 2 - 1

    def show_growth_func(self, u, m):
        plt.plot(
            np.arange(0.0, 1.0, 0.005),
            self.smooth_growth_function(np.arange(0.0, 1.0, 0.005), u, m),
        )
        plt.show()
