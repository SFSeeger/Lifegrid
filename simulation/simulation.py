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

    def generate_smooth_kernels(self, Ks, R, sizeX, sizeY):
        midX = sizeX // 2
        midY = sizeY // 2

        K_forms = [
            np.linalg.norm(np.ogrid[-midX:midX, -midY:midY])
            / R
            * len(K['b'])
            / K['r']
            for K in Ks
        ]
        kernels = [
            (D < len(K['b']))
            * np.asarray(K['b'], dtype=object)[
                np.minimum(D.astype(int), len(K['b']) - 1)
            ]
            * self.generate_bell(D % 1, 0.5, 0.15)
            for D, K in zip(K_forms, Ks)
        ]
        nKs = [K / np.sum(K) for K in kernels]

        K_FFTs = [np.fft.fft2(np.fft.fftshift(K)) for K in nKs]
        return K_FFTs

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

    # smooth
    def calculate_growths(self, As, Ks, K_FFTs):
        A_ffts = [np.fft.fft2(A) for A in As]
        potentials = [
            np.real(np.fft.ifft2(A_ffts[K['c0']] * K_FFT))
            for K_FFT, K in zip(K_FFTs, Ks)
        ]
        growths = [
            self.smooth_growth_function(U, K['m'], K['s'])
            for U, K in zip(potentials, Ks)
        ]

        return growths

    def complex_step(self, As, Gs, Ks, T, layers):
        # Calculate every Layer and their weight
        Hs = [
            sum(K['h'] * G for K, G in zip(Ks, Gs) if K['c1'] == c1)
            for c1 in range(layers)
        ]
        As = [np.clip(A + 1 / T * H, 0, 1) for A, H in zip(As, Hs)]
        return As

    def smooth_growth_function(self, U, m, s):
        return self.sh.generate_bell(U, m, s) * 2 - 1

    def show_growth_func(self, m, s):
        plt.plot(
            np.arange(0.0, 1.0, 0.005),
            self.smooth_growth_function(np.arange(0.0, 1.0, 0.005), m, s),
        )
        plt.show()

    def show_multiple_growth_func(self, Ks):
        for K in Ks:
            plt.plot(
                np.arange(0.0, 1.0, 0.005),
                self.smooth_growth_function(
                    np.arange(0.0, 1.0, 0.005), K['m'], K['s']
                ),
            )
        plt.show()
