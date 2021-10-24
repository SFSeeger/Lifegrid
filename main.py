import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from simulation.simulation import *
from simulation.creatures import creatures
from simulation.utils import *
from simulation.utils import Renderer


def run():
    sh = Simulation_Helper()
    size = 128
    scale = 1
    cx, cy = 20, 20
    orbitum = creatures['emitter']
    dx = orbitum['R']
    dt = orbitum['T']

    time = 0
    layers = np.asarray(orbitum['cells']).shape[0]

    As = [np.zeros([size, size]) for i in range(layers)]
    Cs = [
        scipy.ndimage.zoom(np.asarray(c), scale, order=0)
        for c in orbitum['cells']
    ]
    dx *= scale
    for A, C in zip(As, Cs):
        A[cx : cx + C.shape[0], cy : cy + C.shape[1]] = C
    '''
    C = np.asarray(orbitum['cells'])
    print(C.shape)
    '''

    s = Simulation()
    K_FFTs = sh.generate_smooth_kernels(orbitum['kernels'], dx, size, size)
    s.complex_step(
        As,
        s.calculate_growths(As, orbitum['kernels'], K_FFTs),
        orbitum['kernels'],
        dt,
        layers,
    )

    for i in range(300):
        As = s.complex_step(
            As,
            s.calculate_growths(As, orbitum['kernels'], K_FFTs),
            orbitum['kernels'],
            dt,
            layers,
        )
        plt.imsave(f"out/frame{i}.png", np.dstack(As))


if __name__ == "__main__":
    run()
