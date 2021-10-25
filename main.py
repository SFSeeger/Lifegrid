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
    orbitum = creatures['aquarium']
    dx = orbitum['R']
    dt = orbitum['T']

    time = 0
    layers = layers = np.asarray(orbitum['cells']).shape[0] if len(np.asarray(orbitum['cells']).shape) >= 3 else 1

    # print(np.asarray(orbitum['cells']).shape)
    As = [np.zeros([size, size]) for i in range(layers)]
    Cs = [
        scipy.ndimage.zoom(np.asarray(c), scale, order=0)
        for c in orbitum['cells']
    ]
    dx *= scale
    if layers > 1:
        for A, C in zip(As, Cs):
            A[cx : cx + C.shape[0], cy : cy + C.shape[1]] = C
    else:
        Cs = np.asarray(Cs)
        As[0][cx : cx + Cs.shape[0], cy : cy + Cs.shape[1]] = Cs

    s = Simulation()
    K_FFTs, nKs = sh.generate_smooth_kernels(orbitum['kernels'], dx, size, size)

    if layers > 1:
        img = np.dstack(As)
    else:
        img = As[0]
    plt.imsave(f"out/frame{0}.png", img)

    for i in range(1, 2):
        As = s.complex_step(
            As,
            s.calculate_growths(As, orbitum['kernels'], K_FFTs, layers),
            orbitum['kernels'],
            dt,
            layers,
        )
        if layers > 1:
            img = np.dstack(As)
        else:
            img = As[0]
        plt.imsave(f"out/frame{i}.png", img)

if __name__ == "__main__":
    run()
