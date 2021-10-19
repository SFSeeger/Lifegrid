import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from simulation.simulation import *
from simulation.creatures import creatures
from simulation.utils import *
from simulation.utils import Renderer
from interface import interface


def test():
    sh = Simulation_Helper()
    renderer = Renderer('out/')

    size = 128
    scale = 1
    cx, cy = 20, 20
    mid = size // 2

    orbitum = creatures['fish']
    dx = orbitum['R']
    time_step = orbitum['T']
    # orbitum['b'] = np.asarray(orbitum['b'])

    C = np.asarray(orbitum['cells'])
    ''' create empty world '''
    A = np.zeros([size, size])
    ''' place scaled pattern '''
    C = scipy.ndimage.zoom(C, scale, order=0)
    dx *= scale
    A[cx : cx + C.shape[0], cy : cy + C.shape[1]] = C
    km = [
        (sh.generate_smooth_kernel(np.asarray(b['b']), dx, size, size))
        for b in orbitum['kernels']
    ]

    # s = Simulation(k, time_step, orbitum['m'], orbitum['s'])
    for i in range(len(km)):
        data = sh.trim(km[i][0])
        plt.imsave(f"kernel{i}.png", km[i][0])
        plt.imsave(f"kernel{i}.png", sh.trim(km[i][0]))


def run():
    sh = Simulation_Helper()
    size = 128
    scale = 1
    cx, cy = 20, 20
    mid = size // 2
    orbitum = creatures['fish']
    dx = orbitum['R']
    dt = orbitum['T']

    time = 0

    C = np.asarray(orbitum['cells'])
    ''' create empty world '''
    A = np.zeros([size, size])
    ''' place scaled pattern '''
    C = scipy.ndimage.zoom(C, scale, order=0)
    dx *= scale
    A[cx : cx + C.shape[0], cy : cy + C.shape[1]] = C

    s = Simulation()
    km = [
        (sh.generate_smooth_kernel(np.asarray(b['b']), dx, size, size))
        for b in orbitum['kernels']
    ]
    data = [
        {'m': b['m'], 's': b['s'], 'h': b['h']} for b in orbitum['kernels']
    ]
    h = 0

    for _ in range(len(data) - 1):
        h += data[_]['h']

    for i in range(300):
        growth, potential = s.calculate_growth(
            A, km[0][1], data[0]['m'], data[0]['s']
        )
        world = data[0]['h'] / h * growth
        for _ in range(1, len(km)):
            growth, potential = s.calculate_growth(
                A, km[_][1], data[_]['m'], data[_]['s']
            )
            world += data[_]['h'] / h * growth
        A = np.clip((A + dt * world), 0, 1)
        time += dt
        plt.imsave(f"out/frame{i}.png", A)


def render_window():
    interface.render()

if __name__ == "__main__":
    render_window()
    #run()
