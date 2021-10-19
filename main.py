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
    renderer = Renderer('out/')

    size = 128
    scale = 1
    cx, cy = 20, 20
    mid = size // 2

    orbitum = creatures['fish']
    dx = 1/orbitum['R']
    time_step = orbitum['T']
    #orbitum['b'] = np.asarray(orbitum['b'])

    C = np.asarray(orbitum['cells'])
    ''' create empty world '''
    A = np.zeros([size, size])
    ''' place scaled pattern '''
    C = scipy.ndimage.zoom(C, scale, order=0)
    dx *= scale
    A[cx : cx + C.shape[0], cy : cy + C.shape[1]] = C
    km = [(sh.generate_smooth_kernel(np.asarray(b['b']), dx, size, size)) for b in orbitum['kernels']]

    #s = Simulation(k, time_step, orbitum['m'], orbitum['s'])
    for i in range(len(km)):
        plt.imsave(f"kernel{i}.png", km[i][0])

if __name__ == "__main__":
    run()
