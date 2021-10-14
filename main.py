import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image, ImageDraw
from simulation.simulation import Simulation, Simulation_Helper
from simulation.creatures import creatures

sh = Simulation_Helper()

size = 128
scale = 1
cx, cy = 20, 20

orbitum = creatures['Orbitum']
radius = orbitum['R']
time_step = orbitum['T']

C = np.asarray(orbitum['cells'])
''' create empty world '''
# A = np.random.rand(size, size)
A = np.zeros([size, size])
''' place scaled pattern '''
C = scipy.ndimage.zoom(C, scale, order=0)
radius *= scale
A[cx : cx + C.shape[0], cy : cy + C.shape[1]] = C

# radius = 10
# time_step = 10
kernel_form = (
    np.linalg.norm(np.asarray(np.ogrid[-radius:radius, -radius:radius]) + 1)
    / radius
)

k = (kernel_form < 1) * sh.generate_bell(kernel_form, 0.5, 0.15)
k = k / np.sum(k)

s = Simulation(k, time_step, orbitum['m'], orbitum['s'])
# plt.imsave('kernel.png', k)

plt.imsave(f'out/{0}.png', A)
for i in range(1, 100):
    ss = s.smooth_step(A)
    plt.imsave(f'out/{i}.png', ss)
    A = ss
