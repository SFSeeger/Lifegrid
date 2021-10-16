import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import asfarray
import scipy
import cv2
from simulation.simulation import *
from simulation.creatures import creatures
from simulation.utils import *
from simulation.utils import Renderer

sh = Simulation_Helper()
renderer = Renderer('out/')

size = 128
scale = 1
cx, cy = 20, 20
mid = size // 2

orbitum = creatures['geminium']
radius = orbitum['R']
time_step = orbitum['T']
orbitum['b'] = np.asarray(orbitum['b'])

C = np.asarray(orbitum['cells'])
''' create empty world '''
A = np.zeros([size, size])
''' place scaled pattern '''
C = scipy.ndimage.zoom(C, scale, order=0)
radius *= scale
A[cx : cx + C.shape[0], cy : cy + C.shape[1]] = C
kernel_form = (
    np.linalg.norm(np.asarray(np.ogrid[-mid:mid, -mid:mid], dtype=object) + 1)
    / radius
    * len(orbitum['b'])
)
# k = (kernel_form < len(orbitum['b'])) * sh.generate_bell(kernel_form, 0.5, 0.15)
k = (
    (kernel_form < len(orbitum['b']))
    * orbitum['b'][np.minimum(kernel_form.astype(int), len(orbitum['b']) - 1)]
    * sh.generate_bell(kernel_form % 1, 0.5, 0.15)
)
k = sh.trim(k)
k = k / np.sum(k)

s = Simulation(k, time_step, orbitum['m'], orbitum['s'])
renderer.step_to_image(A, s, 2)
renderer.images_to_video('out.avi', 18)
