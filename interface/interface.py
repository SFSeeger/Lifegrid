import time
import tkinter as tk
import numpy as np
import scipy
from PIL import Image, ImageTk
from simulation.creatures import creatures
from simulation.simulation import Simulation_Helper
import matplotlib.pyplot as plt
from matplotlib import cm

def render():
    root = tk.Tk()
    sh = Simulation_Helper()
    size = 128
    scale = 1
    cx, cy = 20, 20
    mid = size // 2

    orbitum = creatures['fish']
    dx = orbitum['R']
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
    imgpixel = Image.open("kernel0.png")
    canvas = tk.Canvas(root, width=700, height=700)
    canvas.pack()
    for i in range(len(km)-1):
        img = Image.fromarray(np.uint8(cm.plasma(km[i][0]) * 6502))
        img = img.resize((500,500), Image.BILINEAR)
        img = ImageTk.PhotoImage(image=img, format="PGM")

        canvas.create_image(0, 0, anchor="nw", image=img)
        time.sleep(1)
        canvas.update()

    root.mainloop()