import matplotlib.pyplot as plt
import glob
import cv2
import os

from simulation.simulation import Simulation


class Renderer:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path

    def step_to_image(self, A, s: Simulation, frames: int) -> None:
        """
        A: World
        s: Simulation
        """
        plt.imsave(f'{self.image_path}{0}.png', A)
        for i in range(0, frames):
            ss = s.smooth_step(A)
            plt.imsave(f'{self.image_path}{i+1}.png', ss)
            A = ss

    def images_to_video(self, name: str, fps: int) -> None:
        path_array = [
            f for f in os.listdir(self.image_path) if f.endswith(".png")
        ]
        img_array = []
        path_array = sorted(
            path_array, key=lambda x: int(os.path.splitext(x)[0])
        )
        height, width, layers = cv2.imread(
            self.image_path + path_array[0]
        ).shape
        size = (width, height)
        for filename in path_array:
            img = cv2.imread(self.image_path + filename)
            img_array.append(img)
        out = cv2.VideoWriter(
            self.image_path + name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size
        )

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
