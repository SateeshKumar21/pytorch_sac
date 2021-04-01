import os.path as osp
import sys

import imageio
import numpy as np
from PIL import Image

import utils


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, fps=30):
        self.root_dir = root_dir
        self.save_dir = utils.make_dir(root_dir, "video") if root_dir else None
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(mode="rgb_array")
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = osp.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

    def save_frames(self, dirname, ext="png"):
        if self.enabled:
            for i, frame in enumerate(self.frames):
                filename = osp.join(dirname, f"{i}.{ext}")
                image = Image.fromarray(frame)
                image.save(filename, format=ext)
