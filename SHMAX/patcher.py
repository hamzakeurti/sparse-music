
import numpy as np


class Patcher:
    def __init__(self, batch_size, filter_size, width_interval=None, height_interval=None):
        self.batch_size = batch_size
        self.width_interval = width_interval
        self.height_interval = height_interval
        self.filter_size = filter_size

    def extract_patches(self, input):
        # Patches will be selected from index as upper left corner
        if not self.height_interval:
            self.height_interval = (0, input.shape[-2] - self.filter_size + 1)
        if not self.width_interval:
            self.width_interval = (0, input.shape[-1] - self.filter_size + 1)

        patches = patch(input, batch_size=self.batch_size, filter_height=self.filter_size,
                        filter_width=self.filter_size, width_interval=self.width_interval, height_interval=self.height_interval)
        return patches.reshape(self.batch_size, -1)


def patch(input, batch_size, filter_height, filter_width, width_interval=None, height_interval=None):
    if not height_interval:
        height_interval = (0, input.shape[-2] - filter_height + 1)
    if not width_interval:
        width_interval = (0, input.shape[-1] - filter_width + 1)

    height_index = np.random.randint(
        low=height_interval[0], high=height_interval[1], size=batch_size)
    width_index = np.random.randint(
        low=width_interval[0], high=width_interval[1], size=batch_size)
    batch_index = np.random.randint(input.shape[0], size=batch_size)

    patches = np.zeros(
        (batch_size, input.shape[1], filter_height, filter_width))
    for i in range(batch_size):
        b = batch_index[i]
        h = height_index[i]
        w = width_index[i]
        fh = filter_height
        fw = filter_width
        patches[i] = input[b, :, h:h+fh, w:w+fw]

    return patches
