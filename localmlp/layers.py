import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import spams


class BandBlock(nn.Module):
    def __init__(self, spams_param, filters_shape, dict_size, channels_in=1, device='cpu'):
        super(BandBlock, self).__init__()
        self.dict = None
        self.dict_tensor = None
        self.dict_size = dict_size
        self.filters_shape = filters_shape  # H,W
        self.param = spams_param
        self.device = device
        self.param["K"] = self.dict_size
        self.channels_out = dict_size
        self.channels_in = channels_in
        self.model = None

    def train(self, patches, standardize=False):
        # call spams train memory
        if len(data.shape > 2):
            data = patches.reshape(data.shape[0], -1)
        patches = patches.T
        patches = np.asfortranarray(patches)
        self.dict, self.model = learn_dictionary(
            patches, self.param, standardize_input=standardize)
        self.dict_tensor = torch.tensor(
            self.dict.T.reshape((self.channels_out, self.channels_in)+self.filters_shape)).to(device)

    def forward(self, input):
        # do conv
        return F.conv2d(input=input, weight=self.dict_tensor)


class LocalMlpLayer(nn.Module):
    def __init__(self, filters_shape, dict_size, channels_in=1, batch_size=50, band_indices=None, band_blocks=None, n_bands=None, spams_param=None, device='cpu'):
        super(LocalMlpLayer, self).__init__()
        self.dict_size = dict_size
        self.filters_shape = filters_shape
        self.batch_size = batch_size

        # Allow for self self specified band locations, to overlap bands if wanted
        if band_indices:
            self.band_indices = band_indices
        elif band_blocks:
            n_bands = len(band_blocks)
            self.band_blocks = torch.ModuleList([list(band_blocks)])
            self.band_indices = [i*filters_shape[0] for i in range(n_bands)]
        elif n_bands:
            self.band_indices = [i*filters_shape[0] for i in range(n_bands)]

        self.n_bands = len(band_indices)

        if not self.band_blocks:
            self.band_blocks = torch.ModuleList(
                [BandBlock(spams_param.copy(), filters_shape, dict_size, channels_in, device) for i in range(self.n_bands)])

        self.patches = [[] for i in range(self.n_bands)]

    def extract_patches(self, input):
        self.patches = [[] for i in range(self.n_bands)]
        for i in range(len(self.band_blocks)):
            self.patches[i].append(patch(input, batch_size=self.batch_size, filter_height=self.filters_shape[0],
                                    filter_width=self.filters_shape[1], height_interval=[self.band_indices[i], self.band_indices[i]+1]))

    def train(self, data):
        # Patch here?
        self.patches = [np.concatenate(p, axis=0) for p in self.patches]
        for i, band in enumerate(self.band_blocks):
            # Need to make sure we're using patched data
            band.train(self.patches[i])
        del self.patches

    def forward(self, input):
        output = []
        input = torch.tensor(input).to(self.device)
        for i, band in enumerate(self.band_blocks):
            output.append(band.forward(
                input[:, :, self.band_indices[i]:self.band_indices[i]+self.filters_shape[0], :]))
        # conncatenate along the freq dimension
        output = torch.cat(output, dim=-2)
        return output


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


def learn_dictionary(patches, param, model=None, standardize_input=True):
    X = patches.T
    if standardize_input:
        X = standardize(X)
    X = np.asfortranarray(X, dtype=float)
    if model:
        (D, model) = spams.trainDL(
            X, return_model=True, model=model, **param)
        param['D'] = D
    else:
        (D, model) = spams.trainDL_Memory(X, return_model=True, **param)
        param['D'] = D
    return D, model


def standardize(X):
    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
    X = X / np.tile(np.sqrt((X * X).sum(axis=0)), (X.shape[0], 1))
    return X
