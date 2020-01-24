import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy.lib.stride_tricks import as_strided
from patcher import Patcher

import spams

WIDTH_INTERVAL = 'width_interval'
HEIGHT_INTERVAL = 'height_interval'


def to_windows(input, kernel_size, step_size=1):
    out_shape = input.shape[:2]
    out_shape += ((input.shape[2] - kernel_size)//step_size + 1,
                  (input.shape[3] - kernel_size)//step_size + 1) + (kernel_size, kernel_size)
    input_srides = input.strides
    strides = input_srides[:2] + \
        tuple(np.array(input.strides[-2:])*step_size) + input_srides[-2:]
    return as_strided(input, shape=out_shape, strides=strides, writeable=False)


class SLayer(nn.Module):
    def __init__(self, input_channels, output_channels, filter_size, batch_size=500, param={}, layer_config=None):
        super(SLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dictionary_size = output_channels
        self.filter_size = filter_size
        self.pad = self.filter_size // 2
        self.n_features = self.filter_size**2 * self.input_channels
        self.batch_size = batch_size
        self.dictionary = None
        self.model = None
        self.param = param
        self.param['K'] = self.dictionary_size
        width_interval, height_interval = None, None
        if layer_config:
            width_interval = layer_config.get(width_interval, None)
            height_interval = layer_config.get(height_interval, None)
        self.patcher = Patcher(self.batch_size, self.filter_size,width_interval=width_interval,height_interval=height_interval)

    def train(self, input):
        D = self.train_dictionary(input)
        self.dictionary = torch.tensor(D.T.reshape(
            self.output_channels, self.input_channels, self.filter_size, self.filter_size),dtype=torch.float)

    def forward_conv(self, input):
        padded = F.pad(input, (self.pad, self.pad, self.pad, self.pad))
        return F.conv2d(padded, self.dictionary)

    def backward_conv(self, feature_map):
        backward_dictionary = self.dictionary.permute(1, 0, 2, 3)
        padded = F.pad(feature_map, (self.pad, self.pad, self.pad, self.pad))
        return F.conv2d(padded, backward_dictionary)

    '''
    Correlates the input image to a learned dictionary
    (Experimental)
    '''

    def forward_corr(self, input):
        padded = F.pad(input, (self.pad, self.pad, self.pad, self.pad))
        return F.conv2d(padded, self.dictionary.flip((-1, -2)))

    '''
    Correlates the obtained feature map (code) to the learned dictionary
    (Experimental)
    '''

    def backward_corr(self, feature_map):
        backward_dictionary = self.dictionary.permute(
            1, 0, 2, 3)  # permute channels for output / input
        padded = F.pad(feature_map, (self.pad, self.pad, self.pad, self.pad))
        # Flip for correlation instead of convolution
        return F.conv2d(padded, backward_dictionary.flip((-1, -2)))

    def train_dictionary(self, input):
        # input : (batch_size, channels, height, width)
        patches = self.patcher.extract_patches(input)
        X = patches.T
        X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
        X = np.asfortranarray(
            X / np.tile(np.sqrt((X * X).sum(axis=0)), (X.shape[0], 1)), dtype=float)
        if self.model:
            (D, self.model) = spams.trainDL(
                X, return_model=True, model=self.model, **self.param)
            self.param['D'] = D
        else:
            (D, self.model) = spams.trainDL(X, return_model=True, **self.param)
            self.param['D'] = D
        return D


def learn_dictionary(patches,param,model=None,standardize_input=True):
    X = patches.T
    if standardize_input:
        X = standardize(X)
    X = np.asfortranarray(X, dtype=float)
    if model:
        (D, model) = spams.trainDL(
            X, return_model=True, model=model, **param)
        param['D'] = D
    else:
        (D, model) = spams.trainDL(X, return_model=True, **param)
        param['D'] = D
    return D,model

def standardize(X):
    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
    X = X / np.tile(np.sqrt((X * X).sum(axis=0)), (X.shape[0], 1))
    return X