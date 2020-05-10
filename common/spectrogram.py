import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


def create_filters_log2(d=4096, k=240, low=20, high=6000, sr=44000.):
    x = np.linspace(0, 2*np.pi, d, endpoint=False)
    wsin = np.empty((k, 1, d), dtype=np.float32)
    wcos = np.empty((k, 1, d), dtype=np.float32)
    start_freq = low
    end_freq = high
    num_cycles = start_freq*d/sr
    scaling_ind = np.log2(end_freq/start_freq)/k
    window_mask = 1.0-1.0*np.cos(x)
    for ind in range(k):
        wsin[ind, 0, :] = window_mask*np.sin(2**(ind*scaling_ind)*num_cycles*x)
        wcos[ind, 0, :] = window_mask*np.cos(2**(ind*scaling_ind)*num_cycles*x)
    return wsin, wcos


class Spectrogrammer(nn.Module):
    def __init__(self, window, sr, k=None, n_octaves=None, k_per_octave=None, low=20, high=20000,stride=32):
        super(Spectrogrammer, self).__init__()
        self.low = low
        if (n_octaves is not None) and (k_per_octave is not None):
            self.k = n_octaves * k_per_octave
            self.n_octaves = n_octaves
            self.k_per_octave = k_per_octave
            self.high = self.low * 2**n_octaves
        else:
            self.k = k
            self.high = high
        self.window = window
        self.sr = sr
        self.stride = stride
        wsin, wcos = create_filters_log2(
            d=self.window, k=self.k, low=self.low, high=self.high, sr=self.sr)
        self.wsin_var = Variable(torch.from_numpy(wsin),requires_grad=False).type(torch.FloatTensor)
        self.wcos_var = Variable(torch.from_numpy(wcos),requires_grad=False).type(torch.FloatTensor)

    def forward(self,signal):
        if type(signal) is np.ndarray:
            signal = torch.FloatTensor(signal)
        return (F.conv1d(signal[:,None,:], self.wsin_var, stride=self.stride).pow(2) \
            + F.conv1d(signal[:,None,:], self.wcos_var, stride=self.stride).pow(2)).pow(1/2)