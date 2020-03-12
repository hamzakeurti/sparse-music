import numpy as np
from scipy import signal as sig
import torch

import pycochleagram.cochleagram as cgram 

def wav2cgram(sound,sr,n=50,sample_factor=4,non_linearity='power'):
    low_lim = 30
    high_lim = 7860
    x = cgram.cochleagram(sound,sr,n,low_lim,high_lim,sample_factor,nonlinearity=non_linearity)
    if len(x.shape) <3: # no batch dimension
        y = cgram.apply_envelope_downsample(x,mode="poly",audio_sr=sr,env_sr=int(256*sr/x.shape[-1]))
        y = sig.resample_poly(y,256,y.shape[-2])
    else:
        y = []
        for i in range(x.shape[0]):
            y.append( cgram.apply_envelope_downsample(x[i],mode="poly",audio_sr=sr,env_sr=int(256*sr/x[i].shape[-1])))
            y[i] = sig.resample_poly(y[i],256,y[i].shape[-2])
    # x = torch.tensor(y)
    return y


def tiled_norm(input):
    norm = np.linalg.norm(input.reshape(input.shape[0],input.shape[1],-1),axis = 2)
    expanded = np.expand_dims(np.expand_dims(norm,-1),-1)
    tiled = np.tile(expanded, (1,input.shape[-2],input.shape[-1]))
    return tiled

def evaluate_reconstruction(original,reconstruction):
    normalized_original = original / torch.tensor(tiled_norm(original))
    normalized_reconstruction = reconstruction / torch.tensor(tiled_norm(reconstruction))
    total_error = np.linalg.norm((normalized_original - normalized_reconstruction).reshape(original.shape[0],-1),axis = 1).sum()/ original.shape[0]
    test_error = np.linalg.norm(normalized_original[-1] - normalized_reconstruction[-1])
    return total_error,test_error
    