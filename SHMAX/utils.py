import numpy as np
from scipy import signal as sig
import torch

import pycochleagram.cochleagram as cgram 

def wav2cgram(sound,sr):
    x = cgram.cochleagram(sound,sr,50,30,7860,4,nonlinearity='power')
    y = []
    for i in range(x.shape[0]):
        y.append( cgram.apply_envelope_downsample(x[i],mode="poly",audio_sr=sr,env_sr=int(256*sr/x[i].shape[-1])))
        y[i] = sig.resample_poly(y[i],256,211)
    x = torch.tensor(y)
    return x


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
    