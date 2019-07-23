import musicnet
import torch
import numpy as np
import matplotlib.pyplot as plt
import pycochleagram.cochleagram as cgram


from models import SparseLayer

root = 'D:/Projects/Tsinghua/Deep Learning/Project/music-learning2/data/musicnet'

sr = 44100
window = sr/4
batch_size = 5
downsample = window/25*4
m = 128

train_set = musicnet.MusicNet(root=root, train=True, window=window, mmap=False,download=True)#, pitch_shift=5, jitter=.1)
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size)





num_filters = 200
filter_size = (9,9)

s_layer = SparseCode(dictionary_size = num_filters,input_size = filter_size)

