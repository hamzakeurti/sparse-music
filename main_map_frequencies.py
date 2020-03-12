from common import stimulus, utils
import numpy as np

import pickle

freqs = np.zeros(256)
values = np.zeros(256)

warm_start = '/home/hamza/data3/projects/sparse_music/experiments/frequencies_map/list8.p'
with open(warm_start,'rb') as f:
    data = pickle.load(f)

freqs = data['freqs']
values = data['values']

for f in range(550,800,3):
    print(f)
    stim = stimulus.pure_tone(freq=f)
    cgram = utils.wav2cgram(stim,44100)
    val = max(cgram[:,20])
    index = list(cgram[:,20]).index(val)
    if values[index] < val:
        freqs[index] = f
        values[index] = val

filename = "/home/hamza/data3/projects/sparse_music/experiments/frequencies_map/list9.p"
with open(filename,'wb') as f:
    pickle.dump({'freqs':freqs,'values':values},f)