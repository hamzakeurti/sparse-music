import json
import argparse
import numpy as np
import torch
import datetime
import os
from tqdm import tqdm


import data
import pickling
import utils

MUSICNET_DIRECTORY = '/data/valentin/music-learning/musicnet'
COCHLEAGRAMS_DIR = '/data/hamza/datasets/musicnet/cochleagrams/'

DOWNLOAD = 'download'
JITTER = 'jitter'
PITCH_SHIFT = 'pitch_shift', 
WINDOW = 'window'
SR = 'sampling_rate'
SAMPLE_FACTOR = "sample_factor"
NON_LINEARITY = "non_linearity"
NUMBER_FILTERS = "n"
COCHLEAGRAM = 'cochleagram'

config = {
    "subfolder":"folder1",
    'data':{
        "dataset":"musicnet",
        "directory":MUSICNET_DIRECTORY,
        "batch_size":1,
        "sampling_rate":44100,
        "window":44100,
    },
    'cochleagram':{
        "n":50,
        "sample_factor":4,
        "non_linearity":"power"
    }
}
save_subdir = os.path.join(COCHLEAGRAMS_DIR,config['subfolder'])


def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1

if __name__ == '__main__':
    if not os.path.exists(save_subdir):
        os.mkdir(save_subdir)
    
    i1=0
    for f in os.listdir(save_subdir):
        f_i = int(f.split('.')[0])
        if i1 <= f_i:
            i1 = f_i+1
    print(datetime.datetime.today())

    print(config)
    # DATA LOADING
    data_loader = data.DataIterator.from_config(config['data'])
    print('\nData loader successfully initiated\n----------------------')


    print('Starting cochlea processing')
    with data_loader.dataset:
        for i, (x, y) in tqdm_enumerate(data_loader.loader):
            i = i+i1
            if i>50000:
                break
            print(f'processing iteration {i}')
            z = utils.wav2cgram(
                x.numpy(),
                sr = config['data'][SR],
                n=config[COCHLEAGRAM][NUMBER_FILTERS],
                sample_factor=config[COCHLEAGRAM][SAMPLE_FACTOR],
                non_linearity=config[COCHLEAGRAM][NON_LINEARITY])
            save_file = os.path.join(save_subdir,f'{i}.p')
            pickling.save_tensor(z,save_file)


