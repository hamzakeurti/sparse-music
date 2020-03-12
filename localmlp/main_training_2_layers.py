import json
import argparse
import numpy as np
import torch
import datetime

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from sparse_music.localmlp.layers import LocalSHMAX
from sparse_music.common import datasets as data
from sparse_music.common import constants as const

if __name__ == '__main__':

    print(datetime.datetime.today())

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config = json.load(f)
    

    # DATA LOADING
    # data_iterator = data.DataIterator(dataset = dataset, batch_size = batch_size, directory= directory,normalize=True)
    data_iterator = data.DataIterator.from_config(config[const.DATA])
    print('\nData loader successfully initiated\n----------------------')

    # MODEL INSTANCIATION (Layer1)
    save_directory = "/home/hamza/data3/projects/sparse_music/experiments/exp3/pickles/"
    layer1 = LocalSHMAX.from_saved(save_directory)
    layer2 = LocalSHMAX.from_config(config[const.MODEL])
    print('\nModel successfully initiated\n---------------------')

    # PATCHING
    with data_iterator.dataset:
        for i, elem in enumerate(data_iterator.loader):
            fm = layer1.forward(elem)
            layer2.extract_patches(fm)
            if len(layer2.patches[0])*layer2.patch_batch > config[const.TRAINING][const.MAX_PATCHES]:
                break
    print(f'\nExtracted {len(layer2.patches[0])*layer2.patch_batch} patches\n---------------------')
    # TRAINING
    layer2.train()

    # Forwarding

    # Saving
    layer2.save(config[const.TRAINING][const.SAVE_DIR])
    print(f'\nSaved model at {config[const.TRAINING][const.SAVE_DIR]}\n---------------------')




