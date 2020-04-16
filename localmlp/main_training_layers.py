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
    save_directory = config.get(const.PICKLES_DIRECTORY)
    layers = []
    for i in range(len(save_directory)):
        layers.append(LocalSHMAX.from_saved(save_directory[i]))
    layer = LocalSHMAX.from_config(config[const.MODEL])
    print('\nModel successfully initiated\n---------------------')

    # PATCHING
    with data_iterator.dataset:
        for i, elem in enumerate(data_iterator.loader):
            if len(layer.patches[0])*layer.patch_batch > config[const.TRAINING][const.MAX_PATCHES]:
                break
            fm = elem
            for i in range(len(layers)):
                fm = layers[i].forward(fm)
                # print(f'forwarded through layer{i}')
            layer.extract_patches(fm)
            print(f'extracted {len(layer.patches[0])*layer.patch_batch}patches / {config[const.TRAINING][const.MAX_PATCHES]}')
    print(f'\nExtracted {len(layer.patches[0])*layer.patch_batch} patches\n---------------------')
    # TRAINING
    layer.train()


    # Saving
    layer.save(config[const.TRAINING][const.SAVE_DIR])
    print(f'\nSaved model at {config[const.TRAINING][const.SAVE_DIR]}\n---------------------')




