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

    # MODEL INSTANCIATION
    # filters_shape = (50,5)
    # dict_size = 50
    # batch_size = 50
    # band_indices = [0,50,100,150,200]
    model = LocalSHMAX.from_config(config[const.MODEL])
    print('\nModel successfully initiated\n---------------------')

    # WARM START (OPTIONAL)


    # PATCHING
    with data_iterator.dataset:
        for i, elem in enumerate(data_iterator.loader):
            if len(model.patches[0])*model.patch_batch > config[const.TRAINING][const.MAX_PATCHES]:
                break
            model.extract_patches(elem)
    print(f'\nExtracted {len(model.patches[0])*model.patch_batch} patches\n---------------------')
    # TRAINING
    model.train()

    # Forwarding

    # Saving
    model.save(config[const.TRAINING][const.SAVE_DIR])
    print(f'\nSaved model at {config[const.TRAINING][const.SAVE_DIR]}\n---------------------')




