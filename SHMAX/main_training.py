import json
import argparse
import numpy as np
import torch
import datetime

import data
from models import SHMAX
import pickling
import utils
import musicnet

TRAINING = 'training'
MAX_ITERATIONS = "max_iterations"
N_SUBITERATIONS = "n_subiterations"
SAVE_RATE = "save_rate"
SAVE_DIR = "save_dir"


if __name__ == '__main__':

    print(datetime.datetime.today())

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config = json.load(f)
    
    print(config)
    # DATA LOADING
    data_loader = data.DataIterator.from_config(config['data'])
    print('\nData loader successfully initiated\n----------------------')

    # MODEL INSTANCIATION
    config['model'][N_SUBITERATIONS] = config[TRAINING][N_SUBITERATIONS]
    model = SHMAX.from_config(config['model'])
    print('\nModel successfully initiated\n---------------------')

    # WARM START (OPTIONAL)
    iteration = 0
    if pickling.WARM_START in config.keys():
        pickling.update_model_from_config(model,config[pickling.WARM_START])
        iteration = config[pickling.WARM_START][pickling.ITERATION] + 1
        print('\nwarm start model loaded at iteration '+str(iteration)+'\n----------')

    # TRAINING
    max_iterations = config[TRAINING][MAX_ITERATIONS]
    n_subiterations = config[TRAINING][N_SUBITERATIONS]
    save_rate = config[TRAINING][SAVE_RATE]
    save_dir = config[TRAINING][SAVE_DIR]

    print('Starting training')
    with data_loader.dataset:
        for i, elem in enumerate(data_loader.loader):
            if iteration:
                i = i + iteration
            if i > max_iterations:
                break
            print(f'iteration {i}')
            if isinstance(data_loader.dataset,musicnet.MusicNet):
                print(f'transforming signals into cochleagrams')
                cgrams = utils.wav2cgram(elem[0].numpy(),sr=config['data'][data.SR])
            else:
                cgrams = elem
            cgrams = np.expand_dims(cgrams,1)
            cgrams = torch.tensor(cgrams,dtype=torch.float)

            # TODO Why subiterations. Why not batch_size = n_subiters*batch_size??
            for k in range(n_subiterations):
                print(f'sub-iteration {k}')
                model.forward_train(cgrams)

            if i % save_rate == 0 and i > 0:
                pickling.save_model(model,i,save_dir)
                print('Save performed at iteration ' + str(i))



