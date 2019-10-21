import json
import argparse
import numpy as np
import torch

import data
from models import SHMAX
import pickling
import utils

TRAINING = 'training'
MAX_ITERATIONS = "max_iterations"
N_SUBITERATIONS = "n_subiterations"
SAVE_RATE = "save_rate"
SAVE_DIR = "save_dir"


if __name__ == '__main__':

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
        iteration = config[pickling.WARM_START][pickling.ITERATION]
        print('\nwarm start model loaded at iteration '+str(iteration)+'\n----------')

    # TRAINING
    max_iterations = config[TRAINING][MAX_ITERATIONS]
    n_subiterations = config[TRAINING][N_SUBITERATIONS]
    save_rate = config[TRAINING][SAVE_RATE]
    save_dir = config[TRAINING][SAVE_DIR]

    print('Starting training')
    with data_loader.dataset:
        for i, (x, y) in enumerate(data_loader.loader):
            if iteration:
                i = i + iteration
            if i > max_iterations:
                break
            print(f'iteration {i}')
            print(f'transforming signals into cochleagrams')
            cgrams = utils.wav2cgram(x.numpy(),sr=config['data'][data.SR])
            cgrams = np.expand_dims(cgrams,1)
            cgrams = torch.tensor(cgrams)

            for k in range(n_subiterations):
                print(f'sub-iteration {k}')
                model.forward_train(cgrams)

            if i % save_rate == 0 and i > 0:
                pickling.save_model(model,i,save_dir)
                print('Save performed at iteration ' + str(i))



