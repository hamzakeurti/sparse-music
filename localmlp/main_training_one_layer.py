import json
import argparse
import numpy as np
import torch
import datetime

from localmlp.layers import LocalMlpLayer

import data.datasets as data

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
    

    # DATA LOADING
    dataset = 'cochleagram'
    batch_size = 20
    directory = data.MUSICNET_COCH_DIRECTORY

    data_iterator = data.DataIterator(dataset = dataset, batch_size = batch_size, directory= directory,normalize=True)
    print('\nData loader successfully initiated\n----------------------')

    # MODEL INSTANCIATION
    filters_shape = (50,5)
    dict_size = 50
    batch_size = 50
    band_indices = [0,50,100,150,200]
    model = LocalMlpLayer(filters_shape=filters_shape,dict_size=dict_size)
    print('\nModel successfully initiated\n---------------------')

    # WARM START (OPTIONAL)


    # PATCHING


    # TRAINING


    print('Starting training')
    with data_loader.dataset:
        for i, elem in enumerate(data_loader.loader):
            if iteration:
                i = i + iteration
            if i > max_iterations:
                break
            print(f'iteration {i}')

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



