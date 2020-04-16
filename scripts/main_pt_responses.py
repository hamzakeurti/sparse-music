# this script is to produce pure tones tuning curves for a trained model


# Add sparse_music module to be discoverable
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# imports
import argparse
import json
import datetime
import pickle
import torch

from sparse_music.common import datasets as data
from sparse_music.common import constants as const
from sparse_music.common import pickling
from sparse_music.SHMAX.models import SHMAX
from sparse_music.localmlp.layers import LocalSHMAX 


# from sparse_music.common import tuningcurves as tc 

if __name__ == '__main__':

    print(datetime.datetime.today())

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config = json.load(f)
    
    # Data iterator
    directory = '/data3/hamza/datasets/synthesized_tones/pt/'
    dataloader = data.DataIterator(dataset = 'frequencies',batch_size = 100,directory = directory,shuffle=False)

    # Load model
    def load_model(config):
        model_name = config[const.MODEL][const.MODEL_NAME]
        if model_name == const.SHMAX:
            model = SHMAX.from_config(config['model'])
            iteration = 0
            # if not config[const.WARM_START].get(const.PICKLES_DIRECTORY,""):
            #     experiment_config[pickling.WARM_START][pickling.PICKLES_DIRECTORY] = dirname
            pickling.update_model_from_config(model,config[pickling.WARM_START])
            iteration = config[pickling.WARM_START][pickling.ITERATION]
            print('\nwarm start model loaded at iteration '+str(iteration)+'\n----------')
            layers = model.s_layers
        if model_name == const.LOCAL_SHMAX:
            pickles_dir = config[const.MODEL][const.PICKLES_DIRECTORY]
            layers = []
            for directory in pickles_dir:
                layers.append(LocalSHMAX.from_saved(directory))
                print(layers[-1].band_blocks[0].dict_tensor.shape)
        return layers


    layers = load_model(config)

    # Forwarding
    all_responses = [] 
    responses = [[] for i in range(len(layers))]
    frequencies = []

    for i,(cgrams,freqs) in enumerate(dataloader.loader):
        cgrams = cgrams.unsqueeze(1)
        temp = cgrams
        for ilayer in range(len(layers)):    
            temp = layers[ilayer].forward(temp)
            # temp = temp[...,temp.shape[-1]//2]
            # if ilayer == 1:
            responses[ilayer].append(temp)
        frequencies.append(freqs)

    # del responses[0]
    freq_axis = torch.cat(frequencies,0)
    for ilayer in range(len(layers)):
        # if ilayer == 0:
        responses[ilayer] = torch.cat(responses[ilayer],0)[...,responses[ilayer][0].shape[-1]//2]

    # save the responses
    responses_file = config.get(const.CURVES_DIR,None)
    if not responses_file:
        responses_file = config[const.SAVE_EXP]

    dirname = os.path.dirname(responses_file)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    
    for i in range(len(responses)):
        tosave = {'freq_axis':freq_axis,'outs':[responses[i]]}
        print(f'trying to save responses of shape {responses[i].shape}')
        with open(os.path.join(dirname,f'ptresponses{i}.p'),'wb') as f:
            pickle.dump(tosave,f)
            print(f"response tensors saved at ptresponses{i}.p")