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

from common import datasets as data
from common import constants as const
from common import pickling
from SHMAX.models import SHMAX
from localmlp.layers import LocalSHMAX 


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
        model_name = config.get(const.MODEL_NAME)
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
        return layers


    layers = load_model(config)

    # Forwarding
    all_responses = [] 
    responses = [[]*len(layers)]
    frequencies = []

    for i,(cgrams,freqs) in enumerate(dataloader.loader):
        cgrams.unsqueeze(1)
        temp = cgrams
        print(i)
        for ilayer in range(len(layers)):    
            temp = layers[ilayer].forward(temp)[:,:,:,30]
            responses[ilayer].append(temp)
        frequencies.append(freqs)
    freq_axis = torch.cat(frequencies,0)
    for ilayer in range(len(layers)):
        responses[ilayer] = torch.cat(responses[ilayer],0)

    # save the responses
    tosave = {'freq_axis':freq_axis,'outs':responses}
    responses_file = config.get(const.SAVE_EXP)
    with open(responses_file,'wb') as f:
        pickle.dump(tosave,f)
        print(f"{len(responses)} response tensors saved at {responses_file}")

