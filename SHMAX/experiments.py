import matplotlib.pyplot as plt
import os
import numpy as np
import torch

import stimulus
import utils
import pickling

NAME = "name"
HARMONICS_INDEX = "harmonics_index"
JITTERING = "jittering"
FILTERS = "filters"


def new_fig():
    fig, axes = plt.subplots()
    return axes

def plot_filters(dictionary,channel_in,channel_out):
    if channel_in == None:
        filters = dictionary[channel_out,:]
    elif channel_out == None:
        filters = dictionary[:,channel_in]
    else:
        return
    vmax = filters.max()
    vmin = filters.min()
    cols = 10
    lines = filters.shape[0] // cols + 1
    for i in range(filters.shape[0]):
        plt.subplot(lines,cols,i+1)
        plt.imshow(filters[i],interpolation='gaussian',cmap='gray',vmin=vmin,vmax=vmax)
        plt.tick_params(labelbottom=False,labelleft=False,left=False,bottom=False)

def plot_model_filters(model,save_dir):
    for layer in range(model.n_layers):
        save_subdir = os.path.join(save_dir,str(layer))
        if not os.path.exists(save_subdir):
            os.mkdir(save_subdir)
        for channel_in in range(model.s_layers[layer].dictionary.shape[1]):
            new_fig()
            plot_filters(model.s_layers[layer].dictionary,channel_in,None)
            plt.savefig(os.path.join(save_subdir,f'from_{channel_in}'))
            plt.close()

def do_experiment(model,exp_config,window,sr,dirname):
    save_dir = os.path.join(dirname,exp_config[NAME])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if FILTERS in exp_config[NAME]:
        plot_model_filters(model,save_dir)
        return
    freqs = 21.85*np.power(2,1/12)**np.linspace(0,100,100,endpoint=False)    
    harmonics_index = exp_config.get(HARMONICS_INDEX,[1])
    jittering = exp_config.get(JITTERING,0)
    indices = stimulus.jitter_indices(harmonics_index,jittering)
    stimuli = np.array([stimulus.complex_tone(f,indices,window,sr) for f in freqs])
    coch_stimuli  = utils.wav2cgram(stimuli,sr)
    coch_stimuli = np.expand_dims(coch_stimuli,1)

    coch_stimuli = torch.tensor(coch_stimuli)

    tensors = model.forward(coch_stimuli)
    for layer in range(model.n_layers):
        s_index = 2*layer + 1
        c_index = 2*layer + 2
        #               ------- S_layer -------------------------------------
        save_subdir = os.path.join(save_dir,f's_layer{layer}')
        if not os.path.exists(save_subdir):
            os.mkdir(save_subdir)
        for channel in range(tensors[s_index].shape[1]):
            new_fig()
            plt.imshow(tensors[s_index][:,channel,:,:].numpy().mean(axis=-1).T)
            plt.savefig(os.path.join(save_subdir,f'channel{channel}'))
            plt.close()
        #               ---------- C_layer ------------------------------------
        save_subdir = os.path.join(save_dir,f'c_layer{layer}')
        if not os.path.exists(save_subdir):
            os.mkdir(save_subdir)
        for channel in range(tensors[c_index].shape[1]):
            new_fig()
            plt.imshow(tensors[c_index][:,channel,:,:].numpy().mean(axis=-1).T)
            plt.savefig(os.path.join(save_subdir,f'channel{channel}'))
            plt.close()
    pickling.save_tensor(tensors,os.path.join(save_dir,'tensors.p'))

