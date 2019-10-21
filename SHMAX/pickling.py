import pickle
import os.path as path

WARM_START = 'warm_start'
D = 'D'
MODEL = 'model'
DICTIONARY = 'dictionary'
ITERATION = 'iteration'
PICKLES_DIRECTORY = 'pickles_dir'
SAVE_DIR = 'save_dir'


def update_from_pickle(layer,pickle_file):
    with open( pickle_file, "rb" ) as f:
        state = pickle.load(f)
    layer.model = state[MODEL]
    layer.param[D] = state[D]
    layer.dictionary = state[DICTIONARY]

def save_dict(dictionary,pickle_file):
    with open( pickle_file, "wb" ) as f:
        pickle.dump(dictionary,f)
        
def save_state(layer,pickle_file):
    state = {}
    state[MODEL] = layer.model
    state[D] = layer.param[D]
    state[DICTIONARY] = layer.dictionary
    with open( pickle_file, "wb" ) as f:
        pickle.dump(state,f)
    
def update_model_from_config(model,config):
    iteration = config[ITERATION]
    pickles_dir = config[PICKLES_DIRECTORY]
    for layer in range(model.n_layers):
        try:
            pickled_state = path.join(pickles_dir,f's_layer_{layer}_state{iteration}.p')
            update_from_pickle(model.s_layers[layer],pickled_state)
        except:
            print(f'updating from pickle file {pickled_state} failed for layer {layer}')
            continue

def save_model(model,iteration,save_dir):
    for layer in range(model.n_layers):
        try:
            pickled_state = path.join(save_dir,f's_layer_{layer}_state{iteration}.p')
            save_state(model.s_layers[layer],pickled_state)
        except KeyError:
            continue

def save_tensor(tensor,pickle_file):
    with open( pickle_file, "wb" ) as f:
        pickle.dump(tensor,f)