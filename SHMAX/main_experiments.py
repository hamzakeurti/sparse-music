import argparse
import json
import os

from models import SHMAX
import pickling
import data
import experiments

import datetime

EXPERIMENTS_LIST = "experiments_list"


if __name__ == '__main__':

    print(datetime.datetime.today())
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()

    dirname = os.path.dirname(args.experiment)
    config_path = os.path.join(dirname,"config.json")
    with open(args.experiment ,'r') as f:
        experiment_config = json.load(f)
    with open(config_path,'r') as f:
        config = json.load(f)    
    
    # MODEL INSTANCIATION
    model = SHMAX.from_config(config['model'])
    print('\nModel successfully initiated\n---------------------')

    # WARM START (REQUIRED)
    iteration = 0
    if not experiment_config[pickling.WARM_START].get(pickling.PICKLES_DIRECTORY,""):
        experiment_config[pickling.WARM_START][pickling.PICKLES_DIRECTORY] = dirname
    pickling.update_model_from_config(model,experiment_config[pickling.WARM_START])
    iteration = experiment_config[pickling.WARM_START][pickling.ITERATION]
    print('\nwarm start model loaded at iteration '+str(iteration)+'\n----------')

    sr = config["data"][data.SR]
    window = config["data"][data.WINDOW]
    # EXPERIMENTS
    for experiment in experiment_config[EXPERIMENTS_LIST]:
        print(f'Experiment {experiment[experiments.NAME]} started')
        experiments.do_experiment(model,experiment,window,sr,dirname)
    