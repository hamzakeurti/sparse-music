import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json
import argparse
import pickle
import datetime
import numpy as np
import torch

from sparse_music.common import constants as const
from sparse_music.common import tuningcurves as tc

if __name__ == '__main__':

    print(datetime.datetime.today())

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config = json.load(f)

    curves_dir = config.get(const.CURVES_DIR,None)
    if not curves_dir:
        curves_dir = config[const.SAVE_EXP]
    curves_dir = os.path.dirname(curves_dir)
    
    files = os.listdir(curves_dir)
    response_files = []
    for f in files:
        if f.startswith('ptresponses'):
            response_files.append(f)
    
    # Load response curves
    curves = []
    for rf in response_files:
        with open(os.path.join(curves_dir,rf),'rb') as f:
            responses = pickle.load(f)
            freqs = responses['freq_axis']
            curves.append(responses['outs'][0])
    
    if torch.is_tensor(curves[0]):
        for i in range(len(curves)):
            curves[i] = curves[i].numpy()
        freqs = freqs.numpy()
    if freqs.shape[0] != curves[0].shape[-1]:
        for i in range(len(curves)):
            curves[i] = np.moveaxis(curves[i],0,-1)
            

    # Curves smoothing
    print("curves smoothing")
    smoothed_curves = []
    for curve in curves:
        smoothed_curves.append(tc.smooth_curves(curve,config.get(const.WINDOW,31),config.get(const.SIGMA,5)))
        print(smoothed_curves[-1].shape)
    
    # peaks
    print("peaks")
    n = config.get(const.NEIGHBORHOOD_SIZE,6)
    threshold = None
    percentile = None
    if const.THRESHOLD in config:
        threshold = config[const.THRESHOLD]
    else:
        percentile = config.get(const.PERCENTILE,80)
    all_peaks = []
    for curve in smoothed_curves:
        all_peaks.append(tc.peaks(curve,n,threshold=threshold,percentile=percentile))
    
    # ratios
    print('ratios')
    ratios = []
    ratios_func = None
    if config.get(const.ALL_PEAKS,False):
        ratios_func = tc.all_peaks_ratios
    else:
        ratios_func = tc.consecutive_peaks_ratios

    for peaks in all_peaks:
        if config.get(const.PROPER_FRACTIONS,False):
            ratios.append(ratios_func(peaks,freqs,min_n_peaks=config.get(const.MIN_N_PEAKS,2),max_n_peaks=config.get(const.MAX_N_PEAKS,4)))
        else:
            ratios.append(1/ratios_func(peaks,freqs,min_n_peaks=config.get(const.MIN_N_PEAKS,2),max_n_peaks=config.get(const.MAX_N_PEAKS,4)))
    # save
    
    save_dir = os.path.dirname(config[const.SAVE_EXP])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(len(curves)):
        filename = f"smoothed{i}.p"
        tosave = {'freq_axis':freqs,'curves':smoothed_curves[i]}
        
        with open(os.path.join(save_dir,filename),'wb') as f:
            pickle.dump(tosave,f)
        print(f'smoothed curves saved at {filename}')

        filename = f"peaks{i}.p"
        tosave = {'freq_axis':freqs,'peaks':all_peaks[i],'ratios':ratios[i]}
        with open(os.path.join(save_dir,filename),'wb') as f:
            pickle.dump(tosave,f)
        print(f'peaks, ratios saved at {filename}')



    # # histograms
    for i in range(len(ratios)):
        tc.plot_ratios_histogram(ratios[i],bins=config.get(const.BINS,25),save_file=os.path.join(save_dir,f"histogram{i}.jpg"))