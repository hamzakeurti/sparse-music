import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json
import argparse
import pickle
import datetime

from sparse_music.common import constants as const
from sparse_music.common import tuningcurves as tc

if __name__ == '__main__':

    print(datetime.datetime.today())

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config = json.load(f)

    # Load response curves
    with open(config[const.SAVE_EXP],'rb') as f:
        responses = pickle.load(f)
        freqs = responses['freq_axis']
        curves = responses['outs']
    
    # Curves smoothing
    smoothed_curves = []
    for curve in curves:
        smoothed_curves.append(tc.smooth_curves(curve,config.get(const.WINDOW,31),config.get(const.SIGMA,5)))
    
    # peaks
    n = config.get(const.NEIGHBORHOOD_SIZE,6)
    threshold = None
    percentile = None
    if const.THRESHOLD in config:
        threshold = config[const.THRESHOLD]
    if const.PERCENTILE in config:
        percentile = config[const.PERCENTILE]
    all_peaks = []
    for curve in smoothed_curves:
        all_peaks.append(tc.peaks(curve,n,threshold=threshold,percentile=percentile))

    # ratios
    ratios = []
    for peaks in all_peaks:
        ratios.append(1/tc.consecutive_peaks_ratios(peaks,freqs,min_n_peaks=config.get(const.MIN_N_PEAKS,2),max_n_peaks=config.get(const.MAX_N_PEAKS,4)))
    
    # save
    save_dir = os.path.dirname(config[const.SAVE_EXP])
    smoothed_file = os.path.join(save_dir,"smoothed.p")
    tosave = {'freq_axis':freqs,'curves':smoothed_curves,'peaks':all_peaks,'ratios':ratios}
    with open(smoothed_file,'wb') as f:
        pickle.dump(tosave,f)
    print(f'smoothed curves, peaks, ratios saved at {smoothed_file}')

