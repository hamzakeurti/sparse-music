import numpy as np


def gaussian(window, sigma):
    return np.exp(-((np.arange(window) - window//2) ** 2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)


def smooth_curves(tuning_curves, window, sigma):
    """"""
    ret=[]
    smoothing_filter=gaussian(window, sigma)
    if tuning_curves.shape == 2:
        for i in range(tuning_curves.shape[0]):
            ret.append(np.convolve(tuning_curves[i], smoothing_filter, 'same'))
    if tuning_curves.shape == 3:
        for i in range(tuning_curves.shape[0]):
            ret.append(smooth_curves(tuning_curves[i], window, sigma))
    ret=np.stack(ret)
    return ret


def peaks(arr, n, threshold=None, percentile=80):
    if not threshold:
        # find threshold from distribution percentile
        threshold=np.percentile(arr, percentile)
    out_arr=arr > threshold
    for i in range(1, n):
        out_arr[..., n+1:-(n+1)]=out_arr[..., n+1:-(n+1)] & (arr[..., n+1:-(n+1)] >
                                                               arr[..., n+1-i:-(n+1)-i]) & (arr[..., n+1:-(n+1)] > arr[..., n+1+i:-(n+1)+i])
    return out_arr

def consecutive_peaks_ratios(peaks_arr, freqs_arr, min_n_peaks=2, max_n_peaks=4):
    ratios=[]
    for n_peaks in range(min_n_peaks, max_n_peaks):
        n_peaked=freqs_arr[np.where(peaks_arr[peaks_arr.sum(1) == n_peaks])[1]]
        for i in range(int(len(n_peaked)/n_peaks)):
            for j in range(n_peaks-1):
                ratios.append(n_peaked[n_peaks*i+j]/n_peaked[n_peaks*i+j+1])
    return ratios
