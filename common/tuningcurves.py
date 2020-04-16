import numpy as np
import matplotlib.pyplot as plt

musical_ratios2 = np.array([16/15,9/8,6/5,5/4,4/3,3/2,8/5,5/3,16/9]) # list from 3Blue1Brown video on music and measure
musical_ratios = 1/musical_ratios2

octave_ratios2 = np.array([2,3,4,5,6])
octave_ratios = 1/octave_ratios2

harmonic_ratios2 = np.array([5/3,7/4,7/5,9/5,7/6,9/7])
harmonic_ratios = 1/harmonic_ratios2


def gaussian(window, sigma):
    return np.exp(-((np.arange(window) - window//2) ** 2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)


def smooth_curves(tuning_curves, window, sigma):
    """"""
    ret=[]
    smoothing_filter=gaussian(window, sigma)
    if len(tuning_curves.shape) == 2:
        for i in range(tuning_curves.shape[0]):
            ret.append(np.convolve(tuning_curves[i], smoothing_filter, 'same'))
    if len(tuning_curves.shape) == 3:
        for i in range(tuning_curves.shape[0]):
            ret.append(smooth_curves(tuning_curves[i], window, sigma))
    ret=np.stack(ret)
    return ret


def peaks(arr, n, threshold=None, percentile=80):
    out_arr = np.zeros(arr.shape,dtype=bool)
    if threshold is None and percentile is not None:
        print(percentile)
        # find threshold from distribution percentile
        threshold=np.percentile(arr[arr>0.01], percentile)
    if threshold is not None:
        print(f'threshold:{threshold}')
        out_arr[..., n+1:-(n+1)]=arr[..., n+1:-(n+1)] > threshold
    out_arr[..., n+1:-(n+1)] = True
    for i in range(1, n):
        out_arr[..., n+1:-(n+1)] &= (arr[..., n+1:-(n+1)] > arr[..., n+1-i:-(n+1)-i])
        out_arr[..., n+1:-(n+1)] &= (arr[..., n+1:-(n+1)] > arr[..., n+1+i:-(n+1)+i])
    return out_arr

def consecutive_peaks_ratios(peaks_arr, freqs_arr, min_n_peaks=2, max_n_peaks=4):
    ratios=[]
    for n_peaks in range(min_n_peaks, max_n_peaks):
        n_peaked=freqs_arr[np.where(peaks_arr[peaks_arr.sum(-1) == n_peaks])[1]]
        for i in range(int(len(n_peaked)/n_peaks)):
            for j in range(n_peaks-1):
                ratios.append(n_peaked[n_peaks*i+j]/n_peaked[n_peaks*i+j+1])

    return np.array(ratios)


def all_peaks_ratios(peaks_arr, freqs_arr, min_n_peaks=2, max_n_peaks=4):
    ratios=[]
    for n_peaks in range(min_n_peaks, max_n_peaks):
        n_peaked=freqs_arr[np.where(peaks_arr[peaks_arr.sum(-1) == n_peaks])[1]]
        for i in range(int(len(n_peaked)/n_peaks)):
            for j in range(1,n_peaks):
                for k in range(j):
                    ratios.append(n_peaked[n_peaks*i+k]/n_peaked[n_peaks*i+j])
    return np.array(ratios)

def plot_ratios_histogram(ratios,bins=25,alpha=0.5,save_file=None):

    xmusical = musical_ratios2
    xharmonics = harmonic_ratios2
    xoctave = octave_ratios2
    if ratios[0]<1:
        xharmonics = 1/xharmonics
        xmusical = 1/xmusical
        xoctave = 1/xoctave

    xharmonics = xharmonics[((xharmonics-0.1)>ratios.min()) & ((xharmonics + 0.1) < ratios.max())]
    xmusical = xmusical[((xmusical-0.1)>ratios.min()) & ((xmusical + 0.1) < ratios.max())]
    xoctave = xoctave[((xoctave-0.1)>ratios.min()) & ((xoctave + 0.1) < ratios.max())]

 
    plt.figure()
    ret = plt.hist(ratios,bins=bins,alpha = alpha)
    plt.xlabel('Consecutive peaks ratios')
    plt.ylabel('Count')
    for xc in xharmonics[:-1]:
        plt.axvline(x=xc,linestyle = '--',alpha = 0.8)
    plt.axvline(x=xharmonics[-1],linestyle = '--', alpha = 0.8,label='other harmonic ratios')

    for xc in xmusical[:-1]:
        plt.axvline(x=xc,color = 'r',linestyle = '--',alpha = 0.8)
    plt.axvline(x=xmusical[-1],color = 'r', linestyle = '--', alpha = 0.8,label='musical ratios')

    for xc in xoctave[:-1]:
        plt.axvline(x=xc,color = 'g',linestyle = '--',alpha = 0.8)
    plt.axvline(x=xoctave[-1],color = 'g',linestyle = '--', alpha = 0.8,label='octave ratios')


    plt.legend()
    if save_file:
        plt.savefig(save_file)
        plt.close()
    return ret