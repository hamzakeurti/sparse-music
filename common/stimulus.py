import numpy as np

sr = 44100
window = sr

def pure_tone(freq,window=window,sr=sr):
    t = np.linspace(0,window,endpoint=False,num = window,dtype=int)/sr
    return np.sin(2*np.pi*freq*t)

def complex_tone(freq,harmonics_index,window=window,sr=sr):
    signal = pure_tone(freq*harmonics_index[0],window,sr)
    for i in range(1,len(harmonics_index)):
        signal = signal + pure_tone(freq*harmonics_index[i],window,sr)
    return signal/len(harmonics_index)

def jitter_indices(harmonics_index,jittering):
    jitters = []
    for k in harmonics_index:
        jitters.append((1+(1-2*np.random.random())*jittering)*k)
    return jitters