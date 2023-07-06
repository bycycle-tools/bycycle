import sys
# for bycycle analysis
sys.path.insert(0, '/Users/kenton/HOME/coding/python/bycycle_env/bycycle')
import sys
import mycycle
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf as autocorrelate


def create_window_indices_from_signal(bm=None,sig=None, fs=500, window_length=3):
    bm.fit(sig=sig, fs=fs, f_range=(8, 12))
    last_troughs = bm.df_features["sample_last_trough"]
    next_troughs = bm.df_features["sample_next_trough"]
    # print(last_troughs[len(last_troughs)-1])
    # print(next_troughs[len(last_troughs)-2])
    window_bound_collection = [None]*(len(last_troughs)-window_length)
    for i in range(window_length, len(last_troughs)):
        window = (last_troughs[i-window_length],next_troughs[i-1])
        window_bound_collection[i-window_length]=window
    return window_bound_collection

def get_signal_windows(sig, window_idx_collection):
    collection = [None]*len(window_idx_collection)
    for i in range(len(window_idx_collection)):
        collection[i]=sig[window_idx_collection[i][0]:window_idx_collection[i][1]]

    for i in collection:
        plt.plot(np.linspace(0,len(i), len(i)),i)

    # plt.show()
    return collection

def extract_cleaned_hyperparameters_from_signal(bm=None,sig=None, fs=500, window_length=3):
    bm.fit(sig=sig, fs=fs, f_range=(8, 12))
    features = bm.df_features
    keys = features.keys()
    np_features = features.to_numpy()
    subindex_length = len(np_features[0])
    # we actually won't be using this
    for i in range(len(np_features)):
        for j in range(subindex_length):
            if not (np.isfinite(np_features[i][j])):
                np_features[i][j]=-1
    
    np_features=np_features[1:len(np_features-2)]
    return keys, np_features

def autocorrelate_signal_wrong(sig):
    slen = len(sig)
    shifts=[sig]*slen
    for i in range(slen):
        for j in range(slen):
            shifts[i][j]=sig[(j+i) % slen]
    df=pd.DataFrame(shifts)
    corr = df.corr(method='pearson')
    return corr

def autocorrelate_signal(sig):
    return autocorrelate(sig)

def autocorrelate_all_windowed_signals(sig_window_collection):
    result_length = len(sig_window_collection)
    result = [None]*result_length
    for i in range(result_length):
        result[i]=autocorrelate_signal(sig_window_collection[i])
    return result
