# %% [markdown]
# ##imports!!!

# %%

# code and ideas in this document are adopted from https://neurodsp-tools.github.io/neurodsp/
# Thank you to all those who came before me, for your contributions.
# -Kenton Guarian

# general imports
from ref_functions import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from neurodsp.utils import create_times
from neurodsp.utils import create_freqs

# imports for simulating periodic component
from neurodsp.sim import sim_bursty_oscillation
from neurodsp.sim import sim_variable_oscillation


# for simulating aperiodic component.
from neurodsp.sim import sim_powerlaw

# for plotting
from neurodsp.plts import plot_time_series
from neurodsp.plts import plot_power_spectra
from neurodsp.spectral import compute_spectrum
import matplotlib.pyplot as plt

# for bycycle analysis
import numpy as np
import sys
# from scipy.interpolate import make_interp_spline
from scipy.signal import resample
sys.path.insert(0, '/Users/kenton/HOME/coding/python/bycycle_env/bycycle')

import mycycle

# %matplotlib inline


# %% [markdown]
# Constants

# %%

n_seconds = 10
fs = 500
sliding_window_width=3
times = create_times(n_seconds=n_seconds, fs=fs)
freqs = create_freqs(freq_start=0, freq_stop=60)


thresholds = {
    'amp_fraction': 0.5,
    'amp_consistency': 0.2,
    'period_consistency': 0.2,
    'monotonicity': 0.2,
    'min_n_cycles': 1
}

# %% [markdown]
# simulating signals.

# %%
# bursts and signals taken from tutorial pages.
burst0 = sim_bursty_oscillation(n_seconds=n_seconds, fs=fs, freq=10, burst_def='durations', burst_params={
    'n_cycles_burst': 3, 'n_cycles_off': 3})
burst1 = sig = sim_bursty_oscillation(n_seconds=n_seconds, fs=fs, freq=10, burst_params={
    'enter_burst': 0.2, 'leave_burst': 0.8})
burst2 = sim_bursty_oscillation(n_seconds=n_seconds, fs=fs, freq=10, burst_def='durations',
                                burst_params={'n_cycles_burst': 3, 'n_cycles_off': 5})
burst3 = sim_bursty_oscillation(n_seconds=n_seconds, fs=fs, freq=10, burst_def='durations', burst_params={
    'n_cycles_burst': 8, 'n_cycles_off': 20})
burst4 = sig = sim_bursty_oscillation(n_seconds=n_seconds, fs=fs, freq=10, burst_params={
    'enter_burst': 0.2, 'leave_burst': 0.8})
burst5 = sim_bursty_oscillation(n_seconds=n_seconds, fs=fs, freq=10, burst_def='durations',
                                burst_params={'n_cycles_burst': 3, 'n_cycles_off': 3})

sig0 = sim_powerlaw(n_seconds=n_seconds, fs=fs, exponent=-2.0)

sig1 = sim_powerlaw(n_seconds=n_seconds, fs=fs,
                    exponent=-1.5, f_range=(2, None))
sig2 = sim_powerlaw(n_seconds=n_seconds, fs=fs, exponent=-0.5)

sig3 = sim_powerlaw(n_seconds=n_seconds, fs=fs,
                    exponent=-3, f_range=(2, None))


bursts = [burst3, burst5]
sigs = [sig0, sig1, sig2, sig3]
ratios = [10, 1, 0.5, 0.1, 0.0]

combined_sigs = [None]*(len(bursts) * len(sigs)*len(ratios))

# for y_vals in all_to_plot:
#     plot_time_series(times=times, sigs=y_vals)
# tested, looks good.

for i in range(len(bursts)):
    for j in range(len(sigs)):
        for k in range(len(ratios)):
            combined_sigs[len(sigs)*len(ratios)*i + len(ratios)
                          * j+k] = bursts[i]+10*ratios[len(ratios)-k-1]*sigs[j]
            

bm = mycycle.Bycycle()

# %% [markdown]
# Separate simulated signals based on known burst/aperiodic differences.

# %%
# TODO: do this.

# %% [markdown]
# K-Means on hyperparameters

# %%
# for sig in combined_sigs:
#     keys, hyperparams = extract_cleaned_hyperparameters_from_signal(bm, sig,500,3)
#     scaler = preprocessing.StandardScaler().fit(hyperparams)
#     print(keys)
#     print(hyperparams)
#     hyperparams_standardized = scaler.transform(hyperparams)
#     kmeans_model = KMeans(n_clusters=2, random_state=0, n_init="auto")
#     kmeans = kmeans_model.fit_predict(hyperparams_standardized.T)
#     sig_window_indices = create_window_indices_from_signal(bm, sig, 500, 3)
#     sig_windows = get_signal_windows(sig, sig_window_indices)
#     sig_windows = sig_windows[1:len(sig_windows)-1]

#     group0 = []
#     group1 = []
#     for i in range(len(kmeans)):
#         if kmeans[i]==0:
#             group0.append(sig_windows[i-1])
#         else:
#             group1.append(sig_windows[i-1])

#     print(len(group0))
#     print(len(group1))
#     plt.figure()
#     for i in range(len(group0)):
#         plt.plot(np.linspace(0,len(group0[i]),len(group0[i])), group0[i], alpha=0.5)
#     print(len(group0))
#     plt.figure()
#     for i in range(len(group1)):
#         plt.plot(np.linspace(0,len(group1[i]),len(group1[i])), group1[i], alpha=0.5)
#     print(len(group1))
#     plt.show()


# %% [markdown]
# Add autocorrelation hyperparameters to the features NDarray

# %%
for sig in combined_sigs:
    # extract hypeparams
    keys, hyperparams = extract_cleaned_hyperparameters_from_signal(bm, sig,500,3)
    # print(keys)
    # print(hyperparams)
    sig_window_indices = create_window_indices_from_signal(bm, sig, 500, 3)
    sig_windows = get_signal_windows(sig, sig_window_indices)
    # sig_windows = sig_windows[1:len(sig_windows)-1]
    corr = autocorrelate_all_windowed_signals(sig_windows)
    longest_corr = 0
    for i in range(len(corr)):
        currcorrlen = len(corr[i])
        if currcorrlen>longest_corr:
            longest_corr = currcorrlen

    hp_size = len(hyperparams)
    hp0_size = len(hyperparams[0])
    new_hyperparams = [np.zeros(hp0_size+longest_corr)]*hp_size
    for i in range(len(hyperparams)):
        # print("i=%d"%i)
        for j in range(hp0_size):
            new_hyperparams[i][j] = hyperparams[i][j]
        if i>0 and i < len(corr)-1:
            for j in range(len(corr[i-1])):
                array_idx = hp0_size+j
                new_hyperparams[i][array_idx]=corr[i-1][j]
        

    # print(len(new_hyperparams))
    # print(hyperparams.size)
    # standardize for k-means. take out and do for whole including autocorr
    scaler = preprocessing.StandardScaler().fit(new_hyperparams)
    new_hyperparams_standardized = scaler.transform(new_hyperparams)
    kmeans_model = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans = kmeans_model.fit_predict(new_hyperparams_standardized)

    # print(new_hyperparams_standardized)
    # print(new_hyperparams)

    # 
    group0 = []
    group1 = []
    group2 = []
    
    for i in range(1,len(kmeans)-1):
        if kmeans[i]==0:
            group0.append(sig_windows[i-1])
        elif kmeans[i]==1:
            group1.append(sig_windows[i])
        elif kmeans[i]==2:
            group2.append(sig_windows[i])


    # print(len(new_hyperparams_standardized.shape))
    print(len(group0))
    print(len(group1))
    print(len(group2))
    plt.figure()
    for i in range(len(group0)):
        plt.plot(np.linspace(0,len(group0[i]),len(group0[i])), group0[i], alpha=0.5)
    # print(len(group0))
    plt.figure()
    for i in range(len(group1)):
        plt.plot(np.linspace(0,len(group1[i]),len(group1[i])), group1[i], alpha=0.5)
    # print(len(group1))
    plt.figure()
    for i in range(len(group2)):
        plt.plot(np.linspace(0,len(group2[i]),len(group2[i])), group2[i], alpha=0.5)
    # print(len(group2))
    # plt.show()

