"""
5. Running Bycycle on 3D Arrays
===============================

Compute bycycle features for 3D organizations of timeseries.

Bycycle supports computing the features of 3D signals using :func:`~.compute_features_3d`.
Signals may be organized in a different ways, including (n_participants, n_channels, n_timepoints)
or (n_channels, n_epochs, n_timepoints). The difference between these organizations is that
continuity may be assumed across (n_participants, n_channels) but not (n_channels, n_epochs). The
``axis`` argument is used to specificy the axis to iterate over in parallel.
"""

####################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neurodsp.sim import sim_combined
from bycycle.group import compute_features_3d
from bycycle.plts import plot_feature_categorical

####################################################################################################
# The ``axis`` Argument
# ---------------------
#
# Here, we will show how the axis arguments works by iterating over slices of an 3D array.
# The axis argument be may specified as:
#
# - ``axis=0`` : Operate over 2D slices along the zeroth dimension, (i.e. for each channel in
#   (n_channels, n_epochs, n_timepoints)).
# - ``axis=1`` : Operates over 2D slices along the first dimension (i.e. across flatten epochs in
#   (n_epochs, n_timepoints)).
# - ``axis=(0, 1)`` : Operates along 1D slices along the zeroth and first dimension (i.e across
#   each signal independently in (n_participants, n_channels, n_timepoints)).
#

####################################################################################################

arr = np.ones((2, 3, 4))

xlen = np.shape(arr)[0]
ylen = np.shape(arr)[1]

print("axis=0")
for xdim in range(xlen):
    print(np.shape(arr[xdim]))

print("\naxis=1")
for ydim in range(ylen):
    print(np.shape(arr[:, ydim]))

print("\naxis=(0, 1)")
for xdim in range(xlen):
    for ydim in range(ylen):
        print(np.shape(arr[xdim, ydim]))

####################################################################################################
#
# Example 3. 3D Array (n_channels, n_epochs, n_timepoints)
# --------------------------------------------------------
# The features from a 3d array of (n_channels, n_epochs, n_timepoints) will be computed here.
# Bursting frequencies and rise-decay symmetry will be modulated across channels and epochs,
# respectively. The bursting frequencies and rise-decay symmetries will then be compared between
# the simulated parameters and bycycle's caculation.

####################################################################################################

# Simulation settings
n_seconds = 10
fs = 500

freq = 10
f_range = (5, 15)

n_channels = 5
epochs = 10

####################################################################################################
# Simulate a 3d timeseries
sigs = np.zeros((n_channels, epochs, n_seconds*fs))
freqs = np.linspace(5, 45, 5)

for ch_idx, freq in zip(range(n_channels), freqs):

    for ep_idx in range(epochs):

        rdsym = 0.5 if ep_idx % 2 == 0 else 0.8

        sim_components = {'sim_powerlaw': dict(exponent=-2),
                          'sim_bursty_oscillation': dict(freq=freq, cycle='asine', rdsym=rdsym)}

        sigs[ch_idx][ep_idx] = sim_combined(n_seconds, fs, components=sim_components)

####################################################################################################

# Compute features
thresholds = dict(amp_fraction_threshold=0., amp_consistency_threshold=.5,
                  period_consistency_threshold=.5, monotonicity_threshold=.6,
                  min_n_cycles=3)

compute_kwargs = {'burst_method': 'cycles', 'threshold_kwargs': thresholds}

df_features = compute_features_3d(sigs, fs, (1, 50), axis=0,
                                  compute_features_kwargs=compute_kwargs)

####################################################################################################

# Split dfs by epoch type
df_rest = pd.concat([df_features[ch_idx][ep_idx] for ep_idx in range(epochs) if ep_idx % 2 == 0
                     for ch_idx in range(n_channels)], axis=0, ignore_index=True)
df_rest['epoch'] = 'rest'
df_rest = df_rest[df_rest['is_burst'] == True]

df_task = pd.concat([df_features[ch_idx][ep_idx] for ep_idx in range(epochs) if ep_idx % 2 != 0
                     for ch_idx in range(n_channels)], axis=0, ignore_index=True)
df_task['epoch'] = 'task'
df_task = df_task[df_task['is_burst'] == True]

####################################################################################################

# See how well bycycle estimated each bursting cycles period across channels
periods = [pd.concat(df_features[ch_idx], axis=0) for ch_idx in range(n_channels)]
periods = [df[df['is_burst'] == True]['period'].mean() for df in periods]

freqs_est = [round(fs/ period, 3) for period in periods]

df_freqs = pd.DataFrame()
df_freqs['Channel'] = ['CH_0{idx}'.format(idx=idx) for idx in range(n_channels)]
df_freqs['Simulated Freqs'] = freqs
df_freqs['Calculated Freqs'] = freqs_est
df_freqs['Error'] = np.abs(freqs - freqs_est)
df_freqs

####################################################################################################

# See how well bycycle estimated each bursting cycle's rise-decay symmetry within epochs
df_rdsym = pd.DataFrame()

df_rdsym['Epoch Type'] = ['Fixation', 'Task']
df_rdsym['Simulated rdsym'] = [0.5, 0.75]
df_rdsym['Calculated rdsym'] = [df_rest['time_rdsym'].mean(), df_task['time_rdsym'].mean()]
df_rdsym['Error'] = np.abs(df_rdsym['Simulated rdsym'] - df_rdsym['Calculated rdsym'])
df_rdsym

####################################################################################################
