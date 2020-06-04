"""
Theta oscillation cycle feature distributions
=============================================
This tutorial computes the distributions of cycle features for two recordings
"""

####################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.filt import filter_signal
from neurodsp.plts import plot_time_series

from bycycle.plts import plot_cycle_features
from bycycle.features import compute_features

####################################################################################################
# Load and preprocess data
# ~~~~~~~~~~~~~~~~~~~~~~~~

####################################################################################################

# Load data
ca1_raw = np.load('data/ca1.npy')
ec3_raw = np.load('data/ec3.npy')
fs = 1250
f_theta = (4, 10)

####################################################################################################

# Only keep 60 seconds of data
n_seconds = 60
ca1_raw = ca1_raw[:int(n_seconds*fs)]
ec3_raw = ec3_raw[:int(n_seconds*fs)]

####################################################################################################

# Apply a lowpass filter at 25Hz
fc = 25
filter_seconds = .5

ca1 = filter_signal(ca1_raw, fs, 'lowpass', fc, n_seconds=filter_seconds,
                    remove_edges=False)
ec3 = filter_signal(ec3_raw, fs, 'lowpass', fc, n_seconds=filter_seconds,
                    remove_edges=False)

####################################################################################################
# Compute cycle-by-cycle features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

####################################################################################################

# Set parameters for defining oscillatory bursts
osc_kwargs = {'amplitude_fraction_threshold': 0,
              'amplitude_consistency_threshold': .6,
              'period_consistency_threshold': .75,
              'monotonicity_threshold': .8,
              'n_cycles_min': 3}

# Cycle-by-cycle analysis
df_ca1 = compute_features(ca1, fs, f_theta, center_extrema='T',
                          burst_detection_kwargs=osc_kwargs)

df_ec3 = compute_features(ec3, fs, f_theta, center_extrema='T',
                          burst_detection_kwargs=osc_kwargs)

# Limit analysis only to oscillatory bursts
df_ca1_cycles = df_ca1[df_ca1['is_burst']]
df_ec3_cycles = df_ec3[df_ec3['is_burst']]

####################################################################################################
#
# Plot time series for each recording
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

####################################################################################################

# Choose samples to plot
samplims = (10000, 12000)
ca1_plt = ca1_raw[samplims[0]:samplims[1]]/1000
ec3_plt = ec3_raw[samplims[0]:samplims[1]]/1000
times = np.arange(0, len(ca1_plt)/fs, 1/fs)

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

plot_time_series(times, ca1_plt, ax=ax1, xlim=(0, 1.6), ylim=(-2.4, 2.4),
                xlabel="Time (s)", ylabel="CA1 Voltage (mV)")

plot_time_series(times, ec3_plt, ax=ax2, colors='r', xlim=(0, 1.6),
                 ylim=(-2.4, 2.4), xlabel="Time (s)", ylabel="EC3 Voltage (mV)")

####################################################################################################
#
# Plot feature distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

####################################################################################################

plot_cycle_features([df_ca1_cycles, df_ec3_cycles], fs=1250, labels=['CA1', 'EC3'])
