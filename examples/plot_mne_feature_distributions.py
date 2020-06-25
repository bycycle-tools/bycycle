"""
MNE interface cycle feature distributions
=============================================
This example computes the distributions of bycycle features using MNE objects
"""

####################################################################################################
# Import Packages and Load Data
# -----------------------------
#
# First let's import the packages we need. This example depends on mne as well
# as the pactools simulator to make pac and a spurious pac function from the
# pactools spurious pac example.
####################################################################################################

import mne
from mne.datasets import sample
import numpy as np
import matplotlib.pyplot as plt

from bycycle.features import compute_features

###################################################################################################

# frequencies of interest: the alpha band
f_alpha = (8, 15)

# Get the data path for the MNE example data
raw_fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Load the file of example MNE data
raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

###################################################################################################

# Select EEG channels from the dataset
raw = raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')

# Grab the sampling rate from the data
fs = raw.info['sfreq']

# filter to alpha
raw = raw.filter(l_freq=None, h_freq=20.)

###################################################################################################

# Settings for exploring example channels of data (with colors to plot)
chs = {'EEG 042': 'C0', 'EEG 043': 'C1', 'EEG 044': 'C2'}
t_start = 20000
t_stop = int(t_start + (10 * fs))

###################################################################################################

# Extract an example channels to explore
sig, times = raw.get_data(mne.pick_channels(raw.ch_names, list(chs.keys())),
                          start=t_start, stop=t_stop, return_times=True)

####################################################################################################
#
# Plot time series for each recording
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now let's see how each signal looks in time. This looks like standard EEG
# data.
#

####################################################################################################

# plot the signal
fig, ax = plt.subplots(figsize=(10, 4))
for ch, sig0 in zip(chs, sig):
    ax.plot(times, sig0 * 1e6, color=chs[ch], label=ch)
ax.set(title='EEG Signal', xlabel='Time (sec)', ylabel=r'$\mu V$')
ax.legend()
plt.show()

####################################################################################################
# Compute cycle-by-cycle features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here we use the bycycle compute_features function to compute the cycle-by-
# cycle features of the three signals.
#

####################################################################################################

# Set parameters for defining oscillatory bursts
osc_kwargs = {'amplitude_fraction_threshold': 0.3,
              'amplitude_consistency_threshold': 0.4,
              'period_consistency_threshold': 0.5,
              'monotonicity_threshold': 0.8,
              'n_cycles_min': 3}

dfs = dict()
for ch, sig0 in zip(chs, sig):
    # Cycle-by-cycle analysis
    dfs[ch] = compute_features(sig0, fs, f_alpha, center_extrema='T',
                               burst_detection_kwargs=osc_kwargs)


####################################################################################################
#
# Plot feature distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As it turns out, none of the channels in the mne example audio and visual
# task has waveform asymmetry. These data were collected from a healthy
# person while they listened to beeps or saw gratings on a screen
# so this is not unexpected.
#

####################################################################################################

plt.figure(figsize=(5, 5))
for ch, df in dfs.items():
    plt.hist(df['volt_amp'] * 1e6, bins=np.arange(0, 40, 4),
             color=chs[ch], alpha=.5, label=ch)
plt.xticks(np.arange(0, 40, 4), size=12)
plt.legend(fontsize=15)
plt.yticks(size=12)
plt.xlim((0, 40.5))
plt.xlabel('Cycle amplitude (mV)', size=15)
plt.ylabel('# cycles', size=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
for ch, df in dfs.items():
    plt.hist(df['period'] / fs * 1000, bins=np.arange(0, 250, 25),
             color=chs[ch], alpha=.5)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim((0, 250))
plt.xlabel('Cycle period (ms)', size=15)
plt.ylabel('# cycles', size=15)
plt.tight_layout()
plt.show()

bins = np.arange(0, 1, .1)
plt.figure(figsize=(5, 5))
for ch, df in dfs.items():
    plt.hist(df['time_rdsym'], bins=bins, color=chs[ch], alpha=.5)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim((0, 1))
plt.xlabel('Rise-decay asymmetry\n(fraction of cycle in rise period)', size=15)
plt.ylabel('# cycles', size=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
for ch, df in dfs.items():
    plt.hist(df['time_ptsym'], bins=bins, color=chs[ch], alpha=.5)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim((0, 1))
plt.xlabel('Peak-trough asymmetry\n(fraction of cycle in peak period)',
           size=15)
plt.ylabel('# cycles', size=15)
plt.tight_layout()
plt.show()
