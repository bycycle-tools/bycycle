"""
2. 2D and 3D Signals
====================

Compute bycycle features for 2D and 3D organizations of timeseries.

Bycycle supports computing features for 2D and 3D arrays of signals, using
:func:`~.compute_features_2d` or :func:`~.compute_features_3d`. These signals may be organized in a
variety of ways, such as (n_epochs, n_timepoints), (n_channels, n_timepoints), or
(n_channels,  n_epochs, n_timepoints). The ``axis`` keyword argument may be used to handle various
organizations of signals. Understanding the axis argument is important to ensure features are
computed in an appropriate manor.

For 2D arrays:

- Use ``axis = 0`` (default) when each signal is independent from others. This may be the case for
  an (n_channels, n_timepoints) organization.
- Use ``axis = None`` when each signal is dependent on others. This may be the case for an
  (n_epochs, n_timepoints) organization, where each epoch was recorded continuously in a single
  recording.
- A 1D list of feature dataframes is returned.

For 3d arrays:

- Use ``axis = 0`` (default) when computing features independently across the first axis.
  This is likely the case for a (n_channels, n_epochs, n_timepoints) organization.
- Use ``axis = 1`` when computing features independently across the second axis. A possible
  organization is (n_epochs, n_channels, n_timepoints).
- Use ``axis = (0, 1)`` when computing features independently, for each signal, across both axes.
  This may be the case for an (n_subjects, n_channels, n_timpoints) oraganization
- Use ``axis = None`` when each signal is dependent on others. This may be the case for an
  (n_runs, n_epochs, n_timpoints) oraganization, assuming a continuous recording.
- A 2D list of feature dataframes is returned.

"""

####################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neurodsp.sim import sim_combined
from bycycle.group import compute_features_2d, compute_features_3d
from bycycle.plts import plot_feature_categorical, plot_feature_hist

####################################################################################################

# Simulation settings
n_seconds = 10
fs = 500

freq = 10
f_range = (5, 15)

epochs = 10
channels = 5

####################################################################################################

####################################################################################################
#
# Example 1. 2D Array (n_epochs, n_timepoints)
# --------------------------------------------
# The features for a 2d array with a (n_epochs, n_timepoints) organiation will be computed here.
# This is an example of when using ``axis = None`` is appropriate, assuming each epoch was recorded
# continuously. Data will be simulated to produce to alternating epoch types (fixation and task)
# with varying rise-decay symmetries.

# Simulate a timeseries
sigs = np.zeros((epochs, n_seconds*fs))

for idx in range(epochs):

    if idx % 2 == 0:
        # Fixation epoch
        sim_components = {'sim_powerlaw': dict(exponent=-2),
                          'sim_bursty_oscillation': dict(freq=10, cycle='asine',
                                                         enter_burst=0.75, rdsym=0.5)}
    else:
        # Task epoch
        sim_components = {'sim_powerlaw': dict(exponent=-2),
                          'sim_bursty_oscillation': dict(freq=10, cycle='asine',
                                                         enter_burst=0.75, rdsym=0.75)}

    sigs[idx] = sim_combined(n_seconds, fs, components=sim_components)

####################################################################################################

# Compute features
thresholds = dict(amp_fraction_threshold=0., amp_consistency_threshold=.5,
                  period_consistency_threshold=.5, monotonicity_threshold=.8,
                  min_n_cycles=2)

compute_kwargs = {'burst_method': 'cycles', 'threshold_kwargs': thresholds}

df_features = compute_features_2d(sigs, fs, f_range, axis=None,
                                  compute_features_kwargs=compute_kwargs)

####################################################################################################

# Add a column specifying epoch type
for idx, df in enumerate(df_features):
    df['epoch'] = 'fixation' if idx % 2 == 0 else 'task'

# Merge into a single dataframe
df_concat = pd.concat(df_features, axis=0, ignore_index=True)

# Limit to bursting cycles
df_bursts = df_concat[df_concat['is_burst'] == True]

# Plot
plot_feature_categorical(df_bursts, 'time_rdsym', group_by='epoch', ylabel='Rise-Decay Symmetry',
                         xlabel=['Fixation', 'Task'])

####################################################################################################
#
# Here we show what would happen if axis = 0 was incorrectly used when computing the features
# for an (n_epochs, n_timeseries) array. If bursts exist at the edges of epochs, they will be
# incorrectly labeled as non-burst cycles. This is due to the inability to compute consistency
# features for the first and last cycle of each signal.

df_axis0 = compute_features_2d(sigs, fs, f_range, axis=0, compute_features_kwargs=compute_kwargs)
df_concat_axis0 = pd.concat(df_axis0, axis=0, ignore_index=True)
df_bursts_axis0 = df_concat_axis0[df_concat_axis0['is_burst'] == True]

print("""
    Number of bursts (axis = None): {n_bursts_axis_None}
    Number of bursts (axis = 0   ): {n_bursts_axis_0}
""".format(n_bursts_axis_None=len(df_bursts), n_bursts_axis_0=len(df_bursts_axis0)))

####################################################################################################
#
# Example 2. 2D Array (n_copies, n_timepoints)
# --------------------------------------------
# Features for a 2D array with an (n_copies, n_timepoints) organization will be computed
# independently, using axis = 0. Each signal will be indentical, demonstrating the effect
# of various thresholds on burst detection. This example also shows how a list of dictionaries may
# be passed to compute_features_kwargs, allowing unique kwargs for each signal.

# Simulate a timeseries
n_copies = 5

sim_components = {'sim_powerlaw': dict(exponent=-2),
                  'sim_bursty_oscillation': dict(freq=10)}

sig = sim_combined(n_seconds, fs, components=sim_components)

sigs = np.array([sig] * n_copies)

####################################################################################################

# Step amp_consistency thresholds by 0.25
thresholds = dict(amp_fraction_threshold=0., amp_consistency_threshold=.5,
                  period_consistency_threshold=.5, monotonicity_threshold=.8,
                  min_n_cycles=2)

amp_consist_threshes = [0, .25, .5, .75, 1.]

compute_kwargs = [{'threshold_kwargs': dict(thresholds.copy(), amp_consistency_threshold=thresh)}
                  for thresh in amp_consist_threshes]

# Compute features
df_features = compute_features_2d(sigs, fs, f_range, axis=0,
                                  compute_features_kwargs=compute_kwargs)

####################################################################################################

# Plot the number of detected burst at each threshold
n_bursts = [np.count_nonzero(df_features[idx]['is_burst']) for idx in range(n_copies)]

fig = plt.figure(figsize=(8, 8))
plt.plot(amp_consist_threshes, n_bursts, marker="o")
plt.xlabel("Amplitude Consistency Threshold")
plt.ylabel("Number of Bursts")

####################################################################################################
#
# Example 3. 3D Array (n_channels, n_epochs, n_timepoints)
# --------------------------------------------------------
# The features from a 3d array of (n_channels, n_epochs, n_timepoints) will be computed here.
# Bursting frequencies and rise-decay symmetry will be modulated across channels and epochs,
# respectively. The bursting frequencies and rise-decay symmetries will then be compared between
# the simulated parameters and bycycle's caculation.

# Simulate a 3d timeseries
sigs = np.zeros((channels, epochs, n_seconds*fs))
freqs = np.linspace(5, 45, 5)

for ch_idx, freq in zip(range(channels), freqs):

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

# Split dfs by channel
df_chs = [df_features[ch_idx] for ch_idx in range(channels)]

# Split dfs by epoch type
df_fix = pd.concat([df_features[ch_idx][ep_idx] for ep_idx in range(epochs) if ep_idx % 2 == 0
                    for ch_idx in range(channels)], axis=0, ignore_index=True)
df_fix = df_fix[df_fix['is_burst'] == True]

df_task = pd.concat([df_features[ch_idx][ep_idx] for ep_idx in range(epochs) if ep_idx % 2 != 0
                     for ch_idx in range(channels)], axis=0, ignore_index=True)
df_task = df_task[df_task['is_burst'] == True]

####################################################################################################

# See how well bycycle estimated each bursting cycles period across channels
periods = [pd.concat(df_features[ch_idx], axis=0) for ch_idx in range(channels)]
periods = [df[df['is_burst'] == True]['period'].mean() for df in periods]

freqs_est = [round(fs/ period, 3) for period in periods]

df_freqs = pd.DataFrame()
df_freqs['Channel'] = ['CH_0{idx}'.format(idx=idx) for idx in range(channels)]
df_freqs['Simulated Freqs'] = freqs
df_freqs['Calculated Freqs'] = freqs_est
df_freqs['Error'] = np.abs(freqs - freqs_est)
df_freqs

####################################################################################################

# See how well bycycle estimated each bursting cycle's rise-decay symmetry within epochs

df_rdsym = pd.DataFrame()

df_rdsym['Epoch Type'] = ['Fixation', 'Task']
df_rdsym['Simulated rdsym'] = [0.5, 0.75]
df_rdsym['Calculated rdsym'] = [df_fix['time_rdsym'].mean(), df_task['time_rdsym'].mean()]
df_rdsym['Error'] = np.abs(df_rdsym['Simulated rdsym'] - df_rdsym['Calculated rdsym'])
df_rdsym
