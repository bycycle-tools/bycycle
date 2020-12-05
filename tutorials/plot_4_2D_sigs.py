"""
4. Running Bycycle on 2D Arrays
===============================

Compute bycycle features for 2D organizations of timeseries.

Bycycle supports computing the features of 2D signals using :func:`~.compute_features_2d`.
Signals may be organized in a different ways, including (n_epochs, n_timepoints) or
(n_channels, n_timepoints). The difference between these organizataions is that continuity is
preserved across epochs, but not across channels. The ``axis`` argument is used to specificy either
continuous (``axis=1``) and non-continuous (``axis=None``) recordings. This argument functions
identically to numpy functions, such as np.mean or np.sum.
"""

####################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neurodsp.sim import sim_combined
from bycycle.group import compute_features_2d
from bycycle.plts import plot_feature_categorical

####################################################################################################
# The ``axis`` Argument
# ---------------------
#
# Here, we will examine an analogous ``np.sum`` example of the axis argument before moving
# onto specific bycycle examples. The axis argument be may specified as:
#
# - ``axis=1`` : Operates across each row/signal in an array independently (i.e. for each
#   channel in (n_channels, n_timepoints)).
# - ``axis=None`` : Operates on flattened rows/signals (i.e. across flatten epochs
#   in (n_epochs, n_timepoints)).
#

####################################################################################################

# Analagous to (n_channels, n_timepoints) or (n_epochs, n_timepoints)
arr = np.ones((2, 3))

print(arr, "\n")
print("axis=1    (channels): ", np.sum(arr, axis=1))
print("axis=None (epochs)  : ", np.sum(arr, axis=None))

####################################################################################################
#
# Example 1. 2D Array (n_epochs, n_timepoints)
# --------------------------------------------
# The features for a 2d array with a (n_epochs, n_timepoints) organization will be computed here.
# This is an example of when using ``axis = None`` is appropriate, assuming each epoch was recorded
# continuously. Data will be simulated to produce to alternating epoch types (rest and task)
# with varying rise-decay symmetries.

####################################################################################################

# Simulation settings
n_seconds = 10
fs = 500

freq = 10
f_range = (5, 15)

epochs = 10

# Define rdsym values for rest and task trials
rdsym_rest = 0.5
rdsym_task = 0.75

####################################################################################################

# Simulate a timeseries
sigs = np.zeros((epochs, n_seconds*fs))

for idx in range(epochs):

    if idx % 2 == 0:
        # Rest epoch
        sim_components = {'sim_powerlaw': dict(exponent=-2),
                          'sim_bursty_oscillation': dict(freq=10, cycle='asine',
                                                         enter_burst=0.75, rdsym=rdsym_rest)}
    else:
        # Task epoch
        sim_components = {'sim_powerlaw': dict(exponent=-2),
                          'sim_bursty_oscillation': dict(freq=10, cycle='asine',
                                                         enter_burst=0.75, rdsym=rdsym_task)}

    sigs[idx] = sim_combined(n_seconds, fs, components=sim_components)

####################################################################################################

# Compute features with default bycycle thresholds
thresholds = dict(amp_fraction_threshold=0., amp_consistency_threshold=.5,
                  period_consistency_threshold=.5, monotonicity_threshold=.8,
                  min_n_cycles=3)

compute_kwargs = {'burst_method': 'cycles', 'threshold_kwargs': thresholds}

df_features = compute_features_2d(sigs, fs, f_range, axis=None,
                                  compute_features_kwargs=compute_kwargs)

####################################################################################################

# Add a column specifying epoch type
for idx, df in enumerate(df_features):
    df['epoch'] = 'rest' if idx % 2 == 0 else 'task'

# Merge into a single dataframe
df_concat = pd.concat(df_features, axis=0, ignore_index=True)

# Limit to bursting cycles
df_bursts = df_concat[df_concat['is_burst'] == True]

# Plot
plot_feature_categorical(df_bursts, 'time_rdsym', group_by='epoch', ylabel='Rise-Decay Symmetry',
                         xlabel=['Rest', 'Task'])

####################################################################################################
#
# Example 2. 2D Array (n_copies, n_timepoints)
# --------------------------------------------
#
# In this example, we will compute features for a 2D array in which we have organized multiple
# copies of the same signal. This is done to explore applying different thresholds while taking
# advantage of parallel processing capabilities. This example also shows how a list of dictionaries
# may be used to pass unique compute_features_kwargs to each signal.

####################################################################################################

# Simulate a timeseries
n_copies = 5

sim_components = {'sim_powerlaw': dict(exponent=-2),
                  'sim_bursty_oscillation': dict(freq=10)}

sig = sim_combined(n_seconds, fs, components=sim_components)

sigs = np.array([sig] * n_copies)

####################################################################################################

# Step through amp_consistency thresholds by 0.25
thresholds = dict(amp_fraction_threshold=0., amp_consistency_threshold=.5,
                  period_consistency_threshold=.5, monotonicity_threshold=.8,
                  min_n_cycles=3)

amp_consist_threshes = [0, .25, .5, .75, 1.]

compute_kwargs = [{'threshold_kwargs': dict(thresholds, amp_consistency_threshold=thresh)}
                  for thresh in amp_consist_threshes]

# Compute features
df_features = compute_features_2d(sigs, fs, f_range, compute_features_kwargs=compute_kwargs,
                                  axis=1, n_jobs=-1, progress="tqdm")

####################################################################################################

# Plot the number of detected burst at each threshold
n_bursts = [np.count_nonzero(df_features[idx]['is_burst']) for idx in range(n_copies)]

fig = plt.figure(figsize=(8, 8))
plt.plot(amp_consist_threshes, n_bursts, marker="o")
plt.xlabel("Amplitude Consistency Threshold")
plt.ylabel("Number of Bursts")

"""
4. Running Bycycle on 2D Arrays
===============================

Compute bycycle features for 2D organizations of timeseries.

Bycycle supports computing the features of 2D signals using :func:`~.compute_features_2d`.
Signals may be organized in a different ways, including (n_epochs, n_timepoints) or
(n_channels, n_timepoints). The difference between these organizataions is that continuity is
preserved across epochs, but not across channels. The ``axis`` argument is used to specificy either
continuous (``axis=1``) and non-continuous (``axis=None``) recordings. This argument functions
identically to numpy functions, such as np.mean or np.sum.
"""

####################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neurodsp.sim import sim_combined
from bycycle.group import compute_features_2d
from bycycle.plts import plot_feature_categorical

####################################################################################################
# The ``axis`` Argument
# ---------------------
#
# Here, we will examine an analogous ``np.sum`` example of the axis argument before moving
# onto specific bycycle examples. The axis argument be may specified as:
#
# - ``axis=1`` : Operates across each row/signal in an array independently (i.e. for each
#   channel in (n_channels, n_timepoints)).
# - ``axis=None`` : Operates on flattened rows/signals (i.e. across flatten epochs
#   in (n_epochs, n_timepoints)).
#

####################################################################################################

# Analagous to (n_channels, n_timepoints) or (n_epochs, n_timepoints)
arr = np.ones((2, 3))

print(arr, "\n")
print("axis=1    (channels): ", np.sum(arr, axis=1))
print("axis=None (epochs)  : ", np.sum(arr, axis=None))

####################################################################################################
#
# Example 1. 2D Array (n_epochs, n_timepoints)
# --------------------------------------------
# The features for a 2d array with a (n_epochs, n_timepoints) organization will be computed here.
# This is an example of when using ``axis = None`` is appropriate, assuming each epoch was recorded
# continuously. Data will be simulated to produce to alternating epoch types (rest and task)
# with varying rise-decay symmetries.

####################################################################################################

# Simulation settings
n_seconds = 10
fs = 500

freq = 10
f_range = (5, 15)

epochs = 10

# Define rdsym values for rest and task trials
rdsym_rest = 0.5
rdsym_task = 0.75

####################################################################################################

# Simulate a timeseries
sigs = np.zeros((epochs, n_seconds*fs))

for idx in range(epochs):

    if idx % 2 == 0:
        # Rest epoch
        sim_components = {'sim_powerlaw': dict(exponent=-2),
                          'sim_bursty_oscillation': dict(freq=10, cycle='asine',
                                                         enter_burst=0.75, rdsym=rdsym_rest)}
    else:
        # Task epoch
        sim_components = {'sim_powerlaw': dict(exponent=-2),
                          'sim_bursty_oscillation': dict(freq=10, cycle='asine',
                                                         enter_burst=0.75, rdsym=rdsym_task)}

    sigs[idx] = sim_combined(n_seconds, fs, components=sim_components)

####################################################################################################

# Compute features with default bycycle thresholds
thresholds = dict(amp_fraction_threshold=0., amp_consistency_threshold=.5,
                  period_consistency_threshold=.5, monotonicity_threshold=.8,
                  min_n_cycles=3)

compute_kwargs = {'burst_method': 'cycles', 'threshold_kwargs': thresholds}

df_features = compute_features_2d(sigs, fs, f_range, axis=None,
                                  compute_features_kwargs=compute_kwargs)

####################################################################################################

# Add a column specifying epoch type
for idx, df in enumerate(df_features):
    df['epoch'] = 'rest' if idx % 2 == 0 else 'task'

# Merge into a single dataframe
df_concat = pd.concat(df_features, axis=0, ignore_index=True)

# Limit to bursting cycles
df_bursts = df_concat[df_concat['is_burst'] == True]

# Plot
plot_feature_categorical(df_bursts, 'time_rdsym', group_by='epoch', ylabel='Rise-Decay Symmetry',
                         xlabel=['Rest', 'Task'])

####################################################################################################
#
# Example 2. 2D Array (n_copies, n_timepoints)
# --------------------------------------------
#
# In this example, we will compute features for a 2D array in which we have organized multiple
# copies of the same signal. This is done to explore applying different thresholds while taking
# advantage of parallel processing capabilities. This example also shows how a list of dictionaries
# may be used to pass unique compute_features_kwargs to each signal.

####################################################################################################

# Simulate a timeseries
n_copies = 5

sim_components = {'sim_powerlaw': dict(exponent=-2),
                  'sim_bursty_oscillation': dict(freq=10)}

sig = sim_combined(n_seconds, fs, components=sim_components)

sigs = np.array([sig] * n_copies)

####################################################################################################

# Step through amp_consistency thresholds by 0.25
thresholds = dict(amp_fraction_threshold=0., amp_consistency_threshold=.5,
                  period_consistency_threshold=.5, monotonicity_threshold=.8,
                  min_n_cycles=3)

amp_consist_threshes = [0, .25, .5, .75, 1.]

compute_kwargs = [{'threshold_kwargs': dict(thresholds, amp_consistency_threshold=thresh)}
                  for thresh in amp_consist_threshes]

# Compute features
df_features = compute_features_2d(sigs, fs, f_range, compute_features_kwargs=compute_kwargs,
                                  axis=1, n_jobs=-1, progress="tqdm")

####################################################################################################

# Plot the number of detected burst at each threshold
n_bursts = [np.count_nonzero(df_features[idx]['is_burst']) for idx in range(n_copies)]

fig = plt.figure(figsize=(8, 8))
plt.plot(amp_consist_threshes, n_bursts, marker="o")
plt.xlabel("Amplitude Consistency Threshold")
plt.ylabel("Number of Bursts")

