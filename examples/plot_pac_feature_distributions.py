"""
PAC cycle feature distributions
===============================
This example computes the distributions of bycycle for simulated
phase-amplitude coupling (PAC)
"""

####################################################################################################
# Import packages
# ---------------
#
# First let's import the packages we need. This example depends on the
# pactools simulator to make pac and a spurious pac function from the
# pactools spurious pac example.
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from pactools import simulate_pac, Comodulogram
from pactools.utils.pink_noise import pink_noise
from pactools.utils.validation import check_random_state

from bycycle.features import compute_features


def simulate_spurious_pac(n_points, fs, spike_amp=1.5, spike_fwhm=0.01,
                          spike_fq=10., spike_interval_jitter=0.2,
                          random_state=None):
    """Simulate some spurious phase-amplitude coupling (PAC) with spikes

    References
    ----------
    Gerber, E. M., Sadeh, B., Ward, A., Knight, R. T., & Deouell, L. Y. (2016).
    Non-sinusoidal activity can produce cross-frequency coupling in cortical
    signals in the absence of functional interaction between neural sources.
    PloS one, 11(12), e0167351
    """
    n_points = int(n_points)
    fs = float(fs)
    rng = check_random_state(random_state)
    # draw the position of the spikes
    interval_min = 1. / float(spike_fq) * (1. - spike_interval_jitter)
    interval_max = 1. / float(spike_fq) * (1. + spike_interval_jitter)
    n_spikes_max = np.int(n_points / fs / interval_min)
    spike_intervals = rng.uniform(low=interval_min, high=interval_max,
                                  size=n_spikes_max)
    spike_positions = np.cumsum(np.int_(spike_intervals * fs))
    spike_positions = spike_positions[spike_positions < n_points]
    # build the spike time series, using a convolution
    spikes = np.zeros(n_points)
    spikes[spike_positions] = spike_amp
    # full width at half maximum to standard deviation convertion
    spike_std = spike_fwhm / (2 * np.sqrt(2 * np.log(2)))
    spike_shape = scipy.signal.gaussian(M=np.int(spike_std * fs * 10),
                                        std=spike_std * fs)
    spikes = scipy.signal.fftconvolve(spikes, spike_shape, mode='same')
    noise = pink_noise(n_points, slope=1., random_state=random_state)
    return spikes + noise, spikes

####################################################################################################
# Simulate pac data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now we will use the imported functions to make pac data, spurious pac data
# and a sine wave with pink noise.
#

####################################################################################################


f_beta = (14, 28)
trial_len = 40  # seconds
fs = 200.  # Hz
high_fq = 80.0  # Hz; carrier frequency
low_fq = 24.0  # Hz; driver frequency
low_fq_width = 2.0  # Hz

n_points = int(trial_len * fs)
noise_level = 0.25

# simulate beta-gamma pac
signal_pac = simulate_pac(n_points=n_points, fs=fs,
                          high_fq=high_fq, low_fq=low_fq,
                          low_fq_width=low_fq_width,
                          noise_level=noise_level, random_state=99)
# simulate 10 Hz spiking which couples to about 60 Hz
signal_spurious_pac, spikes = simulate_spurious_pac(
    n_points=n_points, fs=fs, spike_amp=1. / noise_level, random_state=999)
# make a sine wave that is the driver frequency
low_fq_signal = np.sin(2 * np.pi * low_fq * np.linspace(0, trial_len,
                                                        n_points))
# add the sine wave to pink noise to make a control, no pac signal
signal_no_pac = low_fq_signal + pink_noise(n_points, slope=1.,
                                           random_state=9999)

####################################################################################################
# Check comodulogram for PAC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Now, we will use the tools from pactools to compute the comodulogram which
is a visualization of phase amplitude coupling. The stronger the driver
phase couples with a particular frequency, the brighter the color value.
The pac signal has 24 - 80 Hz pac as designed, the spurious pac has 10 - 60 Hz
spurious pac and the final signal has no pac, just background noise.
"""
####################################################################################################

fig, axs = plt.subplots(nrows=3, figsize=(10, 12), sharex=True)
for signal, ax in zip((signal_pac, signal_spurious_pac, signal_no_pac), axs):
    # check PAC within only channel; high and low sig are the same-- channel i
    # use the duprelatour driven autoregressive model to fit the data
    estimator = Comodulogram(fs=fs, low_fq_range=np.arange(1, 41),
                             low_fq_width=2., method='duprelatour',
                             progress_bar=True)
    # compute the comodulogram
    estimator.fit(signal)
    # plot the results
    estimator.plot(axs=[ax], tight_layout=False)
plt.show()

####################################################################################################
#
# Plot time series for each recording
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now let's see how each signal looks in time. The third plot of spikes is
# added to pink noise to make the second plot which is the spurious pac.
# This is so that where the spikes occur can be noted in the spurious pac plot.
#

####################################################################################################

# plot the signal and the spikes
n_points_plot = np.int(fs)
time = np.arange(n_points_plot) / fs
fig, axs = plt.subplots(nrows=4, figsize=(10, 12), sharex=True)
axs[0].plot(time, signal_pac[:n_points_plot], color='C0')
axs[0].set(title='Signal with PAC', ylabel=r'$\mu V$')
axs[1].plot(time, signal_spurious_pac[:n_points_plot], color='C1')
axs[1].set(title='Signal with Spurious PAC', ylabel=r'$\mu V$')
axs[2].plot(time, spikes[:n_points_plot], color='C3')
axs[2].set(title='Spikes', ylabel=r'$\mu V$')
axs[3].plot(time, signal_no_pac[:n_points_plot], color='C2')
axs[3].set(xlabel='Time (sec)', title='Signal with no PAC', ylabel=r'$\mu V$')
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

# Cycle-by-cycle analysis
df_pac = compute_features(signal_pac, fs, f_beta, center_extrema='T',
                          burst_detection_kwargs=osc_kwargs)

df_spurious = compute_features(signal_spurious_pac, fs, f_beta,
                               center_extrema='T',
                               burst_detection_kwargs=osc_kwargs)

df_no_pac = compute_features(signal_no_pac, fs, f_beta, center_extrema='T',
                             burst_detection_kwargs=osc_kwargs)


####################################################################################################
#
# Plot feature distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As shown in the feature distributions, the pac signal displays some peak-
# tough asymmetry as does the spurious pac signal.
#

####################################################################################################

# voltage amplitude
plt.figure(figsize=(5, 5))
plt.hist(df_pac['volt_amp'], bins=np.arange(0, 8, .1),
         color='C0', alpha=.5, label='PAC')
plt.hist(df_spurious['volt_amp'], bins=np.arange(0, 8, .1),
         color='C1', alpha=.5, label='spurious')
plt.hist(df_no_pac['volt_amp'], bins=np.arange(0, 8, .1),
         color='C2', alpha=.5, label='no PAC')
plt.xticks(np.arange(5), size=12)
plt.legend(fontsize=15)
plt.yticks(size=12)
plt.xlim((0, 4.5))
plt.xlabel('Cycle amplitude (mV)', size=15)
plt.ylabel('# cycles', size=15)
plt.tight_layout()
plt.show()

# period
plt.figure(figsize=(5, 5))
plt.hist(df_pac['period'] / fs * 1000, bins=np.arange(0, 250, 5),
         color='C0', alpha=.5)
plt.hist(df_spurious['period'] / fs * 1000, bins=np.arange(0, 250, 5),
         color='C1', alpha=.5)
plt.hist(df_no_pac['period'] / fs * 1000, bins=np.arange(0, 250, 5),
         color='C2', alpha=.5)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim((0, 250))
plt.xlabel('Cycle period (ms)', size=15)
plt.ylabel('# cycles', size=15)
plt.tight_layout()
plt.show()

# rise-decay asymmetry
bins = np.arange(0, 1, .1)
plt.figure(figsize=(5, 5))
plt.hist(df_pac['time_rdsym'], bins=bins, color='C0', alpha=.5)
plt.hist(df_spurious['time_rdsym'], bins=bins, color='C1', alpha=.5)
plt.hist(df_no_pac['time_rdsym'], bins=bins, color='C2', alpha=.5)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim((0, 1))
plt.xlabel('Rise-decay asymmetry\n(fraction of cycle in rise period)', size=15)
plt.ylabel('# cycles', size=15)
plt.tight_layout()
plt.show()

# peak-trough asymmetry
plt.figure(figsize=(5, 5))
plt.hist(df_pac['time_ptsym'], bins=bins, color='C0', alpha=.5)
plt.hist(df_spurious['time_ptsym'], bins=bins, color='C1', alpha=.5)
plt.hist(df_no_pac['time_ptsym'], bins=bins, color='C2', alpha=.5)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim((0, 1))
plt.xlabel('Peak-trough asymmetry\n(fraction of cycle in peak period)',
           size=15)
plt.ylabel('# cycles', size=15)
plt.tight_layout()
plt.show()
