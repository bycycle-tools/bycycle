"""
1. Cycle-by-cycle philosophy
============================

Neural signals, like the example shown below, are analyzed in order to extract information about
brain activity. Basically, we process these signals in order to extract features that will hopefully
correlate with a behavior, pathology, or something else.

As the most prominent feature of these signals tends to be the oscillations in them, spectral
analysis is often applied in order to characterize these rhythms in terms of their frequency, power,
and phase.

The conventional approach to analyzing these properties as a function of time is to only study a
narrowband signal by applying a wavelet transform or bandpass filtering followed by the Hilbert
transform. The latter is demonstrated below.

"""

###################################################################################################
#
# Conventional analysis of amplitude and phase: Hilbert Transform
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.filt import filter_signal
from neurodsp.timefrequency import amp_by_time, phase_by_time
from neurodsp.plts import plot_time_series, plot_instantaneous_measure

sig = np.load('data/sim_bursting_more_noise.npy')
fs = 1000  # Sampling rate
f_alpha = (8, 12)
n_seconds_filter = .5

# Compute amplitude and phase
sig_filt = filter_signal(sig, fs, 'bandpass', f_alpha, n_seconds=n_seconds_filter)
theta_amp = amp_by_time(sig, fs, f_alpha, n_seconds=n_seconds_filter)
theta_phase = phase_by_time(sig, fs, f_alpha, n_seconds=n_seconds_filter)

# Plot signal
times = np.arange(0, len(sig)/fs, 1/fs)
xlim = (2, 6)
tidx = np.logical_and(times >= xlim[0], times < xlim[1])

fig, axes = plt.subplots(figsize=(15, 9), nrows=3)

# Plot the raw signal
plot_time_series(times[tidx], sig[tidx], ax=axes[0], ylabel='Voltage (mV)',
                 xlabel='', lw=2, labels='raw signal')

# Plot the filtered signal and oscillation amplitude
plot_instantaneous_measure(times[tidx], [sig_filt[tidx], theta_amp[tidx]],
                           ax=axes[1], measure='amplitude', lw=2, xlabel='',
                           labels=['filtered signal', 'amplitude'])

# Plot the phase
plot_instantaneous_measure(times[tidx], theta_phase[tidx], ax=axes[2], colors='r',
                           measure='phase', lw=2, xlabel='Time (s)')

####################################################################################################
#
# This conventional analysis has some advantages and disadvantages. As for advantages:
#
# - Quick calculation
# - Neat mathematical theory
# - Results largely make sense
# - Defined at every point in time.
#
# Because of this last property, these traces have come to be known as "instantaneous amplitude"
# and "instantaneous phase." And they seem to make a lot of sense, when looking at the raw signal.
#
# However, there are some key disadvantages to this analysis that stem from its sine wave basis.
#
# 1. Being defined at every point in time gives the illusion that the phase and amplitude estimates
#    are valid at all points in time. However, the amplitude and phase estimates are pretty garbage
#    when there's no oscillation going on (the latter half of the time series above). The "amplitude"
#    and "phase" values are meaningless when no oscillation is actually present. Rather, they are
#    influenced by the other aspects of the signal, such as transients. For this reason, these measures
#    are flaws, and burst detection is very important to help alleviate this issue.
# 2. This analysis does not capture a potentially important aspect of the data, in that the
#    oscillatory cycles tend to have short rises and longer decays. This is partly because the signal
#    is filtered in a narrow frequency band (using a sine wave basis) that cannot accurately
#    reconstruct nonsinusoidal waveforms. Furthermore, this nonsinusoidal feature will unintuitively
#    bias amplitude and phase estimates (though perhaps negligibly). Furthermore, there are no apparent
#    tools for extracting nonsinusoidal properties using conventional techniques.
#
#
# Note that different hyperparameter choices for filters can lead to significant differences in results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When effect sizes are small, these hyperparameter choices may make a crucial difference.
#

# Different hyperparameter choices - filter length and center frequency and bandwidth
f_alphas = [(6, 14), (8, 12), (9, 13)]
n_seconds = [.4, .75, 1.2]

amps = []
phases = []

for f_alpha in f_alphas:

    for n_second_filter in n_seconds:

        amp = amp_by_time(sig, fs, f_alpha, n_seconds=n_second_filter)
        phase = phase_by_time(sig, fs, f_alpha, n_seconds=n_second_filter)

        amps.append(amp)
        phases.append(phase)

fig, axes = plt.subplots(figsize=(15, 6), nrows=2)

plot_instantaneous_measure(times, amps, ax=axes[0], xlim=(2,6),
                           measure='amplitude', lw=2, xlabel='')

plot_instantaneous_measure(times, phases, ax=axes[1], xlim=(2,6),
                           measure='phase',lw=2)

####################################################################################################
#
# Cycle-by-cycle approach
# =======================
#
# The main issues in the conventional approach are because the measurements of amplitude and phase
# are very indirect, using certain transforms in the frequency domain defined by sine waves.
# Therefore, we developed an alternative approach that analyzes oscillatory properties more directly
# by staying in the time domain. Arguably, it is best to analyze these signals in the time domain
# because this is the domain in which they are generated (the brain does not generate sums of
# independent sine waves).
#
# The benefits of this alternative approach may include:
#
# - More direct measurements of amplitude and frequency may be more accurate (see Figures 5 and 6 \
#   in the associated preprint).
# - Characterization of waveform shape, in addition to amplitude and phase and frequency.
# - Explicit definitions of which portions of the signal are suitable for analysis (in oscillatory \
#   bursts) or not (no oscillation present).
# - It is important to note that this approach also has some key disadvantages. First, it is not \
#   widely adopted like the conventional techniques. Second, it requires more hyperparameter \
#   choosing and potentialyy more quality control compared to conventional techniques. I emphasize \
#   how important it is to visualize the cycle-by-cycle characterization and burst detection to \
#   assure that the metrics match the intuition. However, this is not commonly expected or \
#   performed using conventional techniques.
#
