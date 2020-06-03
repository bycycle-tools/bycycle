"""Plot burst detection parameters."""

import numpy as np
from scipy.stats import zscore

import matplotlib.pyplot as plt
from  matplotlib import rcParams
rcParams['lines.markersize'] = 10

from neurodsp.plts import plot_time_series
from neurodsp.plts import plot_bursts


def plot_burst_detect_params(sig, fs, df_shape, osc_kwargs, tlims=None,
                             figsize=(16, 3), plot_only_result=False):
    """Create a plot to study how the cycle-by-cycle burst detection
    algorithm determine bursting periods of a signal.

    Parameters
    ----------
    sig : 1d array
        Time series analyzed to compute `df_shape`.
    fs : float
        Sampling rate, in Hz.
    df_shape : pandas DataFrame
        Dataframe output of `features.compute_features()`.
    osc_kwargs : dict
        Dictionary of thresholds for burst detection used in the function
        `features.compute_features()` using the kwarg `burst_detection_kwargs`.
    tlims : tuple of (float, float), optional
        Start and stop times for plot.
    figsize : tuple of (float, float), optional
        Size of figure.
    plot_only_result : bool, optional, default: False
        If True, do not plot the subplots showing the parameters.

    Returns
    -------
    ax : matplotlib axis handle or list of axis handles
        If `plot_only_result` = True: return a plot of the burst
        detection in which periods with bursts are denoted in red.

        If `plot_only_result` = False: return a list of the fig
        handle followed by the 5 axes.

        In the top plot, the raw signal is plotted in black, and the
        red line indicates periods defined as oscillatory bursts.
        The highlighted regions indicate when each burst requirement
        was violated, color-coded consistently with the plots below.

        - blue: amplitude_fraction_threshold,
        - red: amplitude_consistency_threshold
        - yellow: period_consistency_threshold
        - green: monotonicity_threshold
    """

    # Normalize signal
    sig = zscore(sig)

    # Determine time array
    times = np.arange(0, len(sig) / fs, 1 /fs)

    if tlims is None:
        tlims = (times[0], times[-1])

    # Determine extrema labels
    if 'sample_trough' in df_shape.columns:
        center_e = 'trough'
        side_e = 'peak'
    else:
        center_e = 'peak'
        side_e = 'trough'

    # Limit to time periods of interest
    tidx = np.logical_and(times >= tlims[0], times < tlims[1])
    sig = sig[tidx]
    times = times[tidx]
    df_shape = df_shape[(df_shape['sample_last_' + side_e] > int(fs * tlims[0])) &
                        (df_shape['sample_next_' + side_e] < int(fs * tlims[1]))]
    df_shape['sample_last_' + side_e] = df_shape['sample_last_' + side_e] - int(fs * tlims[0])
    df_shape['sample_next_' + side_e] = df_shape['sample_next_' + side_e] - int(fs * tlims[0])
    df_shape['sample_' + center_e] = df_shape['sample_' + center_e] - int(fs * tlims[0])

    # Determine which samples are defined as bursting
    is_osc = np.zeros(len(sig), dtype=bool)
    df_osc = df_shape[df_shape['is_burst']]
    for _, cyc in df_osc.iterrows():
        samp_start_burst = cyc['sample_last_' + side_e]
        samp_end_burst = cyc['sample_next_' + side_e] + 1
        is_osc[samp_start_burst:samp_end_burst] = True

    # Plotting
    tpeaks = times[df_shape['sample_' + center_e]]
    sig_peaks = sig[df_shape['sample_' + center_e]]
    ttroughs = times[df_shape['sample_last_' + side_e]]
    sig_trough = sig[df_shape['sample_last_' + side_e]]

    if plot_only_result:
        # Plot the time series and indicate peaks and troughs
        _, ax = plt.subplots(figsize=figsize)

        plot_bursts(times, sig, is_osc, xlim=tlims, ax=ax,
                    title='Raw z-scored signal. Red trace indicates periods of bursting',
                    xlabel='Time (s)', ylabel='Voltage (normalized)', lw=2)

        plot_time_series([tpeaks, ttroughs], [sig_peaks, sig_trough], ax=ax,
                         xlim=tlims, colors=['m.', 'c.'])

        return ax

    # Plot the time series and indicate peaks and troughs
    fig = plt.figure(figsize=(figsize[0], 5*figsize[1]))
    ax1 = fig.add_subplot(5, 1, 1)

    plot_bursts(times, sig, is_osc, ax=ax1, xlim=tlims, ylim=(-4, 4), lw=2)

    plot_time_series([tpeaks, ttroughs], [sig_peaks, sig_trough], ax=ax1, colors=['m.', 'c.'],
                     ylabel='Black: Raw signal\nRed: oscillatory periods', xlabel='')

    # Highlight where burst detection parameters were violated
    # Use a different color for each burst detection parameter
    ax1.fill_between(ttroughs, -4, 400, where=df_shape['amp_fraction'] <
                     osc_kwargs['amplitude_fraction_threshold'],
                     interpolate=True, facecolor='blue', alpha=0.5, )
    ax1.fill_between(ttroughs, -4, 400, where=df_shape['amp_consistency'] <
                     osc_kwargs['amplitude_consistency_threshold'],
                     interpolate=True, facecolor='red', alpha=0.5)
    ax1.fill_between(ttroughs, -4, 400, where=df_shape['period_consistency'] <
                     osc_kwargs['period_consistency_threshold'],
                     interpolate=True, facecolor='yellow', alpha=0.5)
    ax1.fill_between(ttroughs, -4, 400, where=df_shape['monotonicity'] <
                     osc_kwargs['monotonicity_threshold'],
                     interpolate=True, facecolor='green', alpha=0.5)

    # Plot amplitude fraction threshold
    ax2 = fig.add_subplot(5, 1, 2)

    amp_fraction_thresh = osc_kwargs['amplitude_fraction_threshold']

    plot_time_series([tpeaks, tlims], [df_shape['amp_fraction'], [amp_fraction_thresh]*2],
                     ax=ax2, colors=['k.-', 'k--'], xlim=tlims, ylim=(-.02, 1.02), xlabel='',
                     ylabel=f"Band amplitude fraction\nthreshold={amp_fraction_thresh}",)

    ax2.fill_between(ttroughs, 0, 100, where=df_shape['amp_fraction'] < amp_fraction_thresh,
                     interpolate=True, facecolor='blue', alpha=0.5)

    # Plot amplitude consistency threshold
    ax3 = fig.add_subplot(5, 1, 3)

    amp_consist_thresh = osc_kwargs['amplitude_consistency_threshold']
    plot_time_series([tpeaks, tlims], [df_shape['amp_consistency'], [amp_consist_thresh]*2],
                     ax=ax3, colors=['k.-', 'k--'], xlim=tlims, ylim=(-.02, 1.02), xlabel='',
                     ylabel=f"Amplitude consistency\nthreshold={amp_consist_thresh}")

    ax3.fill_between(ttroughs, 0, 100, where=df_shape['amp_consistency'] < amp_consist_thresh,
                     interpolate=True, facecolor='red', alpha=0.5)

    # Plot period threshold
    ax4 = fig.add_subplot(5, 1, 4)

    period_thresh = osc_kwargs['period_consistency_threshold']
    plot_time_series([tpeaks, tlims], [df_shape['period_consistency'], [period_thresh]*2],
                     ax=ax4, colors=['k.-', 'k--'], xlim=tlims, ylim=(-.02, 1.02), xlabel='',
                     ylabel=f"Period consistency\nthreshold={period_thresh}")

    ax4.fill_between(ttroughs, 0, 100, where=df_shape['period_consistency'] < period_thresh,
                     interpolate=True, facecolor='yellow', alpha=0.5)

    # Plot monotonicity threshold
    ax5 = fig.add_subplot(5, 1, 5)

    mono_thresh = osc_kwargs['monotonicity_threshold']

    plot_time_series([tpeaks, tlims], [df_shape['monotonicity'], [mono_thresh]*2], ax=ax5,
                     colors=['k.-', 'k--'], xlim=tlims, ylim=(-.02, 1.02),
                     xlabel='Time (s)', ylabel=f"Monotonicity\nthreshold={mono_thresh}")

    ax5.fill_between(ttroughs, 0, 100, interpolate=True, facecolor='green', alpha=0.5,
                     where=df_shape['monotonicity'] < osc_kwargs['monotonicity_threshold'])

    # Remove x-axis labels ticks for all except bottom axis
    for axis in fig.axes:
        if fig.axes.index(axis) != len(fig.axes)-1:
            axis.set_xticks([])

    return [fig, ax1, ax2, ax3, ax4, ax5]
