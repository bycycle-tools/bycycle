"""Plot burst detection parameters."""

from itertools import cycle

import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib import rcParams

from neurodsp.plts import plot_time_series
from neurodsp.plts import plot_bursts


def plot_burst_detect_params(sig, fs, df_shape, osc_kwargs, tlims=(None, None),
                             figsize=(15, 3), plot_only_result=False):
    """Create a plot to study how the cycle-by-cycle burst detection
    algorithm determine bursting periods of a signal.

    Parameters
    ----------
    sig : 1d array
        Time series analyzed to compute ``df_shape``.
    fs : float
        Sampling rate, in Hz.
    df_shape : pandas DataFrame
        Dataframe output of :func:`~.compute_features`.
    osc_kwargs : dict
        Dictionary of thresholds for burst detection used in the function
        `features.compute_features()` using the kwarg ``burst_detection_kwargs``.
    tlims : tuple of (float, float), optional
        Start and stop times for plot.
    figsize : tuple of (float, float), optional
        Size of figure.
    plot_only_result : bool, optional, default: False
        If True, do not plot the subplots showing the parameters.

    Returns
    -------
    ax : matplotlib axis handle or list of axis handles
        If ``plot_only_result`` = True: return a plot of the burst
        detection in which periods with bursts are denoted in red.

        If ``plot_only_result`` = False: return a list of the fig
        handle followed by the 5 axes.

        In the top plot, the raw signal is plotted in black, and the
        red line indicates periods defined as oscillatory bursts.
        The highlighted regions indicate when each burst requirement
        was violated, color-coded consistently with the plots below.

        - blue: amplitude_fraction_threshold,
        - red: amplitude_consistency_threshold
        - yellow: period_consistency_threshold
        - green: monotonicity_threshold

    Examples
    --------
    See the `algorithm tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html#compute-features-of-each-cycle>`_.
    """

    # Normalize signal
    sig = zscore(sig)

    # Determine time array
    times = np.arange(0, len(sig) / fs, 1 /fs)

    if tlims is None or any(tlim is None for tlim in tlims):
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

    # Times and signals for peaks and troughs, for plotting
    tpeaks = times[df_shape['sample_' + center_e]]
    sig_peaks = sig[df_shape['sample_' + center_e]]
    ttroughs = times[df_shape['sample_last_' + side_e]]
    sig_trough = sig[df_shape['sample_last_' + side_e]]

    # Plot only the time series and indicate peaks and troughs
    if plot_only_result:

        _, ax = plt.subplots(figsize=figsize)

        plot_bursts(times, sig, is_osc, xlim=tlims, ax=ax,
                    title='Raw z-scored signal. Red trace indicates periods of bursting',
                    xlabel='Time (s)', ylabel='Voltage (normalized)', lw=2)

        plot_time_series([tpeaks, ttroughs], [sig_peaks, sig_trough], ax=ax,
                         xlim=tlims, colors=['m.', 'c.'])

        return ax

    # Burst detection parameter thresholds
    amp_fthresh = osc_kwargs['amplitude_fraction_threshold']
    amp_cthresh = osc_kwargs['amplitude_consistency_threshold']
    period_cthresh = osc_kwargs['period_consistency_threshold']
    mono_thresh = osc_kwargs['monotonicity_threshold']

    # Create figure and subplots
    fig = plt.figure(figsize=(figsize[0], 5*figsize[1]))
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)

    # Plot the time series and indicate peaks and troughs
    plot_bursts(times, sig, is_osc, ax=ax1, xlim=tlims, ylim=(-4, 4), lw=2)

    plot_time_series([tpeaks, ttroughs], [sig_peaks, sig_trough], ax=ax1, colors=['m.', 'c.'],
                     xlabel='', ylabel='Black: Raw signal\nRed: oscillatory periods')

    # Plot amplitude fraction threshold
    plot_time_series([tpeaks, tlims], [df_shape['amp_fraction'], [amp_fthresh]*2],
                     ax=ax2, colors=['k.-', 'k--'], xlim=tlims, ylim=(-.02, 1.02),
                     xlabel='', ylabel=f"Band amplitude fraction\nthreshold={amp_fthresh}")

    # Plot amplitude consistency threshold
    plot_time_series([tpeaks, tlims], [df_shape['amp_consistency'], [amp_cthresh]*2],
                     ax=ax3, colors=['k.-', 'k--'], xlim=tlims, ylim=(-.02, 1.02),
                     xlabel='', ylabel=f"Amplitude consistency\nthreshold={amp_cthresh}")

    # Plot period threshold
    plot_time_series([tpeaks, tlims], [df_shape['period_consistency'], [period_cthresh]*2],
                     ax=ax4, colors=['k.-', 'k--'], xlim=tlims, ylim=(-.02, 1.02),
                     xlabel='', ylabel=f"Period consistency\nthreshold={period_cthresh}")

    # Plot monotonicity threshold
    plot_time_series([tpeaks, tlims], [df_shape['monotonicity'], [mono_thresh]*2],
                     ax=ax5, colors=['k.-', 'k--'], xlim=tlims, ylim=(-.02, 1.02),
                     xlabel='Time (s)', ylabel=f"Monotonicity\nthreshold={mono_thresh}")

    # Highlight where burst detection parameters were violated
    # Use a different color for each burst detection parameter
    _plot_fill(ttroughs, [ax1, ax2], df_shape['amp_fraction'], amp_fthresh, color='blue')
    _plot_fill(ttroughs, [ax1, ax3], df_shape['amp_consistency'], amp_cthresh, color='red')
    _plot_fill(ttroughs, [ax1, ax4], df_shape['period_consistency'], period_cthresh, color='yellow')
    _plot_fill(ttroughs, [ax1, ax5], df_shape['monotonicity'], mono_thresh, color='green')

    # Remove x-axis labels ticks for all except bottom axis
    for axis in fig.axes:
        if fig.axes.index(axis) != len(fig.axes)-1:
            axis.set_xticks([])

    return [fig, ax1, ax2, ax3, ax4, ax5]


def plot_cyclepoints(sig, fs, extrema=(None, None), zerox=(None, None),
                     tlims=(None, None), figsize=(15, 3)):
    """ Plot extrema and/or zero crossings.

    Parameters
    ----------
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    extrema : tuple of (np.array, np.array), optional.
        Peak and trough extrema.
    zerox : tuple of (np.array, np.array), optional
        Rise and decay zero-crossings.
    tlims : tuple of (float, float), optional.
        Start and stop times for plot.

    Notes
    -----
    Either extrema or zerox must be defined.

    Examples
    --------
    Plot cyclepoints onto a simulated signal.

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.utils import create_times
    >>> from bycycle.cyclepoints import find_extrema, find_zerox
    >>> ps, ts = find_extrema(sig, fs=500, f_range=(8, 12))
    >>> zerox_rise, zerox_decay = find_zerox(sig, ps, ts)
    >>> plot_cyclepoints(sig, fs, extrema=(ps, ts), zerox=(zerox_rise, zerox_decay), tlims=(0, 2))
    """

    # Set the markersize for all points
    rcParams['lines.markersize'] = 15

    # Check arguments
    plot_extrema = False if any(arr is None for arr in extrema) else True
    plot_zerox = False if any(arr is None for arr in zerox) else True

    if plot_extrema is False and plot_zerox is False:
        raise TypeError('Either extrema, zerox, or both must be defined as (np.array, np.array).')

    # Determine time array/indices
    times = np.arange(0, len(sig) / fs, 1 /fs)

    if tlims is None or any(tlim is None for tlim in tlims):
        tlims = (times[0], times[-1])

    tidx = np.logical_and(times >= tlims[0], times < tlims[1])

    # Determine peaks and troughs indices
    if plot_extrema:

        ps = extrema[0]
        ts = extrema[1]

        tidx_ps = ps[np.logical_and(ps > tlims[0]*fs, ps < tlims[1]*fs)]
        tidx_ts = ts[np.logical_and(ts > tlims[0]*fs, ts < tlims[1]*fs)]

    # Determine rise and decay indices
    if plot_zerox:

        zerox_decay = zerox[0]
        zerox_rise = zerox[1]

        tidx_ds = zerox_decay[np.logical_and(zerox_decay > tlims[0]*fs, zerox_decay < tlims[1]*fs)]
        tidx_rs = zerox_rise[np.logical_and(zerox_rise > tlims[0]*fs, zerox_rise < tlims[1]*fs)]

    # Create figure
    _, ax = plt.subplots(figsize=figsize)

    # Plot either extrema, zerox, or both
    if plot_extrema and plot_zerox:

        sigs = [sig[tidx], sig[tidx_ps], sig[tidx_ts], sig[tidx_ds], sig[tidx_rs]]
        times = [times[tidx], times[tidx_ps], times[tidx_ts], times[tidx_ds], times[tidx_rs]]
        plot_time_series(times, sigs, ax=ax, colors=['k', 'b.', 'r.', 'm.', 'g.'], xlim=tlims, lw=2)

    elif plot_extrema:

        sigs = [sig[tidx], sig[tidx_ps], sig[tidx_ts]]
        times = [times[tidx], times[tidx_ps], times[tidx_ts]]
        plot_time_series(times, sigs, ax=ax, colors=['k', 'b.', 'r.'], xlim=tlims, lw=2)

    elif plot_zerox:

        sigs = [sig[tidx], sig[tidx_ds], sig[tidx_rs]]
        times = [times[tidx], times[tidx_ds], times[tidx_rs]]
        plot_time_series(times, sigs, ax=ax, colors=['k', 'm.', 'g.'], xlim=tlims, lw=2)

    return ax


def plot_cycle_features(df_burst, fs, labels=None, colors='krbgcmy', figsize=(5, 5)):
    """ Plot histograms of cycle features.

    Parameters
    ----------
    df_burst : list of pandas DataFrames
        Dataframe output(s) from :func:`~.compute_features`.
    fs : float
        Sampling rate, in Hz.
    labels : string or list of strings, optional
        Legend labels for each dataframe.
    figsize : tuple of (float, float), optional
        Size of figure.
    colors : string or list of strings
        Matplotlib color codes for historgrams.

    Examples
    --------
    See the `feature distribution example <http://bycycle-tools.github.io/bycycle/auto_examples/plot_theta_feature_distributions.html#plot-feature-distributions>`_.

    """

    fig = plt.figure(figsize=(figsize[0]*2, figsize[1]*2))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Axis styles
    ax1.set_xlabel('Cycle amplitude (mV)', size=15)
    ax1.set_ylabel('# cycles', size=15)

    ax2.set_xlabel('Cycle period (ms)', size=15)
    ax2.set_ylabel('# cycles', size=15)

    ax3.set_xlabel('Rise-decay asymmetry\n(fraction of cycle in rise period)', size=15)
    ax3.set_ylabel('# cycles', size=15)

    ax4.set_xlabel('Peak-trough asymmetry\n(fraction of cycle in peak period)', size=15)
    ax4.set_ylabel('# cycles', size=15)

    # Plot cycle features
    cycol = cycle(colors)

    for idx, df in enumerate(df_burst):

        color = next(cycol)

        ax1.hist(df['volt_amp'] / 1000, bins='auto', color=color, alpha=.5)

        if labels:
            ax1.legend(labels)

        ax2.hist(df['period'] / fs * 1000, bins='auto', color=color, alpha=.5)

        ax3.hist(df['time_rdsym'], bins='auto', color=color, alpha=.5)

        ax4.hist(df['time_ptsym'], bins='auto', color=color, alpha=.5)



def _plot_fill(times, axes, param, thresh, color):
    """ Fill a plot where a parameter falls below a given threshold.

    Parameters
    ----------
    times : 1d array
        Time definition for the time series to be plotted.
    sig : 1d array
        Time series to plot.
    axes: matplotlib axis or list of matplotlib axes
        Axes the burst detection parameter is plotted on.
    param : pandas series
        An indexed pandas dataframe containing the burst detection parameter.
    thresh : float
        Threshold for a burst detection parameter.
    color : string
        Color to fill plot.
    """

    for ax in axes:

        ylims = ax.get_ylim()
        ax.fill_between(times, ylims[0], ylims[1]*100, where=param < thresh,
                        interpolate=True, facecolor=color, alpha=0.5)
