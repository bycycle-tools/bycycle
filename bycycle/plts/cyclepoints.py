"""Plot extrema and zero-crossings."""

import numpy as np

import matplotlib.pyplot as plt

from neurodsp.plts import plot_time_series
from neurodsp.plts.utils import savefig

from bycycle.utils import limit_df, limit_sig_times, get_extrema

###################################################################################################
###################################################################################################

@savefig
def plot_cyclepoints_df(df, sig, fs, xlim=None, plot_sig=True, plot_extrema=True,
                        plot_zerox=True, ax=None, **kwargs):
    """Plot extrema and/or zero-crossings using a dataframe to define points.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe output of :func:`~.compute_features`.
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    xlim : tuple of (float, float), optional, default: None
        Start and stop times.
    plot_sig : bool, optional, default: True
        Plots the raw signal.
    plot_extrema :  bool, optional, default: True
        Plots peaks and troughs.
    plot_zerox :  bool, optional, default: True
        Plots zero-crossings.
    ax : matplotlib.Axes, optional, default: None
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments to pass into `plot_time_series`.

    Notes
    -----
    Default keyword arguments include:

    - ``figsize``: tuple of (float, float), default: (15, 3)
    - ``xlabel``: str, default: 'Time (s)'
    - ``ylabel``: str, default: 'Voltage (uV)

    """

    # Set default kwargs
    figsize = kwargs.pop('figsize', (15, 3))
    xlabel = kwargs.pop('xlabel', 'Time (s)')
    ylabel = kwargs.pop('ylabel', 'Voltage (uV)')

    # Set times and limits
    times = np.arange(0, len(sig) / fs, 1 / fs)
    xlim = (times[0], times[-1]) if xlim is None else xlim

    # Determine extrema/zero-crossing times and signals
    center_e, side_e = get_extrema(df)

    df = limit_df(df, fs, xlim)
    sig, times = limit_sig_times(sig, times, xlim)

    # Extend plotting based on given arguments
    x_values = []
    y_values = []
    colors = ['k']

    if plot_extrema:

        ps = df['sample_' + center_e].values
        ts = df['sample_last_' + side_e].values
        ts = np.append(ts, df['sample_next_' + side_e].values[-1])

        # Cycles are kept if any cyclepoint is within tlims, this ensures all points to be plotted
        #   are within the x limits.
        ps = ps[(ps >= 0) & (ps < (xlim[1] - xlim[0]) * fs)]
        ts = ts[(ts >= 0) & (ts < (xlim[1] - xlim[0]) * fs)]

        x_values.extend([times[ps], times[ts]])
        y_values.extend([sig[ps], sig[ts]])
        colors.extend(['b', 'r'])

    if plot_zerox:
        zerox_rise = df['sample_zerox_rise'].values
        zerox_rise = zerox_rise[(zerox_rise >= 0) & (zerox_rise < (xlim[1] - xlim[0]) * fs)]
        zerox_decay = df['sample_zerox_decay'].values
        zerox_decay = zerox_decay[(zerox_decay >= 0) & (zerox_decay < (xlim[1] - xlim[0]) * fs)]

        x_values.extend([times[zerox_rise], times[zerox_decay]])
        y_values.extend([sig[zerox_rise], sig[zerox_decay]])
        colors.extend(['g', 'm'])

    # Allow custom colors to overwrite default
    colors = kwargs.pop('colors', colors)

    # Plot cycle points
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if plot_sig:
        plot_time_series(times, sig, colors=colors[0], ax=ax)

    colors = colors[1:] if plot_sig is True else colors

    plot_time_series(x_values, y_values, ax=ax, colors=colors, xlabel=xlabel,
                     ylabel=ylabel, marker='o', ls='', **kwargs)


@savefig
def plot_cyclepoints_array(sig, fs, xlim=None, ps=None, ts=None, zerox_rise=None,
                           zerox_decay=None, ax=None, **kwargs):
    """Plot extrema and/or zero-crossings using arrays to define points.

    Parameters
    ----------
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.fs
    xlim : tuple of (float, float), optional, default: None
        Start and stop times.
    ps : 1d array, optional, default: None
        Peak signal indices from :func:`.find_extrema`.
    ts : 1d array, optional, default: None
        Trough signal indices from :func:`.find_extrema`.
    zerox_rise : 1d array, optional, default: None
        Zero-crossing rise indices from :func:`~.find_zerox`.
    zerox_decay : 1d array, optional, default: None
        Zero-crossing decay indices from :func:`~.find_zerox`.
    ax : matplotlib.Axes, optional, default: None
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments to pass into `plot_time_series`.

    Notes
    -----
    Default keyword arguments include:

    - ``figsize``: tuple of (float, float), default: (15, 3)
    - ``xlabel``: str, default: 'Time (s)'
    - ``ylabel``: str, default: 'Voltage (uV)
    - ``colors``: list, default: ['k', 'b', 'r', 'g', 'm']

    """

    # Set times and limits
    times = np.arange(0, len(sig) / fs, 1 / fs)
    xlim = (times[0], times[-1]) if xlim is None else xlim

    # Restrict sig and times to xlim
    sig, times = limit_sig_times(sig, times, xlim)

    # Set default kwargs
    figsize = kwargs.pop('figsize', (15, 3))
    xlabel = kwargs.pop('xlabel', 'Time (s)')
    ylabel = kwargs.pop('ylabel', 'Voltage (uV)')
    default_colors = ['b', 'r', 'g', 'm']

    # Extend plotting based on given arguments
    x_values = []
    y_values = []
    colors = ['k']

    for idx, points in enumerate([ps, ts, zerox_rise, zerox_decay]):

        if points is not None:

            # Limit times and shift indices of cyclepoints (cps)
            cps = points[(points > xlim[0]*fs) & (points <= xlim[1]*fs)]
            cps = cps - int(xlim[0]*fs)

            y_values.append(sig[cps])
            x_values.append(times[cps])
            colors.append(default_colors[idx])

    # Allow custom colors to overwrite default
    colors = kwargs.pop('colors', colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    plot_time_series(times, sig, ax=ax, colors=colors[0])
    plot_time_series(x_values, y_values, ax=ax, xlabel=xlabel, ylabel=ylabel,
                     colors=colors[1:], marker='o', ls='', **kwargs)
