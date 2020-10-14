"""Plot extrema and zero-crossings."""

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.plts import plot_time_series
from neurodsp.plts.utils import savefig

from bycycle.utils.checks import check_param
from bycycle.utils import limit_signal, get_extrema_df

###################################################################################################
###################################################################################################

@savefig
def plot_cyclepoints_df(df_samples, sig, fs, plot_sig=True, plot_extrema=True,
                        plot_zerox=True, xlim=None, ax=None, **kwargs):
    """Plot extrema and/or zero-crossings from a DataFrame.

    Parameters
    ----------
    df_samples: pandas.DataFrame
        Dataframe output of :func:`~.compute_cyclepoints`.
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    plot_sig : bool, optional, default: True
        Whether to also plot the raw signal.
    plot_extrema :  bool, optional, default: True
        Whether to plots the peaks and troughs.
    plot_zerox :  bool, optional, default: True
        Whether to plots the zero-crossings.
    xlim : tuple of (float, float), optional
        Start and stop times.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments to pass into `plot_time_series`.

    Notes
    -----
    Default keyword arguments include:

    - ``figsize``: tuple of (float, float), default: (15, 3)
    - ``xlabel``: str, default: 'Time (s)'
    - ``ylabel``: str, default: 'Voltage (uV)

    Examples
    --------
    Plot cyclepoints using a dataframe from :func:`~.compute_cyclepoints`:

    >>> from bycycle.features import compute_cyclepoints
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_samples = compute_cyclepoints(sig, fs, f_range=(8, 12))
    >>> plot_cyclepoints_df(df_samples, sig, fs)
    """

    # Ensure arguments are within valid range
    check_param(fs, 'fs', (0, np.inf))

    # Determine extrema/zero-crossings from dataframe
    center_e, side_e = get_extrema_df(df_samples)

    peaks, troughs, rises, decays = [None]*4

    if plot_extrema:

        peaks = df_samples['sample_' + center_e].values
        troughs = np.append(df_samples['sample_last_' + side_e].values,
                            df_samples['sample_next_' + side_e].values[-1])
    if plot_zerox:

        rises = df_samples['sample_zerox_rise'].values
        decays = df_samples['sample_zerox_decay'].values

    plot_cyclepoints_array(sig, fs, peaks=peaks, troughs=troughs, rises=rises,
                           decays=decays, plot_sig=plot_sig, xlim=xlim, ax=ax, **kwargs)


@savefig
def plot_cyclepoints_array(sig, fs, peaks=None, troughs=None, rises=None, decays=None,
                           plot_sig=True, xlim=None, ax=None, **kwargs):
    """Plot extrema and/or zero-crossings from arrays.

    Parameters
    ----------
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    peaks : 1d array, optional
        Peak signal indices from :func:`.find_extrema`.
    troughs : 1d array, optional
        Trough signal indices from :func:`.find_extrema`.
    rises : 1d array, optional
        Zero-crossing rise indices from :func:`~.find_zerox`.
    decays : 1d array, optional
        Zero-crossing decay indices from :func:`~.find_zerox`.
    plot_sig : bool, optional, default: True
        Whether to also plot the raw signal.
    xlim : tuple of (float, float), optional
        Start and stop times.
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

    Examples
    --------
    Plot cyclepoints using arrays from :func:`.find_extrema` and  :func:`~.find_zerox`:

    >>> from bycycle.cyclepoints import find_extrema, find_zerox
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> peaks, troughs = find_extrema(sig, fs, f_range=(8, 12), boundary=0)
    >>> rises, decays = find_zerox(sig, peaks, troughs)
    >>> plot_cyclepoints_array(sig, fs, peaks=peaks, troughs=troughs, rises=rises, decays=decays)
    """

    # Ensure arguments are within valid range
    check_param(fs, 'fs', (0, np.inf))

    # Set times and limits
    times = np.arange(0, len(sig) / fs, 1 / fs)
    xlim = (times[0], times[-1]) if xlim is None else xlim

    # Restrict sig and times to xlim
    sig, times = limit_signal(times, sig, start=xlim[0], stop=xlim[1])

    # Set default kwargs
    figsize = kwargs.pop('figsize', (15, 3))
    xlabel = kwargs.pop('xlabel', 'Time (s)')
    ylabel = kwargs.pop('ylabel', 'Voltage (uV)')
    default_colors = ['b', 'r', 'g', 'm']

    # Extend plotting based on given arguments
    x_values = []
    y_values = []
    colors = ['k']

    for idx, points in enumerate([peaks, troughs, rises, decays]):

        if points is not None:

            # Limit times and shift indices of cyclepoints (cps)
            cps = points[(points >= xlim[0]*fs) & (points < xlim[1]*fs)]
            cps = cps - int(xlim[0]*fs)

            y_values.append(sig[cps])
            x_values.append(times[cps])
            colors.append(default_colors[idx])

    # Allow custom colors to overwrite default
    colors = kwargs.pop('colors', colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if plot_sig:
        plot_time_series(times, sig, colors=colors[0], ax=ax)
        colors = colors[1:]

    plot_time_series(x_values, y_values, ax=ax, xlabel=xlabel, ylabel=ylabel,
                     colors=colors, marker='o', ls='', **kwargs)
