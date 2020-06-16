"""Plot burst detection parameters."""

from itertools import cycle

import numpy as np
from scipy.stats import zscore

import matplotlib.pyplot as plt

from neurodsp.plts import plot_time_series, plot_bursts

from bycycle.plts.cyclepoints import plot_cyclepoints
from bycycle.plts.utils import apply_tlims, get_extrema

###################################################################################################
###################################################################################################

def plot_burst_detect_summary(df, sig, fs, osc_kwargs, tlims=None, figsize=(15, 3),
                              plot_only_result=False, burst_params=None, interp=True):
    """
    Create a plot to study how the cycle-by-cycle burst detection
    algorithm determines bursting periods of a signal.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe output of :func:`~.compute_features`.
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    osc_kwargs : dict
        Burst parameter keys and threshold value pairs, as defined in the 'burst_detection_kwargs'
        argument of :func:`.compute_features`.
    tlims : tuple of (float, float), optional, default: None
        Start and stop times for plot.
    figsize : tuple of (float, float), optional, default: (15, 3)
        Size of each plot.
    plot_only_result : bool, optional, default: False
        Plot only the signal and bursts, excluding burst parameter plots.
    burst_params : list of str, optional, default: None
        The names of the ``df`` burst detection parameter columns. These names may differ from the
        keys in ``osc_kwargs``. This is only needed for non-default burst parameters.
    interp : bool
        Interpolates points if true.


    Notes
    -----

    - If plot_only_result = True: return a plot of the burst detection in which periods with bursts
      are denoted in red.

    - If plot_only_result = False: return a list of the fig handle followed by the 5 axes.

    - In the top plot, the raw signal is plotted in black, and the red line indicates periods
      defined as oscillatory bursts. The highlighted regions indicate when each burst requirement
      was violated, color-coded consistently with the plots below.

     - blue: amplitude_fraction_threshold
     - red: amplitude_consistency_threshold
     - yellow: period_consistency_threshold
     - green: monotonicity_threshold

    - Custom burst parameters may be defined in ``osc_kwargs`` and ``burst_params``, but the order
      is expected to be the same for both arguments.

    """

    # Normalize signal
    sig = zscore(sig)

    # Determine time array and limits
    times = np.arange(0, len(sig) / fs, 1 / fs)
    tlims = (times[0], times[-1]) if tlims is None else tlims

    # Determine if peak of troughs are the sides of an oscillation
    _, side_e = get_extrema(df)

    df_lim, sig_lim, times_lim = apply_tlims(df, sig, times, fs, tlims)

    if burst_params is None:
        burst_params = ['amp_fraction', 'amp_consistency', 'period_consistency', 'monotonicity']

    # Create figure and subplots
    if plot_only_result:
        fig, axes = plt.subplots(figsize=figsize, nrows=1)
        axes = [axes]
    else:
        fig, axes = plt.subplots(figsize=(figsize[0], figsize[1]*len(burst_params)+1),
                                 nrows=len(burst_params)+1, sharex=True)

    # Determine which samples are defined as bursting
    is_osc = np.zeros(len(sig_lim), dtype=bool)
    df_osc =  df_lim[df_lim['is_burst']]

    for _, cyc in df_osc.iterrows():

        samp_start_burst = cyc['sample_last_' + side_e]
        samp_end_burst = cyc['sample_next_' + side_e] + 1
        is_osc[samp_start_burst:samp_end_burst] = True

    # Plot bursts with extrema points
    xlabel = 'Time (s)' if len(axes) == 1 else ''

    plot_bursts(times_lim, sig_lim, is_osc, ax=axes[0], lw=2,
                labels=['Signal', 'Bursts'], xlabel='', ylabel='')

    plot_cyclepoints(df, sig, fs, ax=axes[0], tlims=tlims, plot_zerox=False,
                     plot_sig=False, xlabel=xlabel, ylabel='Voltage\n(normalized)')

    # Plot each burst param
    colors = cycle(['blue', 'red', 'yellow', 'green', 'cyan', 'magenta', 'orange'])

    for idx, column in enumerate(burst_params):

        osc_key = list(osc_kwargs.keys())[idx]

        color = next(colors)

        # Highlight where a burst param falls below threshold
        for _, cyc in df.iterrows():

            if cyc[column] < osc_kwargs[osc_key]:
                axes[0].axvspan(times[cyc['sample_last_' + side_e]],
                                times[cyc['sample_next_' + side_e]],
                                alpha=0.5, color=color, lw=0)

        # Plot each burst param on separate axes
        if not plot_only_result:

            ylabel = osc_key.replace('_threshold', '').replace('_', ' ').capitalize()
            xlabel = 'Time (s)' if idx == len(burst_params)-1 else ''

            plot_burst_detect_param(df, sig, fs, burst_params[idx], osc_kwargs[osc_key],
                                    figsize=figsize, ax=axes[idx+1], tlims=tlims,
                                    xlabel=xlabel, ylabel=ylabel, color=color, interp=interp)


def plot_burst_detect_param(df, sig, fs, burst_param, thresh, tlims=None,
                            ax=None, interp=True, **kwargs):
    """Plot a burst detection parameter and threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe output of :func:`~.compute_features`.
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    burst_param : str
        Column name of the parameter of interest in ``df``.
    thresh : float
        The burst parameter threshold. Parameter values greater
        than ``thresh`` are considered bursts.
    tlims : tuple of (float, float), optional, default: None
        Start and stop times for plot.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    interp : bool
        Interpolates points if true.
    **kwargs
        Keyword arguments to pass into `plot_time_series`.

    Notes
    -----
    Default keyword arguments include:

    - ``figsize``: tuple of (float, float), default: (15, 3)
    - ``xlabel``: str, default: 'Time (s)'
    - ``ylabel``: str, default: 'Voltage (uV)
    - ``color``: str, default: 'r'.

      - Note: ``color`` here is the fill color, rather than line color.

    """

    # Set default kwargs
    figsize = (15, 3) if 'figsize' not in kwargs.keys() else kwargs.pop('figsize')
    xlabel = 'Time (s)' if 'xlabel' not in kwargs.keys() else kwargs.pop('xlabel')
    ylabel = burst_param if 'ylabel' not in kwargs.keys() else kwargs.pop('ylabel')
    color = 'r' if 'color' not in kwargs.keys() else kwargs.pop('color')

    # Determine time array and limits
    times = np.arange(0, len(sig) / fs, 1 / fs)
    tlims = (times[0], times[-1]) if tlims is None else tlims

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Determine extrema strings
    center_e, side_e = get_extrema(df)

    df, sig, times = apply_tlims(df, sig, times, fs, tlims)

    # Plot burst param
    if interp:

        plot_time_series([times[df['sample_' + center_e]], tlims], [df[burst_param], [thresh]*2],
                         ax=ax, colors=['k.-', 'k--'], xlim=tlims, xlabel=xlabel,
                         ylabel="{0:s}\nthreshold={1:.2f}".format(ylabel, thresh), **kwargs)

    else:

        side_times = np.array([])
        side_param = np.array([])

        for _, cyc in df.iterrows():

            side_times = np.append(side_times, [times[cyc['sample_last_' + side_e]],
                                                times[cyc['sample_next_' + side_e]]])

            side_param = np.append(side_param, [cyc[burst_param]]*2)

        plot_time_series([side_times, tlims], [side_param, [thresh]*2],
                         ax=ax, colors=['k.-', 'k--'], xlim=tlims, xlabel=xlabel,
                         ylabel="{0:s}\nthreshold={1:.2f}".format(ylabel, thresh), **kwargs)

    # Highlight where param falls below threshold
    for _, cyc in df.iterrows():

        if cyc[burst_param] < thresh:

            ax.axvspan(times[cyc['sample_last_' + side_e]], times[cyc['sample_next_' + side_e]],
                       alpha=0.5, color=color, lw=0)

