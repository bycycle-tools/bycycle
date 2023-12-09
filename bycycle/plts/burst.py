"""Plot burst detection parameters."""

from itertools import cycle

import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

from neurodsp.plts import plot_time_series, plot_bursts
from neurodsp.plts.utils import savefig

from bycycle.plts.cyclepoints import plot_cyclepoints_df
from bycycle.utils import limit_df, limit_signal, get_extrema_df
from bycycle.utils.checks import check_param_range

###################################################################################################
###################################################################################################

@savefig
def plot_burst_detect_summary(df_features, sig, fs, threshold_kwargs, xlim=None,
                              figsize=(15, 3), plot_only_result=False, interp=True):
    """Plot the cycle-by-cycle burst detection parameters and burst detection summary.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe output of :func:`~.compute_features`. The df must contain sample indices (i.e.
        when ``return_samples = True``).
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    threshold_kwargs : dict
        Burst parameter keys and threshold value pairs, as defined in the 'threshold_kwargs'
        argument of :func:`.compute_features`.
    xlim : tuple of (float, float), optional, default: None
        Start and stop times for plot.
    figsize : tuple of (float, float), optional, default: (15, 3)
        Size of each plot.
    plot_only_result : bool, optional, default: False
        Plot only the signal and bursts, excluding burst parameter plots.
    interp : bool, optional, default: True
        If True, interpolates between given values. Otherwise, plots in a step-wise fashion.

    Notes
    -----

    - If plot_only_result = True: return a plot of the burst detection in which periods with bursts
      are denoted in red.

    - If plot_only_result = False: return a list of the fig handle followed by the 5 axes.

    - In the top plot, the raw signal is plotted in black, and the red line indicates periods
      defined as oscillatory bursts. The highlighted regions indicate when each burst requirement
      was violated, color-coded consistently with the plots below.

      - blue: amp_fraction_threshold
      - red: amp_consistency_threshold
      - yellow: period_consistency_threshold
      - green: monotonicity_threshold

    Examples
    --------
    Plot the burst detection summary of a bursting signal:

    >>> from bycycle.features import compute_features
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> threshold_kwargs = {'amp_fraction_threshold': 0., 'amp_consistency_threshold': .5,
    ...                     'period_consistency_threshold': .5, 'monotonicity_threshold': .8}
    >>> df_features = compute_features(sig, fs, f_range=(8, 12), threshold_kwargs=threshold_kwargs)
    >>> plot_burst_detect_summary(df_features, sig, fs, threshold_kwargs)
    """

    # Ensure arguments are within valid range
    check_param_range(fs, 'fs', (0, np.inf))

    # Normalize signal
    sig_full = zscore(sig)
    times_full = np.arange(0, len(sig_full) / fs, 1 / fs)

    # Limit arrays and dataframe
    if xlim is not None:
        sig, times = limit_signal(times_full, sig_full, start=xlim[0], stop=xlim[1])
        df_features = limit_df(df_features, fs, start=xlim[0], stop=xlim[1], reset_indices=False)
    else:
        sig, times, = sig_full, times_full

    # Determine if peak of troughs are the sides of an oscillation
    _, side_e = get_extrema_df(df_features)

    # Remove this kwarg since it isn't stored cycle by cycle in the df (nothing to plot)
    thresholds = threshold_kwargs.copy()
    if 'min_n_cycles' in thresholds.keys():
        del thresholds['min_n_cycles']

    n_kwargs = len(thresholds.keys())

    # Create figure and subplots
    if plot_only_result:
        fig, axes = plt.subplots(figsize=figsize, nrows=1)
        axes = [axes]
    else:
        fig, axes = plt.subplots(figsize=(figsize[0], figsize[1]*(n_kwargs+1)),
                                 nrows=n_kwargs+1, sharex=True)

    # Determine which samples are defined as bursting
    is_osc = np.zeros(len(sig), dtype=bool)
    df_osc = df_features.loc[df_features['is_burst']]
    start = 0 if xlim is None else xlim[0]

    for cyc in df_osc.to_dict('records'):

        samp_start_burst = int(cyc['sample_last_' + side_e]) - int(fs * start)
        samp_end_burst = int(cyc['sample_next_' + side_e] + 1) - int(fs * start)

        is_osc[samp_start_burst:samp_end_burst] = True

    # Plot bursts with extrema points
    xlabel = 'Time (s)' if len(axes) == 1 else ''

    plot_bursts(times, sig, is_osc, ax=axes[0], lw=2,
                labels=['Signal', 'Bursts'], xlabel='', ylabel='')

    plot_cyclepoints_df(df_features, sig_full, fs, xlim=xlim, ax=axes[0], plot_zerox=False,
                        plot_sig=False, xlabel=xlabel, ylabel='Voltage\n(normalized)',
                        colors=['m', 'c'])

    # Plot each burst param
    colors = cycle(['blue', 'red', 'yellow', 'green', 'cyan', 'magenta', 'orange'])

    for idx, osc_key in enumerate(thresholds.keys()):

        column = osc_key.replace('_threshold', '')

        color = next(colors)

        # Highlight where a burst param falls below threshold
        for cyc in df_features.to_dict('records'):

            last_cyc = int(cyc['sample_last_' + side_e]) - int(fs * start)
            next_cyc = int(cyc['sample_next_' + side_e]) - int(fs * start)
            if cyc[column] < threshold_kwargs[osc_key] and last_cyc > 0:
                axes[0].axvspan(times[last_cyc], times[next_cyc],
                 alpha=0.5, color=color, lw=0)

        # Plot each burst param on separate axes
        if not plot_only_result:

            ylabel = column.replace('_', ' ').capitalize()
            xlabel = 'Time (s)' if idx == n_kwargs-1 else ''

            plot_burst_detect_param(df_features, sig_full, fs, column, thresholds[osc_key],
                                    xlim=xlim, figsize=figsize, ax=axes[idx+1], xlabel=xlabel,
                                    ylabel=ylabel, color=color, interp=interp)


@savefig
def plot_burst_detect_param(df_features, sig, fs, burst_param, thresh,
                            xlim=None, interp=True, ax=None, **kwargs):
    """Plot a burst detection parameter and threshold.

    Parameters
    ----------
    df_features : pandas.DataFrame
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
    xlim : tuple of (float, float), optional, default: None
        Start and stop times for plot.
    interp : bool, optional, default: True
        Interpolates points if true.
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
    - ``color``: str, default: 'r'.

      - Note: ``color`` here is the fill color, rather than line color.

    Examples
    --------
    Plot the monotonicity of a bursting signal:

    >>> from bycycle.features import compute_features
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> threshold_kwargs = {'amp_fraction_threshold': 0., 'amp_consistency_threshold': .5,
    ...                     'period_consistency_threshold': .5, 'monotonicity_threshold': .8}
    >>> df_features = compute_features(sig, fs, f_range=(8, 12),
    ...                                            threshold_kwargs=threshold_kwargs)
    >>> plot_burst_detect_param(df_features, sig, fs, 'monotonicity', .8)
    """

    # Ensure arguments are within valid range
    check_param_range(fs, 'fs', (0, np.inf))

    # Set default kwargs
    figsize = kwargs.pop('figsize', (15, 3))
    xlabel = kwargs.pop('xlabel', 'Time (s)')
    ylabel = kwargs.pop('ylabel', burst_param)
    color = kwargs.pop('color', 'r')

    # Determine time array and limits
    times = np.arange(0, len(sig) / fs, 1 / fs)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Determine extrema strings
    center_e, side_e = get_extrema_df(df_features)

    if xlim is not None:

        # Limit dataframe, sig and times
        df_features = limit_df(df_features, fs, start=xlim[0], stop=xlim[1])
        sig, times = limit_signal(times, sig, start=xlim[0], stop=xlim[1])

        # Remove start / end cycles that tlims falls between
        df_features = df_features[(df_features['sample_last_' + side_e] >= 0) & \
                                  (df_features['sample_next_' + side_e] < xlim[1]*fs)]

    # Plot burst param
    if interp:

        plot_time_series([times[df_features['sample_' + center_e]], (times[0], times[-1])],
                         [df_features[burst_param], [thresh]*2], ax=ax, colors=['k', 'k'],
                         ls=['-', '--'], marker=["o", "None"], xlabel=xlabel,
                         ylabel="{0:s}\nthreshold={1:.2f}".format(ylabel, thresh), **kwargs)

    else:

        # Create steps, from side to side of each cycle, and set the y-value
        #   to the burst parameter value for that cycle
        side_times = np.array([])
        side_param = np.array([])

        for cyc in df_features.to_dict('records'):

            # Get the times for the last and next side of a cycle
            side_times = np.append(side_times, [times[int(cyc['sample_last_' + side_e])],
                                                times[int(cyc['sample_next_' + side_e])]])

            # Set the y-value, from side to side, to the burst param for each cycle
            side_param = np.append(side_param, [cyc[burst_param]] * 2)

        plot_time_series([side_times, (times[0], times[-1])], [side_param, [thresh]*2], ax=ax,
                         colors=['k', 'k'], ls=['-', '--'], marker=["o", "None"], xlabel=xlabel,
                         ylabel="{0:s}\nthreshold={1:.2f}".format(ylabel, thresh), **kwargs)

    # Highlight where param falls below threshold
    for cyc in df_features.to_dict('records'):

        if cyc[burst_param] <= thresh:

            ax.axvspan(times[int(cyc['sample_last_' + side_e])],
                       times[int(cyc['sample_next_' + side_e])],
                       alpha=0.5, color=color, lw=0)
