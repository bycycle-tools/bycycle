"""Plot burst detection parameters."""

from itertools import cycle

import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib import rcParams

from neurodsp.plts import plot_time_series
from neurodsp.plts import plot_bursts


def plot_burst_detect_summary(df, sig, fs, osc_kwargs, tlims=None,
                              figsize=(15,3), plot_only_result=False):
    """
    Create a plot to study how the cycle-by-cycle burst detection
    algorithm determine bursting periods of a signal.

    Parameters
    ----------
    sig : 1d array
        Time series analyzed to compute ``df``.
    fs : float
        Sampling rate, in Hz.
    df : pandas DataFrame
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
    
    Notes
    -----
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

    # Set times and limits
    times = np.arange(0, len(sig) / fs, 1 /fs)

    if tlims is None:
        tlims = (times[0], times[-1])

    # Determine oscillation centers
    center = 'peak' if 'sample_peak' in df.columns else 'trough'
    side = 'trough' if center == 'peak' else 'peak'

    df, sig, times = _limit_df(df, sig, times, fs, tlims, center=center)
    
    # Create figure and subplots
    if plot_only_result:
        _, [axes] = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=(figsize[0], 5*figsize[1]))
        axes = [fig.add_subplot(5, 1, row) for row in range(1, 6)]

    # Determine which samples are defined as bursting
    is_osc = np.zeros(len(sig), dtype=bool)
    
    for _, cyc in df[df['is_burst']].iterrows():
        samp_start_burst = cyc['sample_last_' + side]
        samp_end_burst = cyc['sample_next_' + side] + 1
        is_osc[samp_start_burst:samp_end_burst] = True

    # Plot bursts and extrema points
    plot_bursts(times, sig, is_osc, xlim=tlims, ax=axes[0], ylim=(min(sig), max(sig)),
                title='Black: Raw signal\nRed: oscillatory periods',
                xlabel='Time (s)', ylabel='Voltage (normalized)', lw=2)
    
    plot_cycle_points(df, sig, fs, tlims=tlims, ax=axes[0], plot_zerox=False, plot_sig=False)

    if not plot_only_result:

        # Column labels for burst params
        df_columns = ['amp_fraction', 'amp_consistency', 'period_consistency', 'monotonicity']

        highlight = ['blue', 'red', 'yellow', 'green']

        # Plot each burst param
        for idx, _ in enumerate(df_columns):

            # Create ylabel from osc kwargs
            key = list(osc_kwargs.keys())[idx]
            ylabel = key.replace('_threshold', '').replace('_', ' ').capitalize()
            
            plot_burst_detect_param(df, sig, fs, df_columns[idx], osc_kwargs[key], tlims=tlims,
                                    figsize=figsize, ax=[axes[idx+1], axes[0]], ylabel=ylabel,
                                    xlabel='Time (s)', highlight=highlight[idx])


def plot_burst_detect_param(df, sig, fs, param, thresh, tlims=None, figsize=(15, 3), ax=None,
                            ylabel=None, xlabel='Time (s)', highlight='b'):
    """Plot a burst detection parameter and threshold.

    Parameter
    ---------
    df : pandas DataFrame
        Dataframe output of :func:`~.compute_features`.
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    param : string
        Column name of the parameter of interest. The parameter must be in ``df``.
    thresh : float
        Burst parameters defined in the ``burst_detection_kwargs`` argument of
        :func:`~.compute_features`.
    tlims : tuple of (float, float), optional
        Start and stop times for plot.
    figsize : tuple of (float, float), optional
        Size of figure.
    ax : matplotlib axis, optional
        Axis to plot figure.
    ylabel : string, optional
        Label for the y-axis.
    xlabel : string, optional
        Label for the x-axis.
    highlight : string, optional
        Color to highlight where the burst paramter violates ``thresh``.
    """

    rcParams['lines.markersize'] = 12

    # Set times and limits
    times = np.arange(0, len(sig) / fs, 1 /fs)

    if not tlims:
        tlims = (times[0], times[-1])

    if not ax:
        _, ax = plt.subplots(figsize=(figsize))
    
    axes = [ax] if not isinstance(ax, list) else ax

    if not ylabel:
        ylabel = param

    # Determine oscillation centers
    center = 'peak' if 'sample_peak' in df.columns else 'trough'

    df, _, _ = _limit_df(df, sig, times, fs, tlims, center=center)
    
    times = times[df['sample_' + center]]
    
    sig = sig[df['sample_' + center]]
    
    # Pad the ylims by +/- half a std, so y-axis fov isn't too narrow
    ylims = (np.nanmin(df[param]) - np.std(df[param])/2,
             np.nanmax(df[param]) + np.std(df[param])/2)

    # Plot the burst parameter
    plot_time_series([times, tlims], [df[param], [thresh]*2],
                     ax=axes[0], colors=['k.-', 'k--'], xlim=tlims, ylim=ylims,
                     ylabel="{0:s}\nthreshold={1:.2f}".format(ylabel, thresh))
    
    # Fill regions where parameter falls below threshold
    times_interp = np.linspace(times[0], times[-1], 50*len(times))
    yvals_interp = np.interp(times_interp, times, df[param])
    
    for ax in axes:
        ax.fill_between(times_interp, ax.get_ylim()[0], ax.get_ylim()[1], where=(yvals_interp < thresh),
                        color=highlight, alpha=0.5)


def plot_cycle_points(df, sig, fs, tlims=None, figsize=(15, 3), ax=None, plot_sig=True,
                      plot_extrema=True, plot_zerox=True):
    """Plot extrema and/or zerox.

    Parameter
    ---------
    df : pandas DataFrame
        Dataframe output of :func:`~.compute_features`.
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    tlims : tuple of (float, float), optional.
        Start and stop times for plot.
    figsize : tuple of (float, float), optional
        Size of figure.
    ax : matplotlib axis, optional
        Axis to plot figure.
    plot_sig : boolean, optional
        Plots the raw signal.
    plot_extrema : boolean, optional
        Plots peaks and troughs.
    plot_zerox : boolean, optional
        Plots zero-crossings.
    """

    rcParams['lines.markersize'] = 12

    # Set times and limits
    times = np.arange(0, len(sig) / fs, 1 /fs)

    if not tlims:
        tlims = (times[0], times[-1])

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    # Determine extrema/zero-crossing times and signals
    center = 'peak' if 'sample_peak' in df.columns else 'trough'
    side = 'trough' if center == 'peak' else 'peak'
    
    df, sig_trim, times_trim = _limit_df(df, sig, times, fs, tlims, center=center)
    
    # Extend plotting based on arguments
    x_values = []
    y_values = []
    colors = []

    if plot_sig:
        x_values.extend([times_trim])
        y_values.extend([sig_trim])
        colors.extend(['k'])

    if plot_extrema:
        x_values.extend([times[df['sample_' + center]], times[df['sample_last_' + side]]])
        y_values.extend([sig[df['sample_' + center]], sig[df['sample_last_' + side]]])
        colors.extend(['m.', 'c.'])
     
    if plot_zerox:
        x_values.extend([times[df['sample_zerox_decay']], times[df['sample_zerox_rise']]])
        y_values.extend([sig[df['sample_zerox_decay']], sig[df['sample_zerox_rise']]])
        colors.extend(['b.', 'g.'])

    # Overlay extrema/zero-crossing onto the signal
    plot_time_series(x_values, y_values, ax=ax, xlim=tlims, colors=colors)


def _limit_df(df, sig, times, fs, tlims, center='peak'):
    """Limit dataframe to be within tlims."""

    side = 'trough' if center == 'peak' else 'peak'

    # Limit times and sig to tlim
    tidx = np.logical_and(times >= tlims[0], times < tlims[1])
    sig = sig[tidx]
    times = times[tidx]

    # Limit dataframe to tlims
    df = df[(df['sample_last_' + side] >= int(fs * tlims[0])) &
            (df['sample_next_' + side] < int(fs * tlims[1]))]

    return df, sig, times
