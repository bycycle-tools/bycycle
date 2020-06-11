"""Plot extrema and zero-crossings."""

import numpy as np
from scipy.stats import zscore

import matplotlib.pyplot as plt
from matplotlib import rcParams

from neurodsp.plts import plot_time_series

from bycycle.plts.utils import apply_tlims, get_extrema


def plot_cyclepoints(df, sig, fs, tlims=None, figsize=(15, 3), ax=None, xlabel='Time (s)', 
                     ylabel='Voltage (uV)', plot_sig=True, plot_extrema=True, plot_zerox=True):
    """Plot extrema and/or zerox.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe output of :func:`~.compute_features`.
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    tlims : tuple of (float, float), optional, default: None
        Start and stop times.
    figsize : tuple of (float, float), optional, default: (15, 3)
        Size of figure.
    ax : matplotlib axis, optional, default: None
        Axis to plot figure.
    xlabel : str, optional, default: 'Time (s)'
    ylabel : str, optional, default: 'Voltage (uV)
    plot_sig : bool, optional, default: True
        Plots the raw signal.
    plot_extrema :  bool, optional, default: True
        Plots peaks and troughs.
    plot_zerox :  bool, optional, default: True
        Plots zero-crossings.

    Returns
    -------
    ax : matplotlib axis
        Axis to customize plotting style.
    """
    
    rcParams['lines.markersize'] = 12

    # Set times and limits
    times = np.arange(0, len(sig) / fs, 1 / fs)
    tlims = (times[0], times[-1]) if tlims is None else tlims

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Determine extrema/zero-crossing times and signals
    center_e, side_e = get_extrema(df)
    
    df, sig, times = apply_tlims(df, sig, times, fs, tlims)
    
    # Extend plotting based on given arguments
    x_values = []
    y_values = []
    colors = []

    if plot_sig:

        x_values.extend([times])
        y_values.extend([sig])
        colors.extend(['k'])

    if plot_extrema:

        mask = np.append(df['sample_last_' + side_e].values,
                         df['sample_next_' + side_e].values[-1])
        
        x_values.extend([times[df['sample_' + center_e]], times[mask]])
        y_values.extend([sig[df['sample_' + center_e]], sig[mask]])
        colors.extend(['m.', 'c.'])
     
    if plot_zerox:

        x_values.extend([times[df['sample_zerox_decay']], times[df['sample_zerox_rise']]])
        y_values.extend([sig[df['sample_zerox_decay']], sig[df['sample_zerox_rise']]])
        colors.extend(['b.', 'g.'])

    # Plot cycle points
    plot_time_series(x_values, y_values, ax=ax, xlim=tlims, colors=colors,
                     xlabel=xlabel, ylabel=ylabel)

    return ax
