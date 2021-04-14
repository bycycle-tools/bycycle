"""Spike plotting functions."""

import numpy as np

from neurodsp.plts import plot_time_series
from neurodsp.plts.utils import check_ax

from bycycle.utils.dataframes import get_extrema_df

###################################################################################################
###################################################################################################

def plot_spike(fs, spikes, df_features, index=None, ax=None):
    """Plot a group of spikes or the cyclepoints for an individual spike.

    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.
    spikes : 2d array
        Isolated spikes returned from :func:`~.slice_spikes`.
    df_features : pd.DataFrame
        A dataframe containing cyclepoint locations, as samples indices, for each spike.
        Returned from :func:`~.slice_spikes`
    index : int, optional, default: None
        The index in ``spikes`` and ``df_features`` to plot. If none, plot all spikes.
    ax : matplotlib.Axes, optional, default: None
        Figure axes upon which to plot.
    """
    ax = check_ax(ax, (10, 4))

    center_e, _ = get_extrema_df(df_features)

    # Plot a single spike
    if index is not None:

        # Plot the spike waveform
        times = np.arange(0, len(spikes[index])/fs, 1/fs)
        spike = spikes[index]
        plot_time_series(times, spike, ax=ax)

        # Infer labels
        if center_e == 'trough':
            labels = ['Last Rise', 'Trough', 'Peak', 'Inflection']
            keys = ['sample_last_rise', 'sample_trough', 'sample_next_peak', 'sample_next_decay']
        elif center_e == 'peak':
            labels = ['Last Decay', 'Peak', 'Trough', 'Inflection']
            keys = ['sample_last_decay', 'sample_peak', 'sample_next_trough', 'sample_next_rise']

        colors = ['C0', 'C1', 'C2', 'C3', 'C5', 'C6', 'C7']

        # Plot cyclespoints
        for idx, key in enumerate(keys):

            sample = df_features.iloc[index][key].astype('int')

            plot_time_series(np.array([times[sample]]), np.array([spike[sample]]),
                             colors=colors[idx], labels=labels[idx], ls='', marker='o', ax=ax)

    # Plot all spikes
    else:

        times = np.arange(0, len(spikes[0])/fs, 1/fs)

        plot_time_series(times, spikes, ax=ax)
