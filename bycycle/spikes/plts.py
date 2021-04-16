"""Spike plotting functions."""

import numpy as np

from neurodsp.plts import plot_time_series
from neurodsp.plts.utils import check_ax

from bycycle.utils.dataframes import get_extrema_df

###################################################################################################
###################################################################################################

def plot_spikes(df_features, sig, fs, spikes, index=None, ax=None):
    """Plot a group of spikes or the cyclepoints for an individual spike.

    Parameters
    ----------
    spikes : 2d array
        Isolated spikes returned from :func:`~.slice_spikes`.
    index : int, optional, default: None
        The index in ``spikes`` and ``df_features`` to plot. If none, plot all spikes.
    ax : matplotlib.Axes, optional, default: None
        Figure axes upon which to plot.
    """

    ax = check_ax(ax, (10, 4))

    center_e, _ = get_extrema_df(df_features)

    # Plot a single spike
    if index is not None:

        times = np.arange(0, len(sig)/fs, 1/fs)

        # Get where spike starts/ends
        start = df_features.iloc[index]['sample_start'].astype(int)
        end = df_features.iloc[index]['sample_end'].astype(int)

        sig_lim = sig[start:end+1]
        times_lim = times[start:end+1]

        # Plot the spike waveform
        plot_time_series(times_lim, sig_lim, ax=ax)

        # Plot cyclespoints
        labels, keys = _infer_labels(center_e)
        colors = ['C0', 'C1', 'C2', 'C3']

        for idx, key in enumerate(keys):

            sample = df_features.iloc[index][key].astype('int')

            plot_time_series(np.array([times[sample]]), np.array([sig[sample]]),
                             colors=colors[idx], labels=labels[idx], ls='', marker='o', ax=ax)

    # Plot all spikes
    else:

        times = np.arange(0, len(spikes[0])/fs, 1/fs)

        plot_time_series(times, spikes, ax=ax)


def _infer_labels(center_e):
    """Create labels based on center extrema."""

    # Infer labels
    if center_e == 'trough':
        labels = ['Trough', 'Peak', 'Inflection']
        keys = ['sample_trough', 'sample_next_peak', 'sample_end']
    elif center_e == 'peak':
        labels = ['Peak', 'Trough', 'Inflection']
        keys = ['sample_peak', 'sample_next_trough', 'sample_end']

    return labels, keys