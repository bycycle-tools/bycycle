"""Dataframe and spike segmentation functions."""

import numpy as np
import pandas as pd

from bycycle.cyclepoints import find_zerox
from bycycle.utils.dataframes import get_extrema_df

###################################################################################################
###################################################################################################


def create_cyclepoints_df(sig, starts, decays, troughs, rises, peaks, next_decays, ends):
    """Create a dataframe from arrays.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    start : 1d array
        Spike start location.
    decay : 1d array
        Spike zero-crossing decay location.
    trough : 1d array
        Spike minima location.
    rise : 1d array
        Spike zero-crossing rise location.
    next_peak : 1d array
        Spike maxima location.
    next_decay : 1d array
        Spike zero-crossing decay location, after the peak.
    end : 1d array
        Spike end location.

    Returns
    -------
    df_samples : pd.DataFrame
        Cyclepoint locaions, in samples.
    """
    df_samples = pd.DataFrame()

    df_samples['sample_start'] = starts
    df_samples['sample_decay'] = decays
    df_samples['sample_trough'] = troughs
    df_samples['sample_rise'] = rises
    df_samples['sample_next_peak'] = peaks
    df_samples['sample_next_decay'] = next_decays
    df_samples['sample_end'] = ends

    return df_samples


def split_signal(df_samples, sig):
    """Split a signal into segmented spikes.

    Parameters
    ----------
    df_samples : pd.DataFrame
        Cyclepoint locaions, in samples. Returned from func:`~.create_cyclepoints_df`.
    sig : 1d array
        Voltage time series.

    Returns
    -------
    spikes : 1d array
        Segmented spikes.

    Notes
    -----
    Each spike is trough-centered in the 2d array. Padding (np.nan's) may exists at the beginning
    or end of each spike.
    """

    starts = df_samples['sample_start'].values
    troughs = df_samples['sample_trough'].values
    ends = df_samples['sample_end'].values

    max_left = np.max(troughs - starts)
    max_right = np.max(ends - troughs)

    spikes = np.zeros((len(df_samples), max_left+max_right))
    spikes[:, :] = np.nan

    for idx, (start, trough, end) in enumerate(zip(starts, troughs, ends)):

        pad_left = max_left - (trough - start)
        pad_right = max_right - (end - trough)

        if pad_right != 0:
            spikes[idx][pad_left:-pad_right] = sig[start:end]
        else:
            spikes[idx][pad_left:] = sig[start:end]

    return spikes


def rename_df(df_features):
    """Rename the columns of a peak-centered dataframe.

    Parameters
    ----------
    df_features : pd.DataFrame
        A dataframe containing cyclepoint locations, as samples indices, for each spike.

    Returns
    -------
    df_features : pd.DataFrame
        A renamed dataframe containing updated column names.
    """

    mapping = {}

    orig_keys = ['peak', 'trough', 'rise', 'decay']
    new_keys = ['trough', 'peak', 'decay', 'rise']

    for key in df_features.columns:
        for orig, new in zip(orig_keys, new_keys):
            if orig in key:
                mapping[key] = key.replace(orig, new)

    df_features.rename(columns=mapping, inplace=True)

    return df_features
