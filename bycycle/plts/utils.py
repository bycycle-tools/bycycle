"""Plotting utility functions."""

import numpy as np

###################################################################################################
###################################################################################################

def apply_tlims(df, sig, times, fs, tlims):
    """Limit dataframe to be within tlims.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe output of :func:`~.compute_features`.
    sig : 1d array
        Time series to plot.
    times : 1d array
        Time definition for the time series to be plotted.
    fs : float
        Sampling rate, in Hz.
    tlims : tuple of (float, float), optional, default: None
        Start and stop times.

    Returns
    -------
    df : pandas DataFrame
        A limited dataframe of cycle features.
    sig : 1d array
        A limited time series.
    times : 1d
        A limited time definintion.
    """

    center_e, side_e = get_extrema(df)

    # Limit dataframe to tlims and round to nearest +/- 1 cycle.
    df = df[(df['sample_last_' + side_e].values >= tlims[0]*fs) &
            (df['sample_next_' + side_e].values <= tlims[1]*fs)]

    # Realign with sig and times
    df['sample_last_' + side_e] = df['sample_last_' + side_e] - int(fs * tlims[0])
    df['sample_next_' + side_e] = df['sample_next_' + side_e] - int(fs * tlims[0])
    df['sample_' + center_e] = df['sample_' + center_e] - int(fs * tlims[0])

    # Limit times and sig to tlim
    tidx = np.logical_and(times >= tlims[0], times < tlims[1])
    sig = sig[tidx]
    times = times[tidx]

    return df, sig, times


def get_extrema(df):
    """Determine whether cycles are peaks or troughs centered.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe output of :func:`~.compute_features`.

    Returns
    -------
    center_e : str
        Center extrema, either 'peak' or 'trough'
    side_e : str
        Side extrema, either 'peak' or 'trough'
    """

    center_e = 'peak' if 'sample_peak' in df.columns else 'trough'
    side_e = 'trough' if center_e == 'peak' else 'peak'

    return center_e, side_e
