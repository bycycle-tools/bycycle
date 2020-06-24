"""Plotting utility functions."""

import numpy as np

###################################################################################################
###################################################################################################


def limit_df(df, fs, xlim):
    """Restrict dataframe to be within time limits.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe output of :func:`~.compute_features`.
    fs : float
        Sampling rate, in Hz.
    xlim : tuple of (float, float), optional, default: None
        Start and stop times.

    Returns
    -------
    df : pandas DataFrame
        A limited dataframe of cycle features.
    """

    center_e, side_e = get_extrema(df)

    # Limit dataframe to xlim
    df = df[(df['sample_next_' + side_e].values >= xlim[0]*fs) &
            (df['sample_last_' + side_e].values < xlim[1]*fs)]

    # Shift sample indices to start at 0
    df['sample_last_' + side_e] = df['sample_last_' + side_e] - int(fs * xlim[0])
    df['sample_next_' + side_e] = df['sample_next_' + side_e] - int(fs * xlim[0])
    df['sample_' + center_e] = df['sample_' + center_e] - int(fs * xlim[0])
    df['sample_zerox_rise'] = df['sample_zerox_rise'] - int(fs * xlim[0])
    df['sample_zerox_decay'] = df['sample_zerox_decay'] - int(fs * xlim[0])

    return df


def limit_sig_times(sig, times, xlim):
    """Restrict signal and times to be within time limits.

    Parameters
    ----------
    sig : 1d array
        Time series to plot.
    times : 1d array
        Time definition for the time series.
    xlim : tuple of (float, float)
        Start and stop times.

    Returns
    -------
    sig : 1d array
        A limited time series.
    times : 1d
        A limited time definintion.
    """

    # Limit times and sig to tlim
    tidx = np.logical_and(times >= xlim[0], times < xlim[1])
    sig = sig[tidx]
    times = times[tidx]

    return sig, times


def get_extrema(df):
    """Determine whether cycles are peaks or troughs centered.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe output of :func:`~.compute_features`.

    Returns
    -------
    center_e : str
        Center extrema, either 'peak' or 'trough'.
    side_e : str
        Side extrema, either 'peak' or 'trough'.
    """

    center_e = 'peak' if 'sample_peak' in df.columns else 'trough'
    side_e = 'trough' if center_e == 'peak' else 'peak'

    return center_e, side_e
