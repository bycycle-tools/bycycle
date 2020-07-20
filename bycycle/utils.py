"""General utility functions."""

###################################################################################################
###################################################################################################

def limit_df(df, fs, start=None, stop=None):
    """Restrict dataframe to be within time limits.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe output of :func:`~.compute_features`.
    fs : float
        Sampling rate, in Hz.
    start : float, optional
        The lower time limit, in seconds, to restrict the df.
    stop : float, optional
        The upper time limit, in seconds, to restrict the df.

    Returns
    -------
    df : pandas.DataFrame
        A limited dataframe of cycle features.

    Notes
    -----
    Cycles, or rows in the `df`, are included if any segment of the cycle falls after the
    `stop` time or before the `end` time.
    """

    center_e, side_e = get_extrema(df)

    start = 0 if start is None else start

    df = df[df['sample_next_' + side_e].values >= start*fs]

    if stop is not None:
        df = df[df['sample_last_' + side_e].values < stop*fs]

    # Shift sample indices to start at 0
    df['sample_last_' + side_e] = df['sample_last_' + side_e] - int(fs * start)
    df['sample_next_' + side_e] = df['sample_next_' + side_e] - int(fs * start)
    df['sample_' + center_e] = df['sample_' + center_e] - int(fs * start)
    df['sample_zerox_rise'] = df['sample_zerox_rise'] - int(fs * start)
    df['sample_zerox_decay'] = df['sample_zerox_decay'] - int(fs * start)

    return df


def limit_signal(times, sig, start=None, stop=None):
    """Restrict signal and times to be within time limits.

    Parameters
    ----------
    times : 1d array
        Time definition for the time series.
    sig : 1d array
        Time series.
    start : float
        The lower time limit, in seconds, to restrict the df.
    stop : float
        The upper time limit, in seconds, to restrict the df.

    Returns
    -------
    sig : 1d array
        A limited time series.
    times : 1d array
        A limited time definition.
    """
    # Limit times and sig to start and stop times
    if start is not None:
        sig = sig[times >= start]
        times = times[times >= start]

    if stop is not None:
        sig = sig[times < stop]
        times = times[times < stop]

    return sig, times


def get_extrema(df):
    """Determine whether cycles are peak or trough centered.

    Parameters
    ----------
    df : pandas.DataFrame
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
