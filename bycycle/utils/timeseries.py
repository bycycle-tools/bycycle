"""Utility functions for working with time series."""

###################################################################################################
###################################################################################################

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

    if start is not None:
        sig = sig[times >= start]
        times = times[times >= start]

    if stop is not None:
        sig = sig[times < stop]
        times = times[times < stop]

    return sig, times
