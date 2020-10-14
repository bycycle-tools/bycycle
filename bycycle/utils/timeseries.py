"""Utility functions for working with time series."""

import numpy as np

from bycycle.utils.checks import check_param

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
        The lower time limit, in seconds, to restrict the signal.
    stop : float
        The upper time limit, in seconds, to restrict the signal.

    Returns
    -------
    sig : 1d array
        A limited time series.
    times : 1d array
        A limited time definition.

    Examples
    --------
    Restrict a signal and times to the first second:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from neurodsp.utils import create_times
    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=10)
    >>> times = create_times(n_seconds=10, fs=500)
    >>> sig, times = limit_signal(times, sig, start=0, stop=1)

    """

    # Ensure arguments are within valid range
    check_param(start, 'start', (0, stop))
    check_param(stop, 'stop', (start, np.inf))

    if start is not None:
        sig = sig[times >= start]
        times = times[times >= start]

    if stop is not None:
        sig = sig[times < stop]
        times = times[times < stop]

    return sig, times
