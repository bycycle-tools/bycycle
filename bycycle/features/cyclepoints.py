"""Compute cyclepoint features for individual cycles."""

import pandas as pd
import numpy as np

from bycycle.utils.checks import check_param
from bycycle.cyclepoints import find_extrema, find_zerox

###################################################################################################
###################################################################################################

def compute_cyclepoints(sig, fs, f_range, **find_extrema_kwargs):
    """Compute sample indices for cyclepoints.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range, in Hz, to narrowband filter the signal, used to find zero-crossings.
    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for the function to find peaks and troughs (:func:`~.find_extrema`)
        that change filter parameters or boundary. By default, the boundary is set to zero.

    Returns
    -------
    df_samples : pandas.DataFrame, optional, default: False
        Dataframe containing sample indices of cyclepoints.
        Columns (listed for peak-centered cycles):

        - ``peaks`` : signal indices of oscillatory peaks
        - ``troughs`` :  signal indices of oscillatory troughs
        - ``rises`` : signal indices of oscillatory rising zero-crossings
        - ``decays`` : signal indices of oscillatory decaying zero-crossings
        - ``sample_peak`` : sample of 'sig' at which the peak occurs
        - ``sample_zerox_decay`` : sample of the decaying zero-crossing
        - ``sample_zerox_rise`` : sample of the rising zero-crossing
        - ``sample_last_trough`` : sample of the last trough
        - ``sample_next_trough`` : sample of the next trough

    Examples
    --------
    Compute the signal indices of cyclepoints:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_samples = compute_cyclepoints(sig, fs, f_range=(8, 12))
    """

    # Ensure arguments are within valid range
    check_param(fs, 'fs', (0, np.inf))

    # Find extrema and zero-crossings locations in the signal
    peaks, troughs = find_extrema(sig, fs, f_range, **find_extrema_kwargs)
    rises, decays = find_zerox(sig, peaks, troughs)

    # For each cycle, identify the sample of each extrema and zero-crossing
    samples = {}
    samples['sample_peak'] = peaks[1:]
    samples['sample_last_zerox_decay'] = decays[:-1]
    samples['sample_zerox_decay'] = decays[1:]
    samples['sample_zerox_rise'] = rises
    samples['sample_last_trough'] = troughs[:-1]
    samples['sample_next_trough'] = troughs[1:]

    df_samples = pd.DataFrame.from_dict(samples)

    return df_samples
