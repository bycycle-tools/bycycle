"""Interpolated phase, using cyclepoints."""

import numpy as np

###################################################################################################
###################################################################################################

def extrema_interpolated_phase(sig, peaks, troughs, rises=None, decays=None):
    """Use extrema and (optionally) zero-crossings to estimate instantaneous phase.

    Parameters
    ----------
    sig : 1d array
        Time series.
    peaks : 1d array
        Samples of oscillatory peaks.
    troughs : 1d array
        Samples of oscillatory troughs.
    rises : 1d array, optional
        Samples of oscillatory rising zero-crossings.
    decays : 1d array, optional
        Samples of oscillatory decaying zero-crossings.

    Returns
    -------
    pha : 1d array
        Instantaneous phase time series.

    Notes
    -----
    - Phase is encoded as:
        - phase 0 for peaks
        - phase pi/2 for decay zero-crossing
        - phase pi/-pi for troughs
        - phase -pi/2 for rise zero-crossing
    - Extrema and zero-crossing estimation can be poor if, for example, the signal is noisy.
      In such cases, the same index may be assigned to both a peak and a decaying zero-crossing.
      To address this, we first assign phase values by zero-crossings, and then may overwrite
      them with extrema phases.
    - Using burst detection helps avoid analyzing oscillatory properties of
      non-oscillatory sections of the signal.

    Examples
    --------
    Estimate phase from peaks and troughs:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.cyclepoints import find_extrema
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> peaks, troughs = find_extrema(sig, fs, f_range=(8, 12))
    >>> pha = extrema_interpolated_phase(sig, peaks, troughs)
    """

    # Initialize phase arrays, one for trough pi and trough -pi
    sig_len = len(sig)
    times = np.arange(sig_len)
    pha_tpi = np.zeros(sig_len) * np.nan
    pha_tnpi = np.zeros(sig_len) * np.nan

    # If specified, assign phases to zero-crossings
    if rises is not None:
        pha_tpi[rises] = -np.pi / 2
        pha_tnpi[rises] = -np.pi / 2
    if decays is not None:
        pha_tpi[decays] = np.pi / 2
        pha_tnpi[decays] = np.pi / 2

    # Define phases
    pha_tpi[peaks] = 0
    pha_tpi[troughs] = np.pi
    pha_tnpi[peaks] = 0
    pha_tnpi[troughs] = -np.pi

    # Interpolate to find all phases
    pha_tpi = np.interp(times, times[~np.isnan(pha_tpi)], pha_tpi[~np.isnan(pha_tpi)])
    pha_tnpi = np.interp(times, times[~np.isnan(pha_tnpi)], pha_tnpi[~np.isnan(pha_tnpi)])

    pha = _merge_phases(pha_tpi, pha_tnpi)

    return pha


def _merge_phases(pha_tpi, pha_tnpi):
    """Helper functions for extrema_interpolated_phase."""

    # Create phase differences to determine where phase is decaying/rising in the -pi array
    diffs = np.diff(pha_tnpi)

    # Pad the phase difference array with a NaN to maintain a length equal to the timeseries
    diffs = np.append(diffs, np.nan)

    # Create new phase series, using trough pi for decaying periods & trough -pi for rising periods
    pha = np.array([pha_tpi[idx] if diffs[idx] < 0 else pha for idx, pha in enumerate(pha_tnpi)])

    # Assign the periods before the first empirical phase timepoint to NaN
    diffs = np.diff(pha)
    first_empirical_idx = next(idx for idx, xi in enumerate(diffs) if xi > 0)
    pha[:first_empirical_idx] = np.nan

    # Assign the periods after the last empirical phase timepoint to NaN
    diffs = np.diff(pha)
    last_empirical_idx = next(idx for idx, xi in enumerate(diffs[::-1]) if xi > 0)
    pha[-last_empirical_idx + 1:] = np.nan

    return pha
