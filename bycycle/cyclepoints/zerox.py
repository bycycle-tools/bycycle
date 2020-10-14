"""Find zero-crossings for individual cycles."""

from operator import gt, lt

import numpy as np

###################################################################################################
###################################################################################################

def find_zerox(sig, peaks, troughs):
    """Find zero-crossings within each cycle, from identified peaks and troughs.

    Parameters
    ----------
    sig : 1d array
        Time series.
    peaks : 1d array
        Samples of oscillatory peaks.
    troughs : 1d array
        Samples of oscillatory troughs.

    Returns
    -------
    rises : 1d array
        Samples at which oscillatory rising zero-crossings occur.
    decays : 1d array
        Samples at which oscillatory decaying zero-crossings occur.

    Notes
    -----
    - Zero-crossings are defined as when the voltage crosses midway between one extrema and
      the next - for example, a 'rise' is halfway from the trough to the peak.
    - If this halfway voltage is crossed at multiple times, the temporal median is taken
      as the zero-crossing.
    - Sometimes, due to noise in estimating peaks and troughs when the oscillation
      is absent, the estimated peak might be lower than an adjacent trough. If this
      occurs, the rise and decay zero-crossings will be set to be halfway between
      the peak and trough.
    - Burst detection should be used to restrict phase estimation to periods with oscillations
      present, in order to ignore periods of the signal in which estimation is poor.

    Examples
    --------
    Find the rise and decay zero-crossings locations of a simulated signal:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.cyclepoints import find_extrema
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> peaks, troughs = find_extrema(sig, fs, f_range=(8, 12))
    >>> rises, decays = find_zerox(sig, peaks, troughs)
    """

    # Calculate the number of rises and decays
    n_rises = len(peaks)
    n_decays = len(troughs)
    idx_bias = 0

    # Offset values, depending on order of peaks & troughs
    if peaks[0] < troughs[0]:
        n_rises -= 1
    else:
        n_decays -= 1
        idx_bias += 1

    rises = _find_flank_midpoints(sig, 'rise', n_rises, troughs, peaks, idx_bias)
    decays = _find_flank_midpoints(sig, 'decay', n_decays, peaks, troughs, idx_bias)

    return rises, decays


def find_flank_zerox(sig, flank):
    """Find zero-crossings on rising or decay flanks of a filtered signal.

    Parameters
    ----------
    sig : 1d array
        Time series to detect zero-crossings in.
    flank : {'rise', 'decay'}
        Which flank, rise or decay, to use to get zero crossings.

    Returns
    -------
    zero_xs : 1d array
        Samples of the zero crossings.

    Examples
    --------
    Find rising flanks in a filtered signal:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from neurodsp.filt import filter_signal
    >>> sig = sim_bursty_oscillation(10, 500, freq=10)
    >>> sig_filt = filter_signal(sig, 500, 'lowpass', 30)
    >>> rises_flank = find_flank_zerox(sig_filt, 'rise')
    """

    assert flank in ['rise', 'decay']
    pos = sig <= 0 if flank == 'rise' else sig > 0

    zero_xs = (pos[:-1] & ~pos[1:]).nonzero()[0]

    # If no zero-crossing's found (peak and trough are same voltage), output dummy value
    zero_xs = [int(len(sig) / 2)] if len(zero_xs) == 0 else zero_xs

    return zero_xs


def _find_flank_midpoints(sig, flank, n_flanks, extrema_start, extrema_end, idx_bias):
    """Helper function for find_zerox."""

    assert flank in ['rise', 'decay']
    idx_bias = -idx_bias + 1 if flank == 'rise' else idx_bias
    comp = gt if flank == 'rise' else lt

    flanks = np.zeros(n_flanks, dtype=int)
    for idx in range(n_flanks):

        sig_temp = np.copy(sig[extrema_start[idx]:extrema_end[idx + idx_bias] + 1])
        sig_temp -= (sig_temp[0] + sig_temp[-1]) / 2.

        # If data is all zeros, just set the zero-crossing to be halfway between
        if np.sum(np.abs(sig_temp)) == 0:
            flanks[idx] = extrema_start[idx] + int(len(sig_temp) / 2.)

        # If flank is actually an extrema, just set the zero-crossing to be halfway between
        elif comp(sig_temp[0], sig_temp[-1]):
            flanks[idx] = extrema_start[idx] + int(len(sig_temp) / 2.)

        else:
            flanks[idx] = extrema_start[idx] + int(np.median(find_flank_zerox(sig_temp, flank)))

    return flanks
