"""Find cyclepoints for individual cycles."""

from operator import gt, lt

import numpy as np

from neurodsp.filt import filter_signal

from bycycle.utils.checks import check_param

###################################################################################################
###################################################################################################

def find_extrema(sig, fs, f_range, boundary=None, first_extrema='peak', filter_kwargs=None):
    """Identify peaks and troughs in a time series.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range, in Hz, to narrowband filter the signal, used to find zero-crossings.
    boundary : int, optional
        Number of samples from edge of the signal to ignore.
    first_extrema: {'peak', 'trough', None}
        If 'peak', then force the output to begin with a peak and end in a trough.
        If 'trough', then force the output to begin with a trough and end in peak.
        If None, force nothing.
    filter_kwargs : dict, optional
        Keyword arguments to :func:`~neurodsp.filt.filter.filter_signal`,
        such as 'n_cycles' or 'n_seconds' to control filter length.

    Returns
    -------
    peaks : 1d array
        Indices at which oscillatory peaks occur in the input ``sig``.
    troughs : 1d array
        Indices at which oscillatory troughs occur in the input ``sig``.

    Notes
    -----
    This function assures that there are the same number of peaks and troughs
    if the first extrema is forced to be either peak or trough.
    """

    # Ensure arguments are within valid range
    check_param(fs, 'fs', (0, np.inf))

    # Set default filtering parameters
    if filter_kwargs is None:
        filter_kwargs = {}

    # Default boundary value as 1 cycle length of low cutoff frequency
    if boundary is None:
        boundary = int(np.ceil(fs / float(f_range[0])))

    # Narrowband filter signal
    sig_filt = filter_signal(sig, fs, 'bandpass', f_range,
                             remove_edges=False, **filter_kwargs)

    # Find rising and decaying zero-crossings (narrowband)
    rise_xs = find_flank_zerox(sig_filt, 'rise')
    decay_xs = find_flank_zerox(sig_filt, 'decay')

    # Compute number of peaks and troughs
    if rise_xs[-1] > decay_xs[-1]:
        n_peaks = len(rise_xs) - 1
        n_troughs = len(decay_xs)
    else:
        n_peaks = len(rise_xs)
        n_troughs = len(decay_xs) - 1

    # Calculate peak samples
    peaks = np.zeros(n_peaks, dtype=int)
    for p_idx in range(n_peaks):

        # Calculate the sample range between the most recent zero rise and the next zero decay
        last_rise = rise_xs[p_idx]
        next_decay = decay_xs[decay_xs > last_rise][0]
        # Identify time of peak
        peaks[p_idx] = np.argmax(sig[last_rise:next_decay]) + last_rise

    # Calculate trough samples
    troughs = np.zeros(n_troughs, dtype=int)
    for t_idx in range(n_troughs):

        # Calculate the sample range between the most recent zero decay and the next zero rise
        last_decay = decay_xs[t_idx]
        next_rise = rise_xs[rise_xs > last_decay][0]
        # Identify time of trough
        troughs[t_idx] = np.argmin(sig[last_decay:next_rise]) + last_decay

    # Remove peaks and troughs within the boundary limit
    peaks = peaks[np.logical_and(peaks > boundary, peaks < len(sig) - boundary)]
    troughs = troughs[np.logical_and(troughs > boundary, troughs < len(sig) - boundary)]

    # Force the first extrema to be as desired & assure equal # of peaks and troughs
    if first_extrema == 'peak':
        troughs = troughs[1:] if peaks[0] > troughs[0] else troughs
        peaks = peaks[:-1] if peaks[-1] > troughs[-1] else peaks
    elif first_extrema == 'trough':
        peaks = peaks[1:] if troughs[0] > peaks[0] else peaks
        troughs = troughs[:-1] if troughs[-1] > peaks[-1] else troughs
    elif first_extrema is None:
        pass
    else:
        raise ValueError('Parameter "first_extrema" is invalid')

    return peaks, troughs


def find_zerox(sig, peaks, troughs):
    """Find zero-crossings within each cycle, from identified peaks and troughs.

    A rising zero-crossing occurs when the voltage crosses midway between the trough
    voltage and subsequent peak voltage. A decay zero-crossing is defined similarly.
    If this voltage is crossed at multiple times, the temporal median is taken
    as the zero-crossing.

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
    - Sometimes, due to noise in estimating peaks and troughs when the oscillation
      is absent, the estimated peak might be lower than an adjacent trough. If this
      occurs, the rise and decay zero-crossings will be set to be halfway between
      the peak and trough. Burst detection should be used in order to ignore these
      periods of the signal.
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
        Locations of the zero crossings.
    """

    assert flank in ['rise', 'decay']
    pos = sig < 0 if flank == 'rise' else sig > 0

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
