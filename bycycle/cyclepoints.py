"""Functions to finding cyclepoints for individual cycles."""

from operator import gt, lt

import numpy as np

from neurodsp.filt import filter_signal

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
    rise_xs = find_flank_zero_xs(sig_filt, 'rise')
    decay_xs = find_flank_zero_xs(sig_filt, 'decay')

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

    rises = _find_zerox_flanks(sig, 'rise', n_rises, troughs, peaks, idx_bias)
    decays = _find_zerox_flanks(sig, 'decay', n_decays, peaks, troughs, idx_bias)

    return rises, decays


def _find_zerox_flanks(sig, flank, n_flanks, extrema_start, extrema_end, idx_bias):
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
            flanks[idx] = extrema_start[idx] + int(np.median(find_flank_zero_xs(sig_temp, flank)))

    return flanks


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
    - Sometimes, due to noise, extrema and zero-crossing estimation is poor, and for example,
      the same index may be assigned to both a peak and a decaying zero-crossing.
      Because of this, we first assign phase values by zero-crossings, and then may overwrite
      them with extrema phases.
    - Use of burst detection will help avoid analyzing the oscillatory properties of
      non-oscillatory sections of the signal.
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

    # For the phase time series in which the trough is negative pi, replace the decaying
    #   periods with these periods in the phase time series in which the trough is pi
    diffs = np.diff(pha_tnpi)
    diffs = np.append(diffs, 99)
    pha_tnpi[diffs < 0] = pha_tpi[diffs < 0]

    # Assign the periods before the first empirical phase timepoint to NaN
    diffs = np.diff(pha_tnpi)
    first_empirical_idx = next(idx for idx, xi in enumerate(diffs) if xi > 0)
    pha_tnpi[:first_empirical_idx] = np.nan

    # Assign the periods after the last empirical phase timepoint to NaN
    diffs = np.diff(pha_tnpi)
    last_empirical_idx = next(idx for idx, xi in enumerate(diffs[::-1]) if xi > 0)
    pha_tnpi[-last_empirical_idx + 1:] = np.nan

    return pha_tnpi


def find_flank_zero_xs(sig, flank):
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
