"""Find extrema for individual cycles."""

import numpy as np
from scipy.signal import resample

from neurodsp.filt import filter_signal
from neurodsp.filt.fir import compute_filter_length

from bycycle.utils.checks import check_param_range
from bycycle.cyclepoints.zerox import find_flank_zerox

###################################################################################################
###################################################################################################

def find_extrema(sig, fs, f_range, boundary=0, first_extrema='peak',
                 filter_kwargs=None, pass_type='bandpass', pad=True, optimize=None):
    """Identify peaks and troughs in a time series.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range, in Hz, to narrowband filter the signal, used to find zero-crossings.
    boundary : int, optional, default: 0
        Number of samples from edge of the signal to ignore.
    first_extrema: {'peak', 'trough', None}
        If 'peak', then force the output to begin with a peak and end in a trough.
        If 'trough', then force the output to begin with a trough and end in peak.
        If None, force nothing.
    filter_kwargs : dict, optional, default: None
        Keyword arguments to :func:`~neurodsp.filt.filter.filter_signal`,
        such as 'n_cycles' or 'n_seconds' to control filter length.
    pass_type : str, optional, default: 'bandpass'
        Which kind of filter pass_type is consistent with the frequency definition provided.
    pad : bool, optional, default: True
        Whether to pad ``sig`` with zeros to prevent missed cyclepoints at the edges.
    optimize : tuple of (int, float), optional, default: None
        An integer defining a set sample spacing between correlated cycles and a float that defines
        the correlation threshold to sub-select cycles.

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

    Examples
    --------
    Find the locations of peaks and burst in a signal:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> peaks, troughs = find_extrema(sig, fs, f_range=(8, 12))
    """

    # Ensure arguments are within valid range
    check_param_range(fs, 'fs', (0, np.inf))

    # Set default filtering parameters
    if filter_kwargs is None:
        filter_kwargs = {}

    # Get the original signal and filter lengths
    sig_len = len(sig)
    filt_len = 0

    # Pad beginning of signal with zeros to prevent missing cyclepoints
    if pad:

        filt_len = compute_filter_length(fs, pass_type, f_range[0], f_range[1],
                                         n_seconds=filter_kwargs.get('n_seconds', None),
                                         n_cycles=filter_kwargs.get('n_cycles', 3))

        # Pad the signal
        sig_pad = np.pad(sig, int(np.ceil(filt_len/2)), mode='constant')

    # Narrowband filter signal
    sig_filt = filter_signal(sig_pad, fs, pass_type, f_range, remove_edges=False, **filter_kwargs)

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
    _decay_xs = decay_xs.copy()
    for p_idx in range(n_peaks):

        # Calculate the sample range between the most recent zero rise and the next zero decay
        last_rise = rise_xs[p_idx]

        for idx, decay in enumerate(_decay_xs):
            if decay > last_rise:
                _decay_xs = _decay_xs[idx:]
                break

        next_decay = _decay_xs[0]

        # Identify time of peak
        peaks[p_idx] = np.argmax(sig_pad[last_rise:next_decay]) + last_rise

    # Calculate trough samples
    troughs = np.zeros(n_troughs, dtype=int)
    _rise_xs = rise_xs.copy()
    for t_idx in range(n_troughs):

        # Calculate the sample range between the most recent zero decay and the next zero rise
        last_decay = decay_xs[t_idx]

        for idx, rise in enumerate(_rise_xs):
            if rise > last_decay:
                _rise_xs = _rise_xs[idx:]
                break

        next_rise = _rise_xs[0]

        # Identify time of trough
        troughs[t_idx] = np.argmin(sig_pad[last_decay:next_rise]) + last_decay

    # Remove padding
    peaks = peaks - int(np.ceil(filt_len/2))
    troughs = troughs - int(np.ceil(filt_len/2))

    # Remove peaks and trough outside the boundary limit
    peaks = peaks[np.logical_and(peaks > boundary, peaks < sig_len - boundary)]
    troughs = troughs[np.logical_and(troughs > boundary, troughs < sig_len - boundary)]

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

    if optimize is not None:
        troughs = optimize_troughs(sig, peaks, troughs, optimize[0], optimize[1])

    return peaks, troughs



def optimize_troughs(sig, peaks, troughs, cyc_len, corr_thresh=.75):
    """Optimized troughs based on a pre-determined cycle length.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    cyc_len : int
        Length to fix each cycle to.
    corr_thresh : float, optional, default: 0.75
        Correlation threshold to refine cycle definitions.

    Returns
    -------
    troughs_fixed : 1d array
        Sample of throughs, fixed at the pre-determined length.
    """

    # Resample to target cycle length for comparison
    sig_resample = np.zeros((len(troughs)-1, cyc_len))

    for tind, trough in enumerate(troughs[:-1]):
        _sig = sig[trough:troughs[tind+1]].copy()
        _sig -= _sig.mean()
        sig_resample[tind] = resample(_sig, cyc_len)

    sig_avg = sig_resample.mean(axis=0)

    # Find correlated cycles
    mask = np.zeros(len(troughs), dtype=bool)

    for ind, _sig in enumerate(sig_resample):
        if np.corrcoef(sig_avg, _sig)[0][1] > corr_thresh:
            mask[ind] = True

    starts = np.where(mask)[0]
    inds = np.unique(np.append(starts, starts+1))

    # Evenly space troughs by cyc_len
    troughs_fixed = troughs[starts].copy()

    for ind, start in enumerate(starts):
        if ind < len(starts) - 1 and starts[ind+1] - start == 1:
            troughs_fixed[ind+1] = troughs_fixed[ind] + cyc_len

    ends = np.where(np.diff(starts) != 1)[0]

    troughs_fixed = np.insert(troughs_fixed, ends+1, troughs_fixed[ends] + cyc_len)
    troughs_fixed = np.insert(troughs_fixed, len(troughs_fixed), troughs_fixed[-1] + cyc_len)

    # Split continuous bursts
    split_inds = np.split(np.arange(len(troughs_fixed)),
                          np.where(np.diff(troughs_fixed) != cyc_len)[0]+1)

    # Find optimal x-axis translation, per burst
    half_cyc = cyc_len // 2
    shifts = np.arange(-half_cyc, half_cyc)

    for inds in split_inds:
        tsum = np.zeros(len(shifts))
        for sind, shift in enumerate(shifts):
            _troughs = troughs_fixed[inds].copy()
            _troughs += shift
            tsum[sind] = np.sum(sig[_troughs])

        troughs_fixed[inds] += shifts[np.argmin(tsum)]

    # Add peaks and non-updated troughs
    t_inserts = []

    for ind in range(len(peaks)-1):

        current_peak = peaks[ind]
        next_peak = peaks[ind+1]

        _trough = np.where((troughs_fixed > current_peak) & (troughs_fixed < next_peak))[0]

        if len(_trough) == 1:
            troughs[ind] = troughs_fixed[_trough[0]]

    return troughs
