"""Functions to determine the locations of peaks, troughs, and zero-crossings (rise and decay)
for individual cycles.
"""

import numpy as np

from neurodsp.filt import filter_signal

###################################################################################################
###################################################################################################

def find_extrema(sig, fs, f_range, boundary=None, first_extrema='peak', filter_kwargs=None):
    """Identify peaks and troughs in a time series.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range, in Hz, for narrowband signal of interest,
        used to find zero-crossings of the oscillation.
    boundary : int, optional
        Number of samples from edge of recording to ignore.
    first_extrema: {'peak', 'trough', None}
        If 'peak', then force the output to begin with a peak and end in a trough.
        If 'trough', then force the output to begin with a trough and end in peak.
        If None, force nothing.
    filter_kwargs : dict, optional
        Keyword arguments to :func:`~neurodsp.filt.filter.filter_signal`, such as 'n_cycles' or
        'n_seconds' to control filter length.

    Returns
    -------
    ps : 1d array
        Indices at which oscillatory peaks occur in the input ``sig``.
    ts : 1d array
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
    sig_filt = filter_signal(sig, fs, 'bandpass', f_range, remove_edges=False, **filter_kwargs)

    # Find rising and falling zero-crossings (narrowband)
    zerorise_n = _fzerorise(sig_filt)
    zerofall_n = _fzerofall(sig_filt)

    # Compute number of peaks and troughs
    if zerorise_n[-1] > zerofall_n[-1]:
        pl = len(zerorise_n) - 1
        tl = len(zerofall_n)
    else:
        pl = len(zerorise_n)
        tl = len(zerofall_n) - 1

    # Calculate peak samples
    ps = np.zeros(pl, dtype=int)
    for p_idx in range(pl):

        # Calculate the sample range between the most recent zero rise and the next zero fall
        mrzerorise = zerorise_n[p_idx]
        nfzerofall = zerofall_n[zerofall_n > mrzerorise][0]
        # Identify time of peak
        ps[p_idx] = np.argmax(sig[mrzerorise:nfzerofall]) + mrzerorise

    # Calculate trough samples
    ts = np.zeros(tl, dtype=int)
    for t_idx in range(tl):

        # Calculate the sample range between the most recent zero fall and the next zero rise
        mrzerofall = zerofall_n[t_idx]
        nfzerorise = zerorise_n[zerorise_n > mrzerofall][0]
        # Identify time of trough
        ts[t_idx] = np.argmin(sig[mrzerofall:nfzerorise]) + mrzerofall

    # Remove peaks and troughs within the boundary limit
    ps = ps[np.logical_and(ps > boundary, ps < len(sig) - boundary)]
    ts = ts[np.logical_and(ts > boundary, ts < len(sig) - boundary)]

    # Force the first extrema to be as desired
    # Assure equal # of peaks and troughs
    if first_extrema == 'peak':
        if ps[0] > ts[0]:
            ts = ts[1:]
        if ps[-1] > ts[-1]:
            ps = ps[:-1]
    elif first_extrema == 'trough':
        if ts[0] > ps[0]:
            ps = ps[1:]
        if ts[-1] > ps[-1]:
            ts = ts[:-1]
    elif first_extrema is None:
        pass
    else:
        raise ValueError('Parameter "first_extrema" is invalid')

    return ps, ts


def _fzerofall(sig):
    """Find zero-crossings on falling edge of a filtered signal."""

    pos = sig > 0
    zerofalls = (pos[:-1] & ~pos[1:]).nonzero()[0]

    # In the rare case where no zero-crossing is found (peak and trough are same voltage),
    #   output dummy value.
    if len(zerofalls) == 0:
        zerofalls = [int(len(sig) / 2)]

    return zerofalls


def _fzerorise(sig):
    """Find zero-crossings on rising edge of a filtered signal."""

    pos = sig < 0
    zerorises = (pos[:-1] & ~pos[1:]).nonzero()[0]

    # In the rare case where no zero-crossing is found (peak and trough are same voltage),
    #   output dummy value.
    if len(zerorises) == 0:
        zerorises = [int(len(sig) / 2)]

    return zerorises


def find_zerox(sig, ps, ts):
    """Find zero-crossings within each cycle after peaks and troughs are identified.

    A rising zero=crossing occurs when the voltage crosses midway between the trough
    voltage and subsequent peak voltage. A decay zero-crossing is defined similarly.
    If this voltage is crossed at multiple times, the temporal median is taken
    as the zero-crossing.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    ps : 1d array
        Samples of oscillatory peaks.
    ts : 1d array
        Samples of oscillatory troughs.

    Returns
    -------
    zerox_rise : 1d array
        Samples at which oscillatory rising zero-crossings occur.
    zerox_decay : 1d array
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
    if ps[0] < ts[0]:
        n_rises = len(ps) - 1
        n_decays = len(ts)
        idx_bias = 0
    else:
        n_rises = len(ps)
        n_decays = len(ts) - 1
        idx_bias = 1

    # Find zero-crossings for rise
    zerox_rise = np.zeros(n_rises, dtype=int)
    for idx_rise in range(n_rises):
        sig_temp = np.copy(sig[ts[idx_rise]:ps[idx_rise + 1 - idx_bias] + 1])
        sig_temp -= (sig_temp[0] + sig_temp[-1]) / 2.

        # If data is all zeros, just set the zero-crossing to be halfway between
        if np.sum(np.abs(sig_temp)) == 0:
            zerox_rise[idx_rise] = ts[idx_rise] + int(len(sig_temp) / 2.)

        # If rise is actually decay, just set the zero-crossing to be halfway between
        elif sig_temp[0] > sig_temp[-1]:
            zerox_rise[idx_rise] = ts[idx_rise] + int(len(sig_temp) / 2.)

        else:
            zerox_rise[idx_rise] = ts[idx_rise] + int(np.median(_fzerorise(sig_temp)))

    # Find zero-crossings for decays
    zerox_decay = np.zeros(n_decays, dtype=int)
    for idx_decay in range(n_decays):
        sig_temp = np.copy(sig[ps[idx_decay]:ts[idx_decay + idx_bias] + 1])
        sig_temp -= (sig_temp[0] + sig_temp[-1]) / 2.

        # If data is all zeros, just set the zero-crossing to be halfway between
        if np.sum(np.abs(sig_temp)) == 0:
            zerox_decay[idx_decay] = ps[idx_decay] + int(len(sig_temp) / 2.)

        # If decay is actually rise, just set the zero-crossing to be halfway between
        elif sig_temp[0] < sig_temp[-1]:
            zerox_decay[idx_decay] = ps[idx_decay] + int(len(sig_temp) / 2.)
        else:
            zerox_decay[idx_decay] = ps[idx_decay] + int(np.median(_fzerofall(sig_temp)))

    return zerox_rise, zerox_decay


def extrema_interpolated_phase(sig, ps, ts, zerox_rise=None, zerox_decay=None):
    """Use peaks (phase 0) and troughs (phase pi/-pi) to estimate instantaneous phase.
    Also use rise and decay zero-crossings (phase -pi/2 and pi/2, respectively) if provided.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    ps : 1d array
        Samples of oscillatory peaks.
    ts : 1d array
        Samples of oscillatory troughs.
    zerox_rise : 1d array, optional
        Samples of oscillatory rising zero-crossings.
    zerox_decay : 1d array, optional
        Samples of oscillatory decaying zero-crossings.

    Returns
    -------
    pha : 1d array
        Instantaneous phase time series.

    Notes
    -----
    Sometimes, due to noise, extrema and zero-crossing estimation is poor, and for example,
    the same index may be assigned to both a peak and a decaying zero-crossing.
    Because of this, we first assign phase values by zero-crossings, and then may overwrite
    them with extrema phases.
    Use of burst detection will help avoid analyzing the oscillatory properties of
    non-oscillatory sections of the signal.
    """

    # Initialize phase arrays, one for trough pi and trough -pi
    sig_len = len(sig)
    times = np.arange(sig_len)
    pha_tpi = np.zeros(sig_len) * np.nan
    pha_tnpi = np.zeros(sig_len) * np.nan

    # If specified, assign phases to zero-crossings
    if zerox_rise is not None:
        pha_tpi[zerox_rise] = -np.pi / 2
        pha_tnpi[zerox_rise] = -np.pi / 2
    if zerox_decay is not None:
        pha_tpi[zerox_decay] = np.pi / 2
        pha_tnpi[zerox_decay] = np.pi / 2

    # Define phases
    pha_tpi[ps] = 0
    pha_tpi[ts] = np.pi
    pha_tnpi[ps] = 0
    pha_tnpi[ts] = -np.pi

    # Interpolate to find all phases
    pha_tpi = np.interp(times, times[~np.isnan(pha_tpi)], pha_tpi[~np.isnan(pha_tpi)])
    pha_tnpi = np.interp(times, times[~np.isnan(pha_tnpi)], pha_tnpi[~np.isnan(pha_tnpi)])

    # For the phase time series in which the trough is negative pi:
    # Replace the decaying periods with these periods in the phase time
    # series in which the trough is pi
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
