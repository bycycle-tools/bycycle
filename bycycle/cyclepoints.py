"""
cyclespoints.py
Functions to determine the locations of peaks, troughs, and zerocrossings (rise and decay)
for individual cycles
"""

import numpy as np
from bycycle.filt import bandpass_filter


def find_extrema(x, Fs, f_range, boundary=None, first_extrema='peak',
                 filter_kwargs=None):
    """
    Identify peaks and troughs in a time series.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        sampling rate
    f_range : (low, high), Hz
        frequency range for narrowband signal of interest,
        used to find zerocrossings of the oscillation
    boundary : int
        number of samples from edge of recording to ignore
    first_extrema: str or None
        if 'peak', then force the output to begin with a peak and end in a trough
        if 'trough', then force the output to begin with a trough and end in peak
        if None, force nothing
    filter_kwargs : dict
        keyword arguments to the filt.bandpass_filter(), such as 'N_cycles' or 'N_seconds'
        to control filter length

    Returns
    -------
    Ps : array-like 1d
        indices at which oscillatory peaks occur in the input signal x
    Ts : array-like 1d
        indices at which oscillatory troughs occur in the input signal x

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
        boundary = int(np.ceil(Fs / float(f_range[0])))

    # Narrowband filter signal
    x_filt = bandpass_filter(x, Fs, f_range, remove_edge_artifacts=False, **filter_kwargs)

    # Find rising and falling zerocrossings (narrowband)
    zeroriseN = _fzerorise(x_filt)
    zerofallN = _fzerofall(x_filt)

    # Compute number of peaks and troughs
    if zeroriseN[-1] > zerofallN[-1]:
        P = len(zeroriseN) - 1
        T = len(zerofallN)
    else:
        P = len(zeroriseN)
        T = len(zerofallN) - 1

    # Calculate peak samples
    Ps = np.zeros(P, dtype=int)
    for p in range(P):
        # Calculate the sample range between the most recent zero rise
        # and the next zero fall
        mrzerorise = zeroriseN[p]
        nfzerofall = zerofallN[zerofallN > mrzerorise][0]
        # Identify time fo peak
        Ps[p] = np.argmax(x[mrzerorise:nfzerofall]) + mrzerorise

    # Calculate trough samples
    Ts = np.zeros(T, dtype=int)
    for tr in range(T):
        # Calculate the sample range between the most recent zero fall
        # and the next zero rise
        mrzerofall = zerofallN[tr]
        nfzerorise = zeroriseN[zeroriseN > mrzerofall][0]
        # Identify time of trough
        Ts[tr] = np.argmin(x[mrzerofall:nfzerorise]) + mrzerofall

    # Remove peaks and troughs within the boundary limit
    Ps = Ps[np.logical_and(Ps > boundary, Ps < len(x) - boundary)]
    Ts = Ts[np.logical_and(Ts > boundary, Ts < len(x) - boundary)]

    # Force the first extrema to be as desired
    # Assure equal # of peaks and troughs
    if first_extrema == 'peak':
        if Ps[0] > Ts[0]:
            Ts = Ts[1:]
        if Ps[-1] > Ts[-1]:
            Ps = Ps[:-1]
    elif first_extrema == 'trough':
        if Ts[0] > Ps[0]:
            Ps = Ps[1:]
        if Ts[-1] > Ps[-1]:
            Ts = Ts[:-1]
    elif first_extrema is None:
        pass
    else:
        raise ValueError('Parameter "first_extrema" is invalid')

    return Ps, Ts


def _fzerofall(data):
    """Find zerocrossings on falling edge of a filtered signal"""
    pos = data > 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0]


def _fzerorise(data):
    """Find zerocrossings on rising edge of a filtered signal"""
    pos = data < 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0]


def find_zerox(x, Ps, Ts):
    """
    Find zerocrossings within each cycle after peaks and troughs are identified.
    A rising zerocrossing occurs when the voltage crosses
    midway between the trough voltage and subsequent peak voltage.
    A decay zerocrossing is defined similarly.
    If this voltage is crossed at multiple times, the temporal median is taken
    as the zerocrossing.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : numpy arrays 1d
        time points of oscillatory peaks
    Ts : numpy arrays 1d
        time points of osillatory troughs

    Returns
    -------
    zeroxR : array-like 1d
        indices at which oscillatory rising zerocrossings occur
    zeroxD : array-like 1d
        indices at which oscillatory decaying zerocrossings occur

    Notes
    -----
    * Sometimes, due to noise in estimating peaks and troughs when the oscillation
    is absent, the estimated peak might be lower than an adjacent trough. If this
    occurs, the rise and decay zerocrossings will be set to be halfway between
    the peak and trough. Burst detection should be used in order to ignore these
    periods of the signal.
    """

    # Calculate the number of rises and decays
    if Ps[0] < Ts[0]:
        N_rises = len(Ps) - 1
        N_decays = len(Ts)
        idx_bias = 0
    else:
        N_rises = len(Ps)
        N_decays = len(Ts) - 1
        idx_bias = 1

    # Find zerocrossings for rise
    zeroxR = np.zeros(N_rises, dtype=int)
    for i in range(N_rises):
        x_temp = np.copy(x[Ts[i]:Ps[i + 1 - idx_bias] + 1])
        x_temp -= (x_temp[0] + x_temp[-1]) / 2.

        # If data is all 0s, just set the zerocrossing to be halfway between.
        if np.sum(np.abs(x_temp)) == 0:
            zeroxR[i] = Ts[i] + int(len(x_temp) / 2.)

        # If rise is actually decay, just set the zerocrossing to be halfway between.
        elif x_temp[0] > x_temp[-1]:
            zeroxR[i] = Ts[i] + int(len(x_temp) / 2.)

        else:
            zeroxR[i] = Ts[i] + int(np.median(_fzerorise(x_temp)))

    # Find zerocrossings for decays
    zeroxD = np.zeros(N_decays, dtype=int)
    for i in range(N_decays):
        x_temp = np.copy(x[Ps[i]:Ts[i + idx_bias] + 1])
        x_temp -= (x_temp[0] + x_temp[-1]) / 2.

        # If data is all 0s, just set the zerocrossing to be halfway between.
        if np.sum(np.abs(x_temp)) == 0:
            zeroxD[i] = Ps[i] + int(len(x_temp) / 2.)

        # If decay is actually rise, just set the zerocrossing to be halfway between.
        elif x_temp[0] < x_temp[-1]:
            zeroxD[i] = Ps[i] + int(len(x_temp) / 2.)
        else:
            zeroxD[i] = Ps[i] + int(np.median(_fzerofall(x_temp)))

    return zeroxR, zeroxD


def extrema_interpolated_phase(x, Ps, Ts, zeroxR=None, zeroxD=None):
    """
    Use peaks (phase 0) and troughs (phase pi/-pi) to estimate
    instantaneous phase. Also use rise and decay zerocrossings
    (phase -pi/2 and pi/2, respectively) if provided.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs
    zeroxR : array-like 1d
        indices at which oscillatory rising zerocrossings occur
    zeroxD : array-like 1d
        indices at which oscillatory decaying zerocrossings occur

    Returns
    -------
    pha : array-like 1d
        instantaneous phase

    Notes
    -----
    Sometimes, due to noise, extrema and zerocrossing estimation
    is poor, and for example, the same index may be assigned to
    both a peak and a decaying zerocrossing. Because of this,
    we first assign phase values by zerocrossings, and then
    may overwrite them with extrema phases. Use of burst detection
    will help avoid analyzing the oscillatory properties of
    non-oscillatory sections of the signal.
    """

    # Initialize phase arrays
    # 2 phase arrays: trough pi and trough -pi
    L = len(x)
    t = np.arange(L)
    pha_tpi = np.zeros(L) * np.nan
    pha_tnpi = np.zeros(L) * np.nan

    # If specified, assign phases to zerocrossings
    if zeroxR is not None:
        pha_tpi[zeroxR] = -np.pi / 2
        pha_tnpi[zeroxR] = -np.pi / 2
    if zeroxD is not None:
        pha_tpi[zeroxD] = np.pi / 2
        pha_tnpi[zeroxD] = np.pi / 2

    # Define phases
    pha_tpi[Ps] = 0
    pha_tpi[Ts] = np.pi
    pha_tnpi[Ps] = 0
    pha_tnpi[Ts] = -np.pi

    # Interpolate to find all phases
    pha_tpi = np.interp(t, t[~np.isnan(pha_tpi)], pha_tpi[~np.isnan(pha_tpi)])
    pha_tnpi = np.interp(t, t[~np.isnan(pha_tnpi)], pha_tnpi[~np.isnan(pha_tnpi)])

    # For the phase time series in which the trough is negative pi:
    # Replace the decaying periods with these periods in the phase time
    # series in which the trough is pi
    diffs = np.diff(pha_tnpi)
    diffs = np.append(diffs, 99)
    pha_tnpi[diffs < 0] = pha_tpi[diffs < 0]

    # Assign the periods before the first empirical phase timepoint to NaN
    diffs = np.diff(pha_tnpi)
    first_empirical_idx = next(i for i, xi in enumerate(diffs) if xi > 0)
    pha_tnpi[:first_empirical_idx] = np.nan

    # Assign the periods after the last empirical phase timepoint to NaN
    diffs = np.diff(pha_tnpi)
    last_empirical_idx = next(i for i, xi in enumerate(diffs[::-1]) if xi > 0)
    pha_tnpi[-last_empirical_idx + 1:] = np.nan

    return pha_tnpi
