"""
filt.py
Filter a neural signal using a bandpass, highpass, lowpass, or bandstop filter.
"""

import warnings
import math
import numpy as np
import scipy as sp
from scipy import signal as spsignal
import matplotlib.pyplot as plt


def bandpass_filter(signal, Fs, fc, N_cycles=None, N_seconds=None,
                    plot_frequency_response=False, return_kernel=False,
                    compute_transition_band=True,
                    remove_edge_artifacts=True):
    """
    Apply a bandpass filter to a neural signal

    Parameters
    ----------
    signal : array-like 1d
        voltage time series
    Fs : float
        The sampling rate
    fc : list or tuple (lo, hi)
        The low and high cutoff frequencies for the filter
    N_cycles : float, optional (default: 3)
        Length of filter in terms of number of cycles at the low cutoff frequency
        By default, this is set to 3 cycles of the low cutoff frequency
        This parameter is overwritten by 'N_seconds'
    N_seconds : float, optional
        Length of filter (seconds)
    plot_frequency_response : bool, optional
        if True, plot the frequency response of the filter
    return_kernel : bool, optional
        if True, return the complex filter kernel
    compute_transition_band : bool, optional
        if True, the transition bandwidth is computed,
        defined as the frequency range between -20dB and -3dB attenuation
        This is printed as a warning if the transition bandwidth is
        wider than the passband width
    remove_edge_artifacts : bool, optional
        if True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan

    Returns
    -------
    signal_filt : array-like 1d
        filtered time series
    kernel : length-2 tuple of arrays
        filter kernel
        returned only if 'return_kernel' == True
    """

    # Set default and throw warning if no filter length provided
    if N_cycles is None and N_seconds is None:
        N_cycles = 3
        warnings.warn('''
            No filter length provided. Using default of 3 cycles of the
            low cutoff frequency.
            ''')

    # Assure cutoff frequency input is valid
    if len(fc) != 2:
        raise ValueError('Two cutoff frequencies required for bandpass and bandstop filters')
    if fc[0] >= fc[1]:
        raise ValueError('Second cutoff frequency must be greater than first for bandpass and bandstop filters')

    # Remove any NaN on the edges of 'signal'
    first_nonan = np.where(~np.isnan(signal))[0][0]
    last_nonan = np.where(~np.isnan(signal))[0][-1] + 1
    signal_old = np.copy(signal)
    signal = signal[first_nonan:last_nonan]


    # Compute filter length if specified in seconds
    if N_seconds is not None:
        N = int(np.ceil(Fs * N_seconds))
    else:
        N = int(np.ceil(Fs * N_cycles / fc[0]))

    # Force filter length to be odd
    if N % 2 == 0:
        N = int(N + 1)

    # Raise an error if the filter is longer than the signal
    if N >= len(signal):
        raise ValueError(
            '''The designed filter (length: {:d}) is longer than the signal (length: {:d}).
            The filter needs to be shortened by decreasing the N_cycles or N_seconds parameter.
            However, this will decrease the frequency resolution of the filter.'''.format(N, len(signal)))

    # Compute nyquist frequency
    f_nyq = Fs / 2.

    # Design filter
    kernel = spsignal.firwin(N, (fc[0], fc[1]), pass_zero=False, nyq=f_nyq)

    # Apply filter
    signal_filt = np.convolve(kernel, signal, 'same')

    # Plot frequency response, if desired
    if plot_frequency_response:
        _plot_frequency_response(Fs, kernel)

    # Compute transition bandwidth
    if compute_transition_band:

        # Compute the frequency response in terms of Hz and dB
        b = kernel
        a = 1
        w, h = spsignal.freqz(b, a)
        f_db = w * Fs / (2. * np.pi)
        db = 20 * np.log10(abs(h))

        # Compute pass bandwidth and transition bandwidth
        try:
            pass_bw = fc[1] - fc[0]
            # Identify edges of transition band (-3dB and -20dB)
            cf_20db_1 = next(f_db[i] for i in range(len(db)) if db[i] > -20)
            cf_3db_1 = next(f_db[i] for i in range(len(db)) if db[i] > -3)
            cf_20db_2 = next(f_db[i] for i in range(len(db))[::-1] if db[i] > -20)
            cf_3db_2 = next(f_db[i] for i in range(len(db))[::-1] if db[i] > -3)
            # Compute transition bandwidth
            transition_bw1 = cf_3db_1 - cf_20db_1
            transition_bw2 = cf_20db_2 - cf_3db_2
            transition_bw = max(transition_bw1, transition_bw2)

            if cf_20db_1 == f_db[0]:
                warnings.warn('The low frequency stopband never gets attenuated by more than 20dB. Increase filter length.')
            if cf_20db_2 == f_db[-1]:
                warnings.warn('The high frequency stopband never gets attenuated by more than 20dB. Increase filter length.')

            # Raise warning if transition bandwidth is greater than passband width
            if transition_bw > pass_bw:
                warnings.warn('Transition bandwidth is ' + str(np.round(transition_bw, 1)) + ' Hz. This is greater than the desired pass/stop bandwidth of ' + str(np.round(pass_bw, 1)) + ' Hz')
        except StopIteration:
            raise warnings.warn('Error computing transition bandwidth of the filter. Defined filter length may be too short.')

    # Remove edge artifacts
    if remove_edge_artifacts:
        N_rmv = int(np.ceil(N / 2))
        signal_filt[:N_rmv] = np.nan
        signal_filt[-N_rmv:] = np.nan

    # Add NaN back on the edges of 'signal', if there were any at the beginning
    signal_filt_full = np.ones(len(signal_old)) * np.nan
    signal_filt_full[first_nonan:last_nonan] = signal_filt
    signal_filt = signal_filt_full

    # Return kernel if desired
    if return_kernel:
        return signal_filt, kernel
    else:
        return signal_filt


def _plot_frequency_response(Fs, b, a=1):
    """Compute frequency response of a filter kernel b with sampling rate Fs"""
    w, h = spsignal.freqz(b, a)
    # Plot frequency response
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(w * Fs / (2. * np.pi), 20 * np.log10(abs(h)), 'k')
    plt.title('Frequency response')
    plt.ylabel('Attenuation (dB)')
    plt.xlabel('Frequency (Hz)')
    if isinstance(a, int):
        # Plot filter kernel
        plt.subplot(1, 2, 2)
        plt.plot(b, 'k')
        plt.title('Kernel')
    plt.show()


def phase_by_time(x, Fs, f_range,
                  filter_fn=None, filter_kwargs=None,
                  hilbert_increase_N=False):
    """
    Calculate the phase time series of a neural oscillation

    Parameters
    ----------
    x : array-like, 1d
        Time series
    Fs : float, Hz
        Sampling rate
    f_range : (low, high), Hz
        Frequency range
    filter_fn : function, optional
        The filtering function, with api:
        `filterfn(x, Fs, pass_type, fc, remove_edge_artifacts=True)
    filter_kwargs : dict, optional
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool, optional
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x

    Returns
    -------
    pha : array-like, 1d
        Time series of phase
    """
    # Set default filtering parameters
    if filter_fn is None:
        filter_fn = bandpass_filter
    if filter_kwargs is None:
        filter_kwargs = {}
    # Filter signal
    x_filt = filter_fn(x, Fs, fc=f_range,
                       remove_edge_artifacts=False, **filter_kwargs)
    # Compute phase time series
    pha = np.angle(_hilbert_ignore_nan(x_filt, hilbert_increase_N=hilbert_increase_N))
    return pha


def amp_by_time(x, Fs, f_range,
                filter_fn=None, filter_kwargs=None,
                hilbert_increase_N=False):
    """
    Calculate the amplitude time series

    Parameters
    ----------
    x : array-like, 1d
        Time series
    Fs : float, Hz
        Sampling rate
    f_range : (low, high), Hz
        The frequency filtering range
    filter_fn : function, optional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
        Must have the same API as filt.bandpass
    filter_kwargs : dict, optional
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool, optional
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x

    Returns
    -------
    amp : array-like, 1d
        Time series of amplitude
    """
    # Set default filtering parameters
    if filter_fn is None:
        filter_fn = bandpass_filter
    if filter_kwargs is None:
        filter_kwargs = {}
    # Filter signal
    x_filt = filter_fn(x, Fs, fc=f_range,
                       remove_edge_artifacts=False, **filter_kwargs)
    # Compute amplitude time series
    amp = np.abs(_hilbert_ignore_nan(x_filt, hilbert_increase_N=hilbert_increase_N))
    return amp


def freq_by_time(x, Fs, f_range):
    '''
    Estimate the instantaneous frequency at each sample

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        sampling rate
    f_range : (low, high), Hz
        frequency range for filtering

    Returns
    -------
    i_f : float
        estimate instantaneous frequency for each sample in 'x'

    Notes
    -----
    * This function assumes monotonic phase, so
    a phase slip will be processed as a very high frequency
    '''
    pha = phase_by_time(x, Fs, f_range)
    phadiff = np.diff(pha)
    phadiff[phadiff < 0] = phadiff[phadiff < 0] + 2 * np.pi
    i_f = Fs * phadiff / (2 * np.pi)
    i_f = np.insert(i_f, 0, np.nan)
    return i_f


def _hilbert_ignore_nan(x, hilbert_increase_N=False):
    """
    Compute the hilbert transform of x.
    Ignoring the boundaries of x that are filled with NaN
    """
    # Extract the signal that is not nan
    first_nonan = np.where(~np.isnan(x))[0][0]
    last_nonan = np.where(~np.isnan(x))[0][-1] + 1
    x_nonan = x[first_nonan:last_nonan]

    # Compute hilbert transform of signal without nans
    if hilbert_increase_N:
        N = len(x_nonan)
        N2 = 2**(int(math.log(N, 2)) + 1)
        x_hilb_nonan = spsignal.hilbert(x_nonan, N2)[:N]
    else:
        x_hilb_nonan = spsignal.hilbert(x_nonan)

    # Fill in output hilbert with nans on edges
    x_hilb = np.ones(len(x), dtype=complex) * np.nan
    x_hilb[first_nonan:last_nonan] = x_hilb_nonan
    return x_hilb
