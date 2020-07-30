"""Functions to determine the shape features for individual cycles."""

import numpy as np
import pandas as pd

from neurodsp.timefrequency import amp_by_time

from bycycle.cyclepoints import find_extrema, find_zerox

###################################################################################################
###################################################################################################

def compute_shape_features(sig, fs, f_range, center_extrema='peak', find_extrema_kwargs=None,
                           hilbert_increase_n=False, n_cycles=3, return_samples=True):
    """Compute shapes parameters of each cycle, used for determining burst features.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    center_extrema : {'peak', 'trough'}
        The center extrema in the cycle.

        - 'peak' : cycles are defined trough-to-trough
        - 'trough' : cycles are defined peak-to-peak

    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (:func:`~.find_extrema`)
        to change filter parameters or boundary. By default, it sets the filter length to three
        cycles of the low cutoff frequency (``f_range[0]``).
    hilbert_increase_n : bool, optional, default: False
        Corresponding kwarg for :func:`~neurodsp.timefrequency.hilbert.amp_by_time`.
        If true, this zero-pads the signal when computing the Fourier transform, which can be
        necessary for computing it in a reasonable amount of time.
    n_cycles : int, optional, default: 3
        Length of filter, in number of cycles, at the lower cutoff frequency.
    return_samples : bool, optional, default: True
        Returns samples indices of cyclepoints used for determining features if True.

    Returns
    -------
    df_shape_features : pandas.DataFrame
        Dataframe containing cycle shape features. Each row is one cycle. Columns:

        - ``period`` : period of the cycle
        - ``time_decay`` : time between peak and next trough
        - ``time_rise`` : time between peak and previous trough
        - ``time_peak`` : time between rise and decay zero-crosses
        - ``time_trough`` : duration of previous trough estimated by zero-crossings
        - ``volt_decay`` : voltage change between peak and next trough
        - ``volt_rise`` : voltage change between peak and previous trough
        - ``volt_amp`` : average of rise and decay voltage
        - ``volt_peak`` : voltage at the peak
        - ``volt_trough`` : voltage at the last trough
        - ``time_rdsym`` : fraction of cycle in the rise period
        - ``time_ptsym`` : fraction of cycle in the peak period
        - ``band_amp`` : average analytic amplitude of the oscillation computed using narrowband
          filtering and the Hilbert transform. Filter length is 3 cycles of the low cutoff
          frequency. Average taken across all time points in the cycle.

    df_samples : pandas.DataFrame, optional, default: True
        Dataframe containing sample indices of cyclepoints.
        Columns (listed for peak-centered cycles):

        - ``sample_peak`` : sample of 'sig' at which the peak occurs
        - ``sample_zerox_decay`` : sample of the decaying zero-crossing
        - ``sample_zerox_rise`` : sample of the rising zero-crossing
        - ``sample_last_trough`` : sample of the last trough
        - ``sample_next_trough`` : sample of the next trough

    Notes
    -----
    Peak vs trough centering

        - By default, the first extrema analyzed will be a peak, and the final one a trough.
        - In order to switch the preference, the signal is simply inverted and columns are renamed.
        - Columns are slightly different depending on if ``center_extrema`` is set to 'peak' or
          'trough'.

    """

    # Set defaults if user input is None
    if find_extrema_kwargs is None:
        find_extrema_kwargs = {'filter_kwargs': {'n_cycles': n_cycles}}

    elif 'first_extrema' in find_extrema_kwargs.keys():
        raise ValueError('''This function has been designed to assume that the first extrema
            identified will be a peak. This cannot be overwritten at this time.''')

    # Negate signal if set to analyze trough-centered cycles
    if center_extrema == 'peak':
        pass
    elif center_extrema == 'trough':
        sig = -sig
    else:
        raise ValueError('Parameter "center_extrema" must be either "peak" or "trough"')

    # For each cycle, identify the sample of each extrema and zero-crossing
    df_samples = compute_samples(sig, fs, f_range, **find_extrema_kwargs)

    # Compute durations of period, peaks, and troughs
    period, time_peak, time_trough = compute_durations(df_samples)

    # Compute extrema voltage
    volt_peak, volt_trough = compute_extrema_voltage(df_samples, sig)

    # Compute rise-decay and peak-trough features and characteristics
    sym_features = compute_symmetry(df_samples, sig, period=period,
                                    time_peak=time_peak, time_trough=time_trough)

    # Compute average oscillatory amplitude estimate during cycle
    band_amp = compute_band_amp(df_samples, sig, fs, f_range,
                                hilbert_increase_n=hilbert_increase_n, n_cycles=3)

    # Organize shape features into a dataframe
    shape_features = {}
    shape_features['period'] = period
    shape_features['time_peak'] = time_peak
    shape_features['time_trough'] = time_trough
    shape_features['volt_peak'] = volt_peak
    shape_features['volt_trough'] = volt_trough
    shape_features['time_decay'] = sym_features['time_decay']
    shape_features['time_rise'] = sym_features['time_rise']
    shape_features['volt_decay'] = sym_features['volt_decay']
    shape_features['volt_rise'] = sym_features['volt_rise']
    shape_features['volt_amp'] = sym_features['volt_amp']
    shape_features['time_rdsym'] = sym_features['time_rdsym']
    shape_features['time_ptsym'] = sym_features['time_ptsym']
    shape_features['band_amp'] = band_amp
    df_shape_features = pd.DataFrame.from_dict(shape_features)

    # Rename the dataframe if trough centered
    df_shape_features, df_samples = _rename_df(center_extrema, df_samples, df_shape_features)

    if return_samples:
        return df_shape_features, df_samples

    return df_shape_features


def _rename_df(center_extrema, df_samples, df_features):
    """Helper function for compute_shape_features."""

    # Rename columns if they are actually trough-centered
    if center_extrema == 'trough':

        samples_rename_dict = {'sample_peak': 'sample_trough',
                               'sample_zerox_decay': 'sample_zerox_rise',
                               'sample_zerox_rise': 'sample_zerox_decay',
                               'sample_last_trough': 'sample_last_peak',
                               'sample_next_trough': 'sample_next_peak'}

        features_rename_dict = {'time_peak': 'time_trough',
                                'time_trough': 'time_peak',
                                'volt_peak': 'volt_trough',
                                'volt_trough': 'volt_peak',
                                'time_rise': 'time_decay',
                                'time_decay': 'time_rise',
                                'volt_rise': 'volt_decay',
                                'volt_decay': 'volt_rise'}

        df_samples.rename(columns=samples_rename_dict, inplace=True)
        df_features.rename(columns=features_rename_dict, inplace=True)

        # Need to reverse symmetry measures
        df_features['volt_peak'] = -df_features['volt_peak']
        df_features['volt_trough'] = -df_features['volt_trough']
        df_features['time_rdsym'] = 1 - df_features['time_rdsym']
        df_features['time_ptsym'] = 1 - df_features['time_ptsym']

    return df_features, df_samples


def compute_durations(df_samples):
    """Compute the time durations of periods, peaks, and troughs.

    Parameters
    ---------
    df_samples : pandas.DataFrame
        Dataframe containing sample indices of cyclepoints.

    Returns
    -------
    period : 1d array
        The period of each cycle.
    time_peak : 1d array
        Time between peak and next trough.
    time_trough : 1d array
        Time between peak and previous trough.
    """

    period = df_samples['sample_next_trough'] - df_samples['sample_last_trough']
    time_peak = df_samples['sample_zerox_decay'] - df_samples['sample_zerox_rise']
    time_trough = df_samples['sample_zerox_rise'] - df_samples['sample_last_zerox_decay']

    return period, time_peak, time_trough


def compute_extrema_voltage(df_samples, sig):
    """Compute the voltage of peaks and troughs.

    Parameters
    ---------
    df_samples : pandas.DataFrame
        Dataframe containing sample indices of cyclepoints.
    sig : 1d array
        Time series.

    Returns
    -------
    volt_peak : 1d array
        Voltage at the peak.
    volt_trough : 1d array
        Voltage at the last trough.
    """

    volt_peak = sig[df_samples['sample_peak']]
    volt_trough = sig[df_samples['sample_last_trough']]

    return volt_peak, volt_trough


def compute_symmetry(df_samples, sig, period=None, time_peak=None, time_trough=None):
    """Compute rise-decay and peak-trough characteristics.

    Parameters
    ---------
    df_samples : pandas.DataFrame
        Dataframe containing sample indices of cyclepoints.
    sig : 1d array
        Time series.
    period : 1d array, optional, default: None
        The period of each cycle.
    time_peak : 1d array, optional, default: None
        Time between peak and next trough.
    time_trough : 1d array, optional, default: None
        Time between peak and previous trough

    Returns
    -------
    sym_features : dict
        Contains 1d arrays of symmetry features. Keys include:

        - time_decay : Time between peak and next trough.
        - time_rise : Time between peak and previous trough.
        - volt_decay : Voltage change between peak and next trough.
        - volt_rise : Voltage change between peak and previous trough.
        - volt_amp : Average of rise and decay voltage.
        - time_rdsym : Fraction of cycle in the rise period.
        - time_ptsym : Fraction of cycle in the peak period.

    """

    # Determine rise and decay characteristics
    sym_features = {}

    time_decay = df_samples['sample_next_trough'] - df_samples['sample_peak']
    time_rise = df_samples['sample_peak'] - df_samples['sample_last_trough']
    volt_decay = sig[df_samples['sample_peak']] - sig[df_samples['sample_next_trough']]
    volt_rise = sig[df_samples['sample_peak']] - sig[df_samples['sample_last_trough']]
    volt_amp = (volt_decay + volt_rise) / 2

    # Compute rise-decay symmetry features
    if period is None or time_peak is None or time_trough is None:
        period, time_peak, time_trough = compute_durations(df_samples)

    time_rdsym = time_rise / period

    # Compute peak-trough symmetry features
    time_ptsym = time_peak / (time_peak + time_trough)

    sym_features['time_decay'] = time_decay
    sym_features['time_rise'] = time_rise
    sym_features['volt_decay'] = volt_decay
    sym_features['volt_rise'] = volt_rise
    sym_features['volt_amp'] = volt_amp
    sym_features['time_rdsym'] = time_rdsym
    sym_features['time_ptsym'] = time_ptsym

    return sym_features


def compute_band_amp(df_samples, sig, fs, f_range, hilbert_increase_n=False, n_cycles=3):
    """Compute the average amplitude of each oscillation.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    hilbert_increase_n : bool, optional, default: False
        Corresponding kwarg for :func:`~neurodsp.timefrequency.hilbert.amp_by_time`.
        If true, this zero-pads the signal when computing the Fourier transform, which can be
        necessary for computing it in a reasonable amount of time.
    n_cycles : int, optional, default: 3
        Length of filter, in number of cycles, at the lower cutoff frequency.

    Returns
    -------
    band_amp : 1d array
        Average analytic amplitude of the oscillation computed using narrowband filtering and the
        Hilbert transform. Filter length is 3 cycles of the low cutoff frequency. Average taken
        across all time points in the cycle.
    """

    amp = amp_by_time(sig, fs, f_range, n_cycles=n_cycles, hilbert_increase_n=hilbert_increase_n)

    troughs = np.append(df_samples['sample_last_trough'].values[0],
                        df_samples['sample_next_trough'].values)

    band_amp = [np.mean(amp[troughs[sig_idx]:troughs[sig_idx + 1]]) for
                sig_idx in range(len(df_samples['sample_peak']))]

    return band_amp


def compute_samples(sig, fs, f_range, **find_extrema_kwargs):
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
        Keyword arguments for function to find peaks and troughs (:func:`~.find_extrema`)
        to change filter parameters or boundary. By default, it sets the filter length to three
        cycles of the low cutoff frequency (``f_range[0]``).

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

    """

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
