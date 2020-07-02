"""Quantify the shape of oscillatory waveforms on a cycle-by-cycle basis."""

import warnings

import numpy as np
import pandas as pd

from neurodsp.timefrequency import amp_by_time
from neurodsp.burst import detect_bursts_dual_threshold

from bycycle.cyclepoints import find_extrema, find_zerox

###################################################################################################
###################################################################################################

def compute_shapes(sig, fs, f_range, center_extrema='peak',
                   find_extrema_kwargs=None, hilbert_increase_n=False):
    """Compute shapes parameters of each cycle, used for determining burst features.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
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
        to change filter Parameters or boundary. By default, it sets the filter length to three
        cycles of the low cutoff frequency (``f_range[0]``).
    hilbert_increase_n : bool, optional, default: False
        Corresponding kwarg for :func:`~neurodsp.timefrequency.hilbert.amp_by_time`.
        If true, this zero-pads the signal when computing the Fourier transform, which can be
        necessary for computing it in a reasonable amount of time.

    Returns
    -------
    df_shapes : pandas.DataFrame
        Dataframe containing shape features that may be used to determine burst features using
        :func:`~.compute_features`. Each row is one cycle.
        Columns (listed for peak-centered cycles):

        - ``sample_peak`` : sample of 'sig' at which the peak occurs
        - ``sample_zerox_decay`` : sample of the decaying zero-crossing
        - ``sample_zerox_rise`` : sample of the rising zero-crossing
        - ``sample_last_trough`` : sample of the last trough
        - ``sample_next_trough`` : sample of the next trough
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
        find_extrema_kwargs = {'filter_kwargs': {'n_cycles': 3}}
    else:
        # Raise warning if switch from peak start to trough start
        if 'first_extrema' in find_extrema_kwargs.keys():
            raise ValueError('''
                This function assumes that the first extrema identified will be a peak.
                This cannot be overwritten at this time.''')

    # Negate signal if to analyze trough-centered cycles
    if center_extrema == 'peak':
        pass
    elif center_extrema == 'trough':
        sig = -sig
    else:
        raise ValueError('Parameter "center_extrema" must be either "P" or "T"')

    # Find peak and trough locations in the signal
    ps, ts = find_extrema(sig, fs, f_range, **find_extrema_kwargs)

    # Find zero-crossings
    zerox_rise, zerox_decay = find_zerox(sig, ps, ts)

    # For each cycle, identify the sample of each extrema and zero-crossing
    shapes = {}
    shapes['sample_peak'] = ps[1:]
    shapes['sample_zerox_decay'] = zerox_decay[1:]
    shapes['sample_zerox_rise'] = zerox_rise
    shapes['sample_last_trough'] = ts[:-1]
    shapes['sample_next_trough'] = ts[1:]

    # Compute duration of period
    shapes['period'] = shapes['sample_next_trough'] - shapes['sample_last_trough']

    # Compute duration of peak
    shapes['time_peak'] = shapes['sample_zerox_decay'] - shapes['sample_zerox_rise']

    # Compute duration of last trough
    shapes['time_trough'] = zerox_rise - zerox_decay[:-1]

    # Determine extrema voltage
    shapes['volt_peak'] = sig[ps[1:]]
    shapes['volt_trough'] = sig[ts[:-1]]

    # Determine rise and decay characteristics
    shapes['time_decay'] = (ts[1:] - ps[1:])
    shapes['time_rise'] = (ps[1:] - ts[:-1])

    shapes['volt_decay'] = sig[ps[1:]] - sig[ts[1:]]
    shapes['volt_rise'] = sig[ps[1:]] - sig[ts[:-1]]
    shapes['volt_amp'] = (shapes['volt_decay'] + shapes['volt_rise']) / 2

    # Compute rise-decay symmetry features
    shapes['time_rdsym'] = shapes['time_rise'] / shapes['period']

    # Compute peak-trough symmetry features
    shapes['time_ptsym'] = shapes['time_peak'] / (shapes['time_peak'] + shapes['time_trough'])

    # Compute average oscillatory amplitude estimate during cycle
    amp = amp_by_time(sig, fs, f_range, hilbert_increase_n=hilbert_increase_n, n_cycles=3)

    shapes['band_amp'] = [np.mean(amp[ts[sig_idx]:ts[sig_idx + 1]]) for sig_idx in
                          range(len(shapes['sample_peak']))]

    # Convert feature dictionary into a DataFrame
    df_shapes = pd.DataFrame.from_dict(shapes)

    # Rename columns if they are actually trough-centered
    if center_extrema == 'trough':

        rename_dict = {'sample_peak': 'sample_trough',
                       'sample_zerox_decay': 'sample_zerox_rise',
                       'sample_zerox_rise': 'sample_zerox_decay',
                       'sample_last_trough': 'sample_last_peak',
                       'sample_next_trough': 'sample_next_peak',
                       'time_peak': 'time_trough',
                       'time_trough': 'time_peak',
                       'volt_peak': 'volt_trough',
                       'volt_trough': 'volt_peak',
                       'time_rise': 'time_decay',
                       'time_decay': 'time_rise',
                       'volt_rise': 'volt_decay',
                       'volt_decay': 'volt_rise'}
        df_shapes.rename(columns=rename_dict, inplace=True)

        # Need to reverse symmetry measures
        df_shapes['volt_peak'] = -df_shapes['volt_peak']
        df_shapes['volt_trough'] = -df_shapes['volt_trough']
        df_shapes['time_rdsym'] = 1 - df_shapes['time_rdsym']
        df_shapes['time_ptsym'] = 1 - df_shapes['time_ptsym']

    return df_shapes


def compute_features(df_shapes, sig, dual_threshold_kwargs=None):
    """Compute burst features for each cycle.

    Parameters
    ----------
    df_shapes : pandas DataFrame
        Shape parameters for each cycle, determined using :func:`~.compute_shapes`.
    sig : 1d array
        Voltage time series used for determining monotonicity.
    dual_threshold_kwargs : dict, optional, default: None
        Additional arguments in :func:`~.compute_burst_fraction`.

    Returns
    -------
    df_features : pandas.DataFrame
        Dataframe containing burst features. Each row is one cycle.
        Columns (listed for peak-centered cycles):

    - ``amplitude_fraction`` : normalized amplitude
    - ``amplitude_consistency`` : difference in the rise and decay voltage within a cycle
    - ``period_consistency`` : difference between a cycle’s period and the period of the
        adjacent cycles
    - ``monotonicity`` : fraction of instantaneous voltage changes between consecutive
        samples that are positive during the rise phase and negative during the decay phase

    Notes
    -----
    Dual amplitude threshold burst detection will be used, rather than cycle feature consistency, if
    ``dual_threshold_kwargs`` is not None.
    """

    # Compute burst features.
    df_features = pd.DataFrame()

    # Use feature consistency burst detection
    if dual_threshold_kwargs is None:

        # Custom feature functions may be inserted here as long as an array is return with a length
        #   length equal to the number of cycles, or rows in df_shapes.
        df_features['amplitude_fraction'] = compute_amplitude_fraction(df_shapes)
        df_features['amplitude_consistency'] = compute_amplitude_consistency(df_shapes)
        df_features['period_consistency'] = compute_period_consistency(df_shapes)
        df_features['monotonicity'] = compute_monotonicity(df_shapes, sig)

    # Use dual threshold burst detection
    else:

        df_features['burst_fraction'] = compute_burst_fraction(df_shapes, sig,
                                                               **dual_threshold_kwargs)

    return df_features


def compute_amplitude_fraction(df_shapes):
    """Compute the amplitude fraction of each cycle.

    Parameters
    ----------
    df_shapes : pandas DataFrame
        Shape parameters for each cycle, determined using :func:`~.compute_shapes`.


    Returns
    -------
    amp_fract : 1d array
        The amplitude fraction of each cycle.
    """

    return df_shapes['volt_amp'].rank()/len(df_shapes)


def compute_amplitude_consistency(df_shapes):
    """Compute ampltidue consistency for each cycle.

    Parameters
    ----------
    df_shapes : pandas DataFrame
        Shape parameters for each cycle, determined using :func:`~.compute_shapes`.


    Returns
    -------
    amp_consist : 1d array
        The amplitude consistency of each cycle.
    """

    # Compute amplitude consistency
    cycles = len(df_shapes)
    amplitude_consistency = np.ones(cycles) * np.nan
    rises = df_shapes['volt_rise'].values
    decays = df_shapes['volt_decay'].values

    for cyc in range(1, cycles-1):

        consist_current = np.min([rises[cyc], decays[cyc]]) / np.max([rises[cyc], decays[cyc]])

        if 'sample_peak' in df_shapes.columns:
            consist_last = np.min([rises[cyc], decays[cyc-1]]) / np.max([rises[cyc], decays[cyc-1]])
            consist_next = np.min([rises[cyc+1], decays[cyc]]) / np.max([rises[cyc+1], decays[cyc]])

        else:
            consist_last = np.min([rises[cyc-1], decays[cyc]]) / np.max([rises[cyc-1], decays[cyc]])
            consist_next = np.min([rises[cyc], decays[cyc+1]]) / np.max([rises[cyc], decays[cyc+1]])

        amplitude_consistency[cyc] = np.min([consist_current, consist_next, consist_last])

    return amplitude_consistency


def compute_period_consistency(df_shapes):
    """Compute the period consistency of each cycle.

    Parameters
    ----------
    df_shapes : pandas DataFrame
        Shape parameters for each cycle, determined using :func:`~.compute_shapes`.

    Returns
    -------
    period_consistency : 1d array
        The period consistency of each cycle.
    """

    # Compute period consistency
    cycles = len(df_shapes)
    period_consistency = np.ones(cycles) * np.nan
    periods = df_shapes['period'].values

    for cyc in range(1, cycles-1):

        consist_last = np.min([periods[cyc], periods[cyc-1]]) / \
            np.max([periods[cyc], periods[cyc-1]])
        consist_next = np.min([periods[cyc+1], periods[cyc]]) / \
            np.max([periods[cyc+1], periods[cyc]])

        period_consistency[cyc] = np.min([consist_next, consist_last])

    return period_consistency


def compute_monotonicity(df_shapes, sig):
    """Compute the monotonicity of each cycle.

    Parameters
    ----------
    df_shapes : pandas DataFrame
        Shape parameters for each cycle, determined using :func:`~.compute_shapes`.

    Returns
    -------
    monotonicity : 1d array
        The monotonicity of each cycle.
    """

    # Compute monotonicity
    cycles = len(df_shapes)
    monotonicity = np.ones(cycles) * np.nan

    for idx, row in df_shapes.iterrows():

        if 'sample_peak' in df_shapes.columns:
            rise_period = sig[int(row['sample_last_trough']):int(row['sample_peak'])]
            decay_period = sig[int(row['sample_peak']):int(row['sample_next_trough'])]

        else:
            decay_period = sig[int(row['sample_last_peak']):int(row['sample_trough'])]
            rise_period = sig[int(row['sample_trough']):int(row['sample_next_peak'])]

        decay_mono = np.mean(np.diff(decay_period) < 0)
        rise_mono = np.mean(np.diff(rise_period) > 0)
        monotonicity[idx] = np.mean([decay_mono, rise_mono])

    return monotonicity


def compute_burst_fraction(df_shapes, sig, fs, f_range, amp_threshes=(1, 2),
                           n_cycles_min=3, filter_kwargs=None):
    """ Compute the proportion of a cycle that is bursting.

    Parameters
    ----------
    df_shapes : pandas DataFrame
        Shape parameters for each cycle, determined using :func:`~.compute_shapes`.
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, Hz.
    f_range : tuple of (float, float)
        Frequency range (Hz) for oscillator of interest.
    amp_threshes : tuple (low, high), optional, default: (1, 2)
        Threshold values for determining timing of bursts.
        These values are in units of amplitude (or power, if specified) normalized to
        the median amplitude (value 1).
    n_cycles_min : int, optional, default: 3
        Minimum number of cycles to be identified as truly oscillating needed in a row in
        order for them to remain identified as truly oscillating.
    filter_kwargs : dict, optional, default: None
        Keyword arguments to :func:`~neurodsp.filt.filter.filter_signal`.

    Returns
    -------
    burst_fraction : 1d array
        The proportion of each cycle that is bursting.
    """

    filter_kwargs = {} if filter_kwargs is None else filter_kwargs

    # Detect bursts using the dual amplitude threshold approach
    is_burst = detect_bursts_dual_threshold(sig, fs, amp_threshes, f_range,
                                            min_n_cycles=n_cycles_min, **filter_kwargs)

    # Compute fraction of each cycle that's bursting
    burst_fraction = []
    for _, row in df_shapes.iterrows():
        fraction_bursting = np.mean(is_burst[int(row['sample_last_trough']):
                                             int(row['sample_next_trough'] + 1)])
        burst_fraction.append(fraction_bursting)

    return burst_fraction
