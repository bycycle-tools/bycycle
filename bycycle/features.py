"""
features.py
Quantify the shape of oscillatory waveforms on a cycle-by-cycle basis
"""

import numpy as np
import pandas as pd
from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.filt import amp_by_time
from bycycle.burst import detect_bursts_cycles, detect_bursts_df_amp
import warnings


def compute_features(x, Fs, f_range,
                     center_extrema='P',
                     burst_detection_method='cycles',
                     burst_detection_kwargs=None,
                     find_extrema_kwargs=None,
                     hilbert_increase_N=False):
    """
    Segment a recording into individual cycles and compute
    simple features for each cycle

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        sampling rate (Hz)
    f_range : (low, high), Hz
        frequency range for narrowband signal of interest
    center_extrema : str
        The center extrema in the cycle
        'P' : cycles are defined trough-to-trough
        'T' : cycles are defined peak-to-peak
    burst_detection_method: str in ('consistency', 'amplitude')
        Method for detecting bursts
        'cycles': detect bursts based on the consistency
            of consecutive periods and amplitudes
        'amp': detect bursts using an amplitude threshold
    burst_detection_kwargs : dict or None
        Keyword arguments for function to find label cycles
        as in or not in an oscillation
    find_extrema_kwargs : dict or None
        Keyword arguments for function to find peaks and
        troughs (cyclepoints.find_extrema) to change filter
        parameters or boundary.
        By default, it sets the filter length to three cycles
        of the low cutoff frequency (`f_range[0]`)
    hilbert_increase_N : bool
        corresponding kwarg for filt.amp_by_time
        If true, this zeropads the signal when computing the
        Fourier transform, which can be necessary for
        computing it in a reasonable amount of time.

    Returns
    -------
    df : pandas DataFrame
        dataframe containing several features and identifiers
        for each cycle. Each row is one cycle.
        Note that columns are slightly different depending on if
        'center_extrema' is set to 'P' or 'T'.
        Each column is described below for peak-centered cycles,
        but are similar for trough-centered cycles:
        sample_peak : sample of 'x' at which the peak occurs
        sample_zerox_decay : sample of the decaying zerocrossing
        sample_zerox_rise : sample of the rising zerocrossing
        sample_last_trough : sample of the last trough
        sample_next_trough : sample of the next trough
        period : period of the cycle
        time_decay : time between peak and next trough
        time_rise : time between peak and previous trough
        time_peak : time between rise and decay zerocrosses
        time_trough : duration of previous trough estimated by zerocrossings
        volt_decay : voltage change between peak and next trough
        volt_rise : voltage change between peak and previous trough
        volt_amp : average of rise and decay voltage
        volt_peak : voltage at the peak
        volt_trough : voltage at the last trough
        time_rdsym : fraction of cycle in the rise period
        time_ptsym : fraction of cycle in the peak period
        band_amp : average analytic amplitude of the oscillation
                   computed using narrowband filtering and the Hilbert
                   transform. Filter length is 3 cycles of the low
                   cutoff frequency. Average taken across all time points
                   in the cycle.
        is_burst : True if the cycle is part of a detected oscillatory burst

    Notes
    -----
    * By default, the first extrema analyzed will be a peak,
    and the final one a trough. In order to switch the preference,
    the signal is simply inverted and columns are renamed.
    """

    # Set defaults if user input is None
    if burst_detection_kwargs is None:
        burst_detection_kwargs = {}
        warnings.warn('''
            No burst detection parameters are provided.
            This is very much not recommended.
            Please inspect your data and choose appropriate
            parameters for "burst_detection_kwargs".
            Default burst detection parameters are likely
            not well suited for your desired application.
            ''')
    if find_extrema_kwargs is None:
        find_extrema_kwargs = {'filter_kwargs': {'N_cycles': 3}}
    else:
        # Raise warning if switch from peak start to trough start
        if 'first_extrema' in find_extrema_kwargs.keys():
            raise ValueError('''This function has been designed
                to assume that the first extrema identified will be a peak.
                This cannot be overwritten at this time.''')

    # Negate signal if to analyze trough-centered cycles
    if center_extrema == 'P':
        pass
    elif center_extrema == 'T':
        x = -x
    else:
        raise ValueError('Parameter "center_extrema" must be either "P" or "T"')

    # Find peak and trough locations in the signal
    Ps, Ts = find_extrema(x, Fs, f_range, **find_extrema_kwargs)

    # Find zero-crossings
    zeroxR, zeroxD = find_zerox(x, Ps, Ts)

    # For each cycle, identify the sample of each extrema and zerocrossing
    shape_features = {}
    shape_features['sample_peak'] = Ps[1:]
    shape_features['sample_zerox_decay'] = zeroxD[1:]
    shape_features['sample_zerox_rise'] = zeroxR
    shape_features['sample_last_trough'] = Ts[:-1]
    shape_features['sample_next_trough'] = Ts[1:]

    # Compute duration of period
    shape_features['period'] = shape_features['sample_next_trough'] - \
        shape_features['sample_last_trough']

    # Compute duration of peak
    shape_features['time_peak'] = shape_features['sample_zerox_decay'] - \
        shape_features['sample_zerox_rise']

    # Compute duration of last trough
    shape_features['time_trough'] = zeroxR - zeroxD[:-1]

    # Determine extrema voltage
    shape_features['volt_peak'] = x[Ps[1:]]
    shape_features['volt_trough'] = x[Ts[:-1]]

    # Determine rise and decay characteristics
    shape_features['time_decay'] = (Ts[1:] - Ps[1:])
    shape_features['time_rise'] = (Ps[1:] - Ts[:-1])

    shape_features['volt_decay'] = x[Ps[1:]] - x[Ts[1:]]
    shape_features['volt_rise'] = x[Ps[1:]] - x[Ts[:-1]]
    shape_features['volt_amp'] = (shape_features['volt_decay'] + shape_features['volt_rise']) / 2

    # Comptue rise-decay symmetry features
    shape_features['time_rdsym'] = shape_features['time_rise'] / shape_features['period']

    # Compute peak-trough symmetry features
    shape_features['time_ptsym'] = shape_features['time_peak'] / (shape_features['time_peak'] + shape_features['time_trough'])

    # Compute average oscillatory amplitude estimate during cycle
    amp = amp_by_time(x, Fs, f_range, hilbert_increase_N=hilbert_increase_N, filter_kwargs={'N_cycles': 3})
    shape_features['band_amp'] = [np.mean(amp[Ts[i]:Ts[i + 1]]) for i in range(len(shape_features['sample_peak']))]

    # Convert feature dictionary into a DataFrame
    df = pd.DataFrame.from_dict(shape_features)

    # Define whether or not each cycle is part of a burst
    if burst_detection_method == 'cycles':
        df = detect_bursts_cycles(df, x, **burst_detection_kwargs)
    elif burst_detection_method == 'amp':
        df = detect_bursts_df_amp(df, x, Fs, f_range, **burst_detection_kwargs)
    else:
        raise ValueError('Invalid entry for "burst_detection_method"')

    # Rename columns if they are actually trough-centered
    if center_extrema == 'T':
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
        df.rename(columns=rename_dict, inplace=True)

        # Need to reverse symmetry measures
        df['volt_peak'] = -df['volt_peak']
        df['volt_trough'] = -df['volt_trough']
        df['time_rdsym'] = 1 - df['time_rdsym']
        df['time_ptsym'] = 1 - df['time_ptsym']

    return df
