"""Compute spike shape features."""

import numpy as np
import pandas as pd

###################################################################################################
###################################################################################################

def compute_shape_features(df_features, sig, center='trough'):
    """Compute shape features for each spike.

    Parameters
    ---------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
    sig : 1d array
        Voltage time series.
    center : {'trough', 'peak'}
        Center extrema of the spike.

    Returns
    -------
    df_shape_features : pd.DataFrame
        Dataframe containing spike shape features. Each row is one cycle. Columns:

        - time_decay : time between trough and start
        - time_rise : time between trough and next peak
        - time_decay_sym : fraction of cycle in the first decay period
        - time_rise_sym : fraction of cycle in the rise period
        - time_next_decay_sym : fraction of the cycle in the second decay period
        - volt_trough : Voltage at the trough.
        - volt_peak : Voltage at the next peak.
        - volt_last_rise : Voltage at the start of the spike.
        - volt_decay : Voltage at the decay before the trough.
        - volt_rise : Voltage at the rise after the trough.
        - volt_next_decay : Voltage at the decay after the peak.
        - period : The period of each spike.
        - time_trough : Time between zero-crossings adjacent to trough.
        - time_peak : Time between zero-crossings adjacent to peak.

    """

    # Compute durations of period, peaks, and troughs
    period, time_trough, time_peak = compute_durations(df_features)

    # Compute extrema and zero-crossing voltage
    volt_trough, volt_peak, volt_last_rise, volt_decay, volt_rise, volt_next_decay = \
        compute_voltages(df_features, sig)

    # Compute symmetry characteristics
    sym_features = compute_symmetry(df_features, period, time_trough, time_peak)

    # Organize shape features into a dataframe
    shape_features = {}
    shape_features['period'] = period
    shape_features['time_trough'] = time_trough
    shape_features['time_peak'] = time_peak

    shape_features['volt_trough'] = volt_trough
    shape_features['volt_peak'] = volt_peak
    shape_features['volt_last_rise'] = volt_last_rise
    shape_features['volt_decay'] = volt_decay
    shape_features['volt_rise'] = volt_rise
    shape_features['volt_next_decay'] = volt_next_decay

    shape_features['time_decay'] = sym_features['time_decay']
    shape_features['time_rise'] = sym_features['time_rise']
    shape_features['time_next_decay'] = sym_features['time_next_decay']
    shape_features['time_decay_sym'] = sym_features['time_decay_sym']
    shape_features['time_rise_sym'] = sym_features['time_rise_sym']
    shape_features['time_next_decay_sym'] = sym_features['time_next_decay_sym']

    df_shape_features = pd.DataFrame.from_dict(shape_features)

    return df_shape_features


def compute_symmetry(df_features, period=None, time_trough=None, time_peak=None):
    """Compute symmetry characteristics.

    Parameters
    ---------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
    period : 1d array, optional, default: None
        The period of each spike.
    time_trough : 1d array, optional, default: None
        Time between zero-crossings adjacent to trough.
    time_peak : 1d array, optional, default: None
        Time between zero-crossings adjacent to peak.

    Returns
    -------
    sym_features : dict
        Contains 1d arrays of symmetry features. Trough-centered key definitions:

        - time_decay : time between trough and first peak
        - time_rise : time between peak and trough
        - time_decay_sym : fraction of cycle in the first decay period
        - time_rise_sym : fraction of cycle in the rise period
        - time_next_decay_sym : fraction of the cycle in the second decay period

    """

    # Determine rise and decay characteristics
    sym_features = {}

    if period is None or time_peak is None or time_trough is None:
        period, time_trough, time_peak = compute_durations(df_features)

    time_decay =  df_features['sample_trough'] - df_features['sample_start']
    time_rise = df_features['sample_next_peak'] - df_features['sample_trough']
    time_next_decay = df_features['sample_next_decay'] - df_features['sample_next_peak']

    time_decay_sym = time_decay.values / period
    time_rise_sym = (time_peak - time_trough) / period
    time_next_decay_sym = (period - time_peak) / period

    sym_features['time_decay'] = time_decay.values.astype('int')
    sym_features['time_rise'] = time_rise.values.astype('int')
    sym_features['time_next_decay'] = time_next_decay.values.astype('int')
    sym_features['time_decay_sym'] = time_decay_sym
    sym_features['time_rise_sym'] = time_rise_sym
    sym_features['time_next_decay_sym'] = time_next_decay_sym

    return sym_features


def compute_voltages(df_features, sig):
    """Compute the voltage of extrema and zero-crossings.

    Parameters
    ---------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
    sig : 1d array
        Voltage time series.

    Returns
    -------
    volt_trough : 1d array
        Voltage at the trough.
    volt_peak : 1d array
        Voltage at the next peak.
    volt_last_rise : 1d array
        Voltage at the start of the spike.
    volt_decay : 1d array
        Voltage at the decay before the trough.
    volt_rise : 1d array
        Voltage at the rise after the trough.
    volt_next_decay : 1d array
        Voltage at the decay after the peak.
    """

    volt_trough = sig[df_features['sample_trough'].values]
    volt_peak = sig[df_features['sample_next_peak'].values]
    volt_last_rise = sig[df_features['sample_start'].values]
    volt_decay = sig[df_features['sample_decay'].values]
    volt_rise = sig[df_features['sample_rise'].values]
    volt_next_decay = sig[df_features['sample_next_decay'].values]

    return volt_trough, volt_peak, volt_last_rise, volt_decay, volt_rise, volt_next_decay


def compute_durations(df_features):
    """Compute the time durations of periods, peaks, and troughs.

    Parameters
    ---------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.

    Returns
    -------
    period : 1d array
        The period of each spike.
    time_trough : 1d array
        Time between zero-crossings adjacent to trough.
    time_peak : 1d array
        Time between zero-crossings adjacent to peak.
    """

    period = df_features['sample_end'] - df_features['sample_start'] + 1
    time_trough = df_features['sample_rise'] - df_features['sample_decay']
    time_peak = df_features['sample_next_decay'] - df_features['sample_rise']

    period = period.values.astype('int')
    time_trough = time_trough.values.astype('int')
    time_peak = time_peak.values.astype('int')

    return period, time_trough, time_peak
