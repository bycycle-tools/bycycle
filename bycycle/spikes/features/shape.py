"""Compute spike shape features."""

import pandas as pd

###################################################################################################
###################################################################################################


def compute_shape_features(df_samples, sig, center='trough'):
    """Compute shape features for each spike.

    Parameters
    ---------
    df_samples : pandas.DataFrame
        Contains cycle points locations for each spike.
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
        - volt_trough : Voltage at the trough.
        - volt_last_peak : Voltage at the last peak.
        - volt_next_peak : Voltage at the next peak.
        - volt_decay : Voltage at the decay before the trough.
        - volt_rise : Voltage at the rise after the trough.
        - period : The period of each spike.
        - time_trough : Time between zero-crossings adjacent to trough.

    """

    # Compute durations
    period, time_trough = compute_durations(df_samples)

    # Compute extrema and zero-crossing voltage
    volts = compute_voltages(df_samples, sig)

    volt_trough, volt_last_peak, volt_next_peak, volt_decay, volt_rise,  = volts

    # Compute symmetry characteristics
    sym_features = compute_symmetry(df_samples)

    # Organize shape features into a dataframe
    shape_features = {}
    shape_features['period'] = period
    shape_features['time_trough'] = time_trough

    shape_features['volt_trough'] = volt_trough
    shape_features['volt_last_peak'] = volt_next_peak
    shape_features['volt_next_peak'] = volt_last_peak
    shape_features['volt_decay'] = volt_decay
    shape_features['volt_rise'] = volt_rise

    shape_features['time_decay'] = sym_features['time_decay']
    shape_features['time_rise'] = sym_features['time_rise']
    shape_features['time_decay_sym'] = sym_features['time_decay_sym']
    shape_features['time_rise_sym'] = sym_features['time_rise_sym']

    df_shape_features = pd.DataFrame.from_dict(shape_features)

    return df_shape_features


def compute_symmetry(df_samples):
    """Compute symmetry characteristics.

    Parameters
    ---------
    df_samples : pandas.DataFrame
        Contains cycle points locations for each spike.

    Returns
    -------
    sym_features : dict
        Contains 1d arrays of symmetry features. Trough-centered key definitions:

        - time_decay : time between trough and first peak
        - time_rise : time between peak and trough
        - time_decay_sym : fraction of cycle in the decay period
        - time_rise_sym : fraction of cycle in the rise period

    """

    # Determine rise and decay characteristics
    sym_features = {}

    time_decay =  df_samples['sample_trough'] - df_samples['sample_last_peak']
    time_rise = df_samples['sample_next_peak'] - df_samples['sample_trough']

    time_rise_sym = time_rise / (time_rise + time_decay)
    time_decay_sym = 1 - time_rise_sym

    sym_features['time_decay'] = time_decay.values.astype('int')
    sym_features['time_rise'] = time_rise.values.astype('int')
    sym_features['time_decay_sym'] = time_decay_sym
    sym_features['time_rise_sym'] = time_rise_sym

    return sym_features


def compute_voltages(df_samples, sig):
    """Compute the voltage of extrema and zero-crossings.

    Parameters
    ---------
    df_samples : pandas.DataFrame
        Contains cycle points locations for each spike.
    sig : 1d array
        Voltage time series.

    Returns
    -------
    volt_trough : 1d array
        Voltage at the trough.
    volt_last_peak : 1d array
        Voltage at the last peak.
    volt_next_peak : 1d array
        Voltage at the next peak.
    volt_decay : 1d array
        Voltage at the decay before the trough.
    volt_rise : 1d array
        Voltage at the rise after the trough.
    """

    volt_trough = sig[df_samples['sample_trough'].values]
    volt_last_peak = sig[df_samples['sample_last_peak'].values]
    volt_next_peak = sig[df_samples['sample_next_peak'].values]
    volt_decay = sig[df_samples['sample_decay'].values]
    volt_rise = sig[df_samples['sample_rise'].values]

    return volt_trough, volt_last_peak, volt_next_peak, volt_decay, volt_rise


def compute_durations(df_samples):
    """Compute the time durations for spikes.

    Parameters
    ---------
    df_samples : pandas.DataFrame
        Contains cycle points locations for each spike.

    Returns
    -------
    period : 1d array
        The period of each spike.
    time_trough : 1d array
        Time between zero-crossings adjacent to trough.
    time_peak : 1d array
        Time between zero-crossings adjacent to peak.
    """

    period = df_samples['sample_next_peak'] - df_samples['sample_last_peak'] + 1
    time_trough = df_samples['sample_rise'] - df_samples['sample_decay']

    period = period.values.astype('int')
    time_trough = time_trough.values.astype('int')

    return period, time_trough
