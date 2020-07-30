"""Utility functions for working with ByCycle DataFrames."""

import numpy as np

from bycycle.utils.checks import check_param

###################################################################################################
###################################################################################################

def limit_df(df, fs, start=None, stop=None):
    """Restrict dataframe to be within time limits.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe output of :func:`~.compute_features`.
    fs : float
        Sampling rate, in Hz.
    start : float, optional
        The lower time limit, in seconds, to restrict the df.
    stop : float, optional
        The upper time limit, in seconds, to restrict the df.

    Returns
    -------
    df : pandas.DataFrame
        A limited dataframe of cycle features.

    Notes
    -----
    Cycles, or rows in the `df`, are included if any segment of the cycle falls after the
    `stop` time or before the `end` time.
    """

    # Ensure arguments are within valid range
    check_param(fs, 'fs', (0, np.inf))
    check_param(start, 'start', (0, stop))
    check_param(stop, 'stop', (start, np.inf))

    center_e, side_e = get_extrema_df(df)

    start = 0 if start is None else start

    df = df[df['sample_next_' + side_e].values >= start*fs]

    if stop is not None:
        df = df[df['sample_last_' + side_e].values < stop*fs]

    # Shift sample indices to start at 0
    df['sample_last_' + side_e] = df['sample_last_' + side_e] - int(fs * start)
    df['sample_next_' + side_e] = df['sample_next_' + side_e] - int(fs * start)
    df['sample_' + center_e] = df['sample_' + center_e] - int(fs * start)
    df['sample_zerox_rise'] = df['sample_zerox_rise'] - int(fs * start)
    df['sample_zerox_decay'] = df['sample_zerox_decay'] - int(fs * start)

    return df


def get_extrema_df(df):
    """Determine whether cycles are peak or trough centered.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe output of :func:`~.compute_features`.

    Returns
    -------
    center_e : str
        Center extrema, either 'peak' or 'trough'.
    side_e : str
        Side extrema, either 'peak' or 'trough'.
    """

    center_e = 'peak' if 'sample_peak' in df.columns else 'trough'
    side_e = 'trough' if center_e == 'peak' else 'peak'

    return center_e, side_e


def rename_extrema_df(center_extrema, df_samples, df_features):
    """Rename a dataframe based on the centered extrema.

    Parameters
    ----------
    center_extrema : {'trough', 'peak'}
        Which extrema is centered.
    df_samples, df_features : pandas.DataFrames
        ByCycle dataframes to rename, given the centered extrema.

    Returns
    -------
    df_features, df_samples : pandas.DataFrames
        Updated dataframes.
    """

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
