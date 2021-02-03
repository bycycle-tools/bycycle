"""Utility functions for working with ByCycle DataFrames."""

from itertools import product

import numpy as np
import pandas as pd

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

    Examples
    --------
    Limit a samples dataframe to the first second of a simulated signal:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.features import compute_features
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_features = compute_features(sig, fs, f_range=(8, 12))
    >>> df_features = limit_df(df_features, fs, start=0, stop=1)
    """

    # Ensure arguments are within valid range
    check_param(fs, 'fs', (0, np.inf))
    check_param(start, 'start', (0, stop))
    check_param(stop, 'stop', (start, np.inf))

    center_e, side_e = get_extrema_df(df)

    start = 0 if start is None else start

    df = df[df['sample_last_' + side_e].values >= start*fs]

    if stop is not None:
        df = df[df['sample_next_' + side_e].values <= stop*fs]

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

    Examples
    --------
    Confirm that cycles are peak-centered:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.features import compute_features
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_features = compute_features(sig, fs, f_range=(8, 12), center_extrema='peak')
    >>> center_e, side_e = get_extrema_df(df_features)
    >>> center_e
    'peak'
    """

    center_e = 'peak' if 'sample_peak' in df.columns else 'trough'
    side_e = 'trough' if center_e == 'peak' else 'peak'

    return center_e, side_e


def rename_extrema_df(center_extrema, df_features, return_samples=True):
    """Rename a dataframe based on the centered extrema.

    Parameters
    ----------
    center_extrema : {'trough', 'peak'}
        Which extrema is centered.
    df_features : pandas.DataFrames
        Bycycle dataframes to rename, given the centered extrema.
    return_samples : bool, optional, default: True
        Whether to rename sample columns if ``returns_samples`` is True when computing
        ``df_features`` using :func:`~.compute_features`.

    Returns
    -------
    df_features : pandas.DataFrames
        Updated dataframes.

    Examples
    --------
    Convert the column labels of a peak-centered dataframe to a trough-centered dataframe:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.features import compute_features
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> sig = -sig  # invert the signal, flipping peaks and troughs
    >>> df_features = compute_features(sig, fs, f_range=(8, 12), center_extrema='peak')
    >>> df_features = rename_extrema_df('trough', df_features)
    """

    # Rename columns if they are actually trough-centered
    if center_extrema == 'trough':

        features_rename_dict = {'time_peak': 'time_trough',
                                'time_trough': 'time_peak',
                                'volt_peak': 'volt_trough',
                                'volt_trough': 'volt_peak',
                                'time_rise': 'time_decay',
                                'time_decay': 'time_rise',
                                'volt_rise': 'volt_decay',
                                'volt_decay': 'volt_rise'}

        df_features.rename(columns=features_rename_dict, inplace=True)

        # Need to reverse symmetry measures
        df_features['volt_peak'] = -df_features['volt_peak']
        df_features['volt_trough'] = -df_features['volt_trough']
        df_features['time_rdsym'] = 1 - df_features['time_rdsym']
        df_features['time_ptsym'] = 1 - df_features['time_ptsym']

        if return_samples:

            samples_rename_dict = {'sample_peak': 'sample_trough',
                                   'sample_zerox_decay': 'sample_zerox_rise',
                                   'sample_zerox_rise': 'sample_zerox_decay',
                                   'sample_last_zerox_decay': 'sample_last_zerox_rise',
                                   'sample_last_trough': 'sample_last_peak',
                                   'sample_next_trough': 'sample_next_peak'}

            df_features.rename(columns=samples_rename_dict, inplace=True)

    return df_features


def split_samples_df(df_features):
    """Move cyclepoints sample indices columns to a separate dataframe.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe output of :func:`~.compute_features` or :func`~.compute_shape_features`.

    Returns
    -------
    df_features : pandas.DataFrame
        A dataframe without sample indices columns removed.
    df_samples : pandas.DataFrame
        A dataframe only containing sample indices columns.

    Examples
    --------
    Separate sample/signal indices into a separate dataframe:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.features import compute_features
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_features = compute_features(sig, fs, f_range=(8, 12), center_extrema='peak')
    >>> df_features, df_samples = split_samples_df(df_features)
    """

    df_samples = pd.concat([df_features.pop(col) for col in df_features.columns.values \
        if col.startswith('sample_')], axis=1)

    return df_features, df_samples


def drop_samples_df(df_features):
    """Remove cyclepoints sample columns from a dataframe.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe output of :func:`~.compute_features` or :func`~.compute_shape_features`.

    Returns
    -------
    df_features : pandas.DataFrame
        A dataframe without sample indices columns removed.

    Examples
    --------
    Drop cyclepoint sample columns from a dataframe:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.features import compute_features
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_features = compute_features(sig, fs, f_range=(8, 12), center_extrema='peak')
    >>> df_features = drop_samples_df(df_features)
    """

    sample_columns = [col for col in df_features.columns if col.startswith('sample_')]
    df_features = df_features.drop(sample_columns, axis=1)

    return df_features


def epoch_df(df_features, sig_len, epoch_len):
    """Reshape a dataframe into a list of dataframes.

    Parameters
    ----------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
    sig_len : int
        Length of a 1D time series.
    epoch_len : int
        The length of each epoched data in units of signal samples.

    Returns
    -------
    dfs_features : list of pd.DataFrame
        A list of features dataframes that have been epoched.

    Examples
    --------
    Epoch a dataframe in 1 seconds intervals:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.features import compute_features
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_features = compute_features(sig, fs, f_range=(8, 12), center_extrema='peak')
    >>> dfs_features = epoch_df(df_features, len(sig), fs)
    """

    # Reshape the dataframe into original sigs shape
    center_extrema, _ = get_extrema_df(df_features)
    last_sample = 'sample_next_trough' if center_extrema == 'peak' else 'sample_next_peak'

    dfs_features = []
    sig_last_idxs = np.arange(epoch_len, sig_len + epoch_len, epoch_len)
    sig_first_idxs = np.append(0, sig_last_idxs[:-1])

    for first_idx, last_idx in zip(sig_first_idxs, sig_last_idxs):

        # Get the range for each df
        idx_range = np.where((df_features[last_sample].values <= last_idx) & \
                                (df_features[last_sample].values > first_idx))[0]

        df_single = df_features.iloc[idx_range]
        df_single.reset_index(drop=True, inplace=True)

        # Shift sample indices
        sample_cols = [col for col in df_single.columns if 'sample_' in col]

        for col in sample_cols:
            df_single[col] = df_single[col] - first_idx

        dfs_features.append(df_single)

    return dfs_features


def flatten_dfs(dfs_features, labels, column_name='Label'):
    """Flatten a list of dataframes into a single dataframe with a group column(s).

    Parameters
    ----------
    dfs_features : 1D or 2D list of pd.DataFrames
        List of dataframes returned from `~.compute_features_2D` or `~.compute_features_3D`.
    labels : 1D or 2D list
        List of group labels to append to the final dataframe.
    column_name : str, optional, default: 'Label'
        The name of the column used to identify sub-dataframes.

    Returns
    -------
    df_features : pd.DataFrame
        A single dataframe containing 1 or 2 group columns.

    Examples
    --------
    Flatten an epoched a dataframe:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.features import compute_features
    >>> from bycycle.utils.dataframes import epoch_df
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_features = compute_features(sig, fs, f_range=(8, 12), center_extrema='peak')
    >>> dfs_features = epoch_df(df_features, len(sig), fs)
    >>> df_features = flatten_dfs(dfs_features, ["{sec}s".format(sec=sec) for sec in range(10)])
    """

    labels = np.array(labels) if isinstance(labels, list) else labels
    labels = labels.flatten()

    if isinstance(dfs_features[0], pd.DataFrame):

        if len(labels) != len(dfs_features):
            raise ValueError("The labels and dfs_features must be the same size.")

        # Add labels
        for idx, df in enumerate(dfs_features):
            df[column_name] = labels[idx]

        # Flatten
        df_features = pd.concat(dfs_features, axis=0)

    elif isinstance(dfs_features[0][0], pd.DataFrame):

        dim0_len = len(dfs_features)
        dim1_len = len(dfs_features[0])

        if len(labels) != dim0_len * dim1_len:
            raise ValueError("The labels and dfs_features must be the same size.")

        # Add labels
        for idx, (dim0, dim1) in enumerate(product(range(dim0_len), range(dim1_len))):
            dfs_features[dim0][dim1][column_name] = labels[idx]

        # Flatten
        df_features = pd.concat([df for dfs in dfs_features for df in dfs])

    return df_features
