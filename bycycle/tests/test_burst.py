"""Tests the burst detection functionality

NOTES
-----
The tests here are not strong tests for accuracy.
    They serve rather as 'smoke tests', for if anything fails completely.
"""

import numpy as np
from bycycle import burst, filt, features
import itertools
import os

# Set data path
data_path = '/'.join(os.path.dirname(bycycle.__file__).split('/')[:-1]) + '/tutorials/data/'


def test_detect_bursts_cycles():
    """Test amplitude and period consistency burst detection"""

    # Load signal
    signal = np.load(data_path + 'sim_bursting.npy')
    Fs = 1000  # Sampling rate
    f_range = (6, 14)  # Frequency range

    signal = filt.lowpass_filter(signal, Fs, 30, N_seconds=.3,
                                 remove_edge_artifacts=False)

    # Compute cycle-by-cycle df without burst detection column
    df = features.compute_features(signal, Fs, f_range,
                                   burst_detection_method='amp',
                                   burst_detection_kwargs={'amp_threshes': (1, 2),
                                                           'filter_kwargs': {'N_seconds': .5}})
    df.drop('is_burst', axis=1, inplace=True)

    # Apply consistency burst detection
    df_burst_cycles = burst.detect_bursts_cycles(df, signal)

    # Make sure that burst detection is only boolean
    assert df_burst_cycles.dtypes['is_burst'] == 'bool'
    assert df_burst_cycles['is_burst'].mean() > 0
    assert df_burst_cycles['is_burst'].mean() < 1
    assert np.min([sum(1 for _ in group) for key, group in itertools.groupby(df_burst_cycles['is_burst']) if key]) >= 3


def test_detect_bursts_df_amp():
    """Test amplitde-threshold burst detection"""

    # Load signal
    signal = np.load(data_path + 'sim_bursting.npy')
    Fs = 1000  # Sampling rate
    f_range = (6, 14)  # Frequency range
    signal = filt.lowpass_filter(signal, Fs, 30, N_seconds=.3,
                                 remove_edge_artifacts=False)

    # Compute cycle-by-cycle df without burst detection column
    df = features.compute_features(signal, Fs, f_range,
                                   burst_detection_method='amp',
                                   burst_detection_kwargs={'amp_threshes': (1, 2),
                                                           'filter_kwargs': {'N_seconds': .5}})
    df.drop('is_burst', axis=1, inplace=True)

    # Apply consistency burst detection
    df_burst_amp = burst.detect_bursts_df_amp(df, signal, Fs, f_range,
                                              amp_threshes=(.5, 1),
                                              N_cycles_min=4, filter_kwargs={'N_seconds': .5})

    # Make sure that burst detection is only boolean
    assert df_burst_amp.dtypes['is_burst'] == 'bool'
    assert df_burst_amp['is_burst'].mean() > 0
    assert df_burst_amp['is_burst'].mean() < 1
    assert np.min([sum(1 for _ in group) for key, group in itertools.groupby(df_burst_amp['is_burst']) if key]) >= 4
