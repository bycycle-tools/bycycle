"""Tests burst detection."""

import os
import itertools
import numpy as np

from neurodsp.filt import  filter_signal

from bycycle.features import compute_shapes, compute_features
from bycycle.burst import detect_bursts_df_amp, detect_bursts_cycles

# Set data path
DATA_PATH = os.getcwd() + '/tutorials/data/'

###################################################################################################
###################################################################################################

def test_detect_bursts_cycles():
    """Test amplitude and period consistency burst detection."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_bursting.npy')

    fs = 1000
    f_range = (6, 14)

    sig_filt = filter_signal(sig, fs, 'lowpass', 30, n_seconds=.3, remove_edges=False)

    # Compute cycle shapes
    df_shapes = compute_shapes(sig_filt, fs, f_range)

    # Compute cycle features
    df_features = compute_features(df_shapes, sig)

    # Apply consistency burst detection for consistency detection
    burst_detection_kwargs = {'amplitude_fraction_threshold': 0.,
                              'amplitude_consistency_threshold': .5,
                              'period_consistency_threshold': .5,
                              'monotonicity_threshold': .5,
                              'n_cycles_min': 3}

    df_burst_cycles = detect_bursts_cycles(df_features, **burst_detection_kwargs)

    # Make sure that burst detection is only boolean
    assert df_burst_cycles.dtypes['is_burst'] == 'bool'
    assert df_burst_cycles['is_burst'].mean() >= 0
    assert df_burst_cycles['is_burst'].mean() <= 1
    assert np.min([sum(1 for _ in group) for key, group in \
        itertools.groupby(df_burst_cycles['is_burst']) if key]) >= 3


def test_detect_bursts_df_amp():
    """Test amplitude-threshold burst detection."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_bursting.npy')

    fs = 1000
    f_range = (6, 14)

    sig_filt = filter_signal(sig, fs, 'lowpass', 30, n_seconds=.3, remove_edges=False)

     # Compute cycle shapes
    df_shapes = compute_shapes(sig, fs, f_range)

    # Compute cycle features for dual threshold detection
    dual_threshold_kwargs = {'fs': fs, 'f_range':f_range, 'amp_threshes': (1, 2),
                             'n_cycles_min': 3, 'filter_kwargs': None}
    df_features = compute_features(df_shapes, sig, dual_threshold_kwargs=dual_threshold_kwargs)

    # Apply dual threshold burst detection
    df_features = detect_bursts_df_amp(df_features, burst_fraction_threshold=1, n_cycles_min=3)

    # Make sure that burst detection is only boolean
    assert df_features.dtypes['is_burst'] == 'bool'
    assert df_features['is_burst'].mean() >= 0
    assert df_features['is_burst'].mean() <= 1
    assert np.min([sum(1 for _ in group) for key, group \
        in itertools.groupby(df_features['is_burst']) if key]) >= 4
