"""Tests burst detection."""

import itertools

import numpy as np

from bycycle.features import compute_burst_features
from bycycle.burst import detect_bursts_amp, detect_bursts_cycles

###################################################################################################
###################################################################################################

def test_detect_bursts_cycles(sim_args):

    df_features = sim_args['df_features']
    threshold_kwargs = sim_args['threshold_kwargs']

    # Apply consistency burst detection
    df_burst_cycles = detect_bursts_cycles(df_features, **threshold_kwargs)

    # Make sure that burst detection is only boolean
    assert df_burst_cycles.dtypes['is_burst'] == 'bool'
    assert df_burst_cycles['is_burst'].mean() >= 0
    assert df_burst_cycles['is_burst'].mean() <= 1
    assert np.min([sum(1 for _ in group) for key, group in \
        itertools.groupby(df_burst_cycles['is_burst']) if key]) >= 3


def test_detect_bursts_amp(sim_args):

    df_shape_features = sim_args['df_shapes']
    df_samples = sim_args['df_samples']
    sig = sim_args['sig']
    burst_kwarg = {'fs': sim_args['fs'], 'f_range': sim_args['f_range'], 'amp_threshes': (0.5, 1)}

    df_features = compute_burst_features(df_shape_features, df_samples, sig,
                                         burst_method='amp',
                                         burst_kwargs=burst_kwarg)

    # Apply dual threshold burst detection
    df_features = detect_bursts_amp(df_features, burst_fraction_threshold=1, n_cycles_min=3)

    # Make sure that burst detection is only boolean
    assert df_features.dtypes['is_burst'] == 'bool'
    assert df_features['is_burst'].mean() >= 0
    assert df_features['is_burst'].mean() <= 1
    assert np.min([sum(1 for _ in group) for key, group \
        in itertools.groupby(df_features['is_burst']) if key]) >= 3
