"""Tests for burst.cycle."""

import itertools

import numpy as np

from bycycle.features import compute_burst_features

from bycycle.burst.cycle import *

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
