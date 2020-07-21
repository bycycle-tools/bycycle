"""Tests the cycle-by-cycle burst feature computation function."""

import pytest

import numpy as np

from bycycle.features import compute_features

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("burst_detection_method",
    [
        'cycles',
        'amplitude',
        pytest.param(None, marks=pytest.mark.xfail)
    ]
)
@pytest.mark.parametrize("return_samples", [True, False])
def test_features_features(sim_args, return_samples, burst_detection_method):

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    # Set burst detection kwargs
    if burst_detection_method == 'amplitude':

        burst_threshold_kwargs = {'burst_fraction_threshold': 1, 'n_cycles_min': 3}

    else:

        burst_threshold_kwargs = sim_args['burst_threshold_kwargs']

    # Test returning sample indices in a separate dataframe.
    if return_samples:

        df_features, df_samples = compute_features(sig, fs, f_range, \
            burst_detection_method=burst_detection_method, \
            burst_threshold_kwargs=burst_threshold_kwargs, return_samples=return_samples)

        assert len(df_features) == len(df_samples)

    else:

        df_features = compute_features(sig, fs, f_range, return_samples=return_samples)

    # Assert that np.nan isn't in dataframe columns
    for _, column in df_features.iteritems():

        assert not np.isnan(column[1:-1]).any()

    if return_samples:

        for _, row in df_samples.iterrows():

            assert not np.isnan(row).any()