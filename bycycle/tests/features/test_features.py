"""Tests for features.features."""

import pytest

import numpy as np

from bycycle.features import *

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("burst_method",
    [
        'cycles',
        'amp',
        pytest.param(None, marks=pytest.mark.xfail)
    ]
)
@pytest.mark.parametrize("return_samples", [True, False])
def test_compute_features(sim_args, return_samples, burst_method):

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    # Set burst detection kwargs
    if burst_method == 'amp':

        threshold_kwargs = {'burst_fraction_threshold': 1, 'min_n_cycles': 3}

    else:

        threshold_kwargs = sim_args['threshold_kwargs']

    # Test returning sample indices in a separate dataframe
    df_features = compute_features(sig, fs, f_range, burst_method=burst_method, \
        threshold_kwargs=threshold_kwargs, return_samples=return_samples)

    if return_samples:

        df_features, df_samples = df_features[0], df_features[1]
        assert len(df_features) == len(df_samples)

    # Assert that np.nan isn't in dataframe columns
    for _, column in df_features.iteritems():

        assert not np.isnan(column[1:-1]).any()

    if return_samples:

        for _, row in df_samples.iterrows():

            assert not np.isnan(row).any()
