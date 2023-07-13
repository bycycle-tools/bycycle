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

        sample_cols = [col for col in list(df_features.columns) if "sample_" in col]
        assert len(sample_cols) == 6

    # Assert that np.nan isn't in dataframe columns
    for column in df_features.keys():

        assert not np.isnan(df_features[column].iloc[1:-1]).any()
