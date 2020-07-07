"""Tests the cycle-by-cycle burst feature computation function."""

import numpy as np
import pytest

from bycycle.features import compute_features

###################################################################################################
###################################################################################################


@pytest.mark.parametrize("return_samples", [True, False])
def test_features_features(sim_args, return_samples):

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    if return_samples:

        df_features, df_samples = compute_features(sig, fs, f_range, return_samples=return_samples)

        assert len(df_features) == len(df_samples)

    else:

        df_features = compute_features(sig, fs, f_range, return_samples=return_samples)

    # Assert that np.nan isn't in dataframe columns
    for _, column in df_features.iteritems():

        assert not np.isnan(column[1:-1]).any()

    if return_samples:

        for idx, row in df_samples.iterrows():

            assert not np.isnan(row).any()
