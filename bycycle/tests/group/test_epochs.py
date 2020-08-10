"""Test functions to compute features across epoched data."""

import time

import numpy as np

import pytest

from bycycle.group.epochs import compute_features_2d

###################################################################################################
###################################################################################################

def test_compute_features_2d(sim_args):

    sigs = np.array([sim_args['sig']] * 50)
    fs  = sim_args['fs']
    f_range = sim_args['f_range']

    # Test returning only features, without samples
    features = compute_features_2d(sigs, fs, f_range, n_jobs=1,
                                   compute_features_kwargs={'return_samples': False})

    for df_features in features:
        assert df_features.equals(features[0])

    # Sequential processing check
    features_seq, samples_seq = \
        compute_features_2d(sigs, fs, f_range, n_jobs=1,
                            compute_features_kwargs={'return_samples': True})

    # Parallel processing check
    features, samples = compute_features_2d(sigs, fs, f_range, n_jobs=-1,
                                            compute_features_kwargs={'return_samples': True})

    # Since the same signal is used, check that each df is the same
    for df_features in features_seq:
        assert df_features.equals(features_seq[0])
    for df_features in features:
        assert df_features.equals(features[0])
    for df_samples in samples_seq:
        assert df_samples.equals(samples_seq[0])
    for df_samples in samples:
        assert df_samples.equals(samples[0])

    # Assert that sequential and parallel processing is equivalent
    for idx, df_features in enumerate(features):
        assert df_features.equals(features_seq[idx])

    for idx, df_samples in enumerate(samples):
        assert df_samples.equals(samples_seq[idx])
