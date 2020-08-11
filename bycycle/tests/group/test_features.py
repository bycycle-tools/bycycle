"""Test functions to compute features across epoched data."""

import time

import numpy as np

from pytest import mark, param

from bycycle.group.features import compute_features_2d

###################################################################################################
###################################################################################################

@mark.parametrize("compute_features_kwargs_error", [False, param(True, marks=mark.xfail)])
@mark.parametrize("compute_features_kwargs_dtype", ['dict', 'list', None])
def test_compute_features_2d(sim_args, compute_features_kwargs_dtype,
                             compute_features_kwargs_error):

    n_sigs = 5
    sigs = np.array([sim_args['sig']] * n_sigs)
    fs  = sim_args['fs']
    f_range = sim_args['f_range']

    # return_samples is disregarded when used in compute_features_kwargs.
    #   This variable is set directly in the function.
    compute_features_kwargs = {'center_extrema': 'peak', 'return_samples': False}

    if compute_features_kwargs_dtype == 'list' and compute_features_kwargs_error is False:
        compute_features_kwargs = [compute_features_kwargs] * n_sigs
    elif compute_features_kwargs_dtype == 'list' and compute_features_kwargs_error is True:
        compute_features_kwargs = [compute_features_kwargs] * 2
    elif compute_features_kwargs_dtype == None:
         compute_features_kwargs = None

    # Test returning only features, without samples
    features = compute_features_2d(sigs, fs, f_range, n_jobs=1, return_samples=False,
                                   compute_features_kwargs=compute_features_kwargs)

    for df_features in features:
        assert df_features.equals(features[0])

    # Sequential processing check
    features_seq, samples_seq = \
        compute_features_2d(sigs, fs, f_range, n_jobs=1, return_samples=True,
                            compute_features_kwargs=compute_features_kwargs)

    # Parallel processing check
    features, samples = compute_features_2d(sigs, fs, f_range, n_jobs=-1, return_samples=True,
                                            compute_features_kwargs=compute_features_kwargs)

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
