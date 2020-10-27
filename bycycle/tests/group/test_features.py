"""Test functions to compute features across epoched data."""

import numpy as np
import pandas as pd

from pytest import mark, param

from bycycle.group.features import compute_features_2d, compute_features_3d

###################################################################################################
###################################################################################################

@mark.parametrize("compute_features_kwargs_error", [False, param(True, marks=mark.xfail)])
@mark.parametrize("compute_features_kwargs_dtype", ['dict', 'list', None])
@mark.parametrize("global_features", [True, False])
def test_compute_features_2d(sim_args, compute_features_kwargs_dtype,
                             compute_features_kwargs_error, global_features):

    n_sigs = 5
    sigs = np.array([sim_args['sig']] * n_sigs)
    fs  = sim_args['fs']
    f_range = sim_args['f_range']

    # return_samples is disregarded when used in compute_features_kwargs
    #   this variable is set directly in the function
    compute_features_kwargs = {'center_extrema': 'peak', 'return_samples': False}

    if compute_features_kwargs_dtype == 'list' and compute_features_kwargs_error is False:
        compute_features_kwargs = [compute_features_kwargs] * n_sigs
    elif compute_features_kwargs_dtype == 'list' and compute_features_kwargs_error is True:
        compute_features_kwargs = [compute_features_kwargs] * 2
    elif compute_features_kwargs_dtype == None:
         compute_features_kwargs = None

    # Sequential processing check
    features_seq = compute_features_2d(sigs, fs, f_range, n_jobs=1, return_samples=True,
                                       compute_features_kwargs=compute_features_kwargs,
                                       global_features=global_features)

    # Parallel processing check
    if global_features is False:

        features = compute_features_2d(sigs, fs, f_range, n_jobs=-1, return_samples=True,
                                    compute_features_kwargs=compute_features_kwargs)


        # Assert that sequential and parallel processing is equivalent
        for idx, df_features in enumerate(features):
            assert df_features.equals(features_seq[idx])

    elif global_features is True:

        # When using global features, edge artifacts will exist in the first/last df
        for df_features in features_seq[2:-1]:

            # This fixes float precision issues with band_amp
            pd.testing.assert_frame_equal(features_seq[1], df_features)


        # Edge artifacts will cause the first/last dataframes to differ
        assert not features_seq[0].equals(features_seq[-1])
        assert not features_seq[0].equals(features_seq[1])
        assert not features_seq[-1].equals(features_seq[1])


@mark.parametrize("compute_features_kwargs_error", [False, param(True, marks=mark.xfail)])
@mark.parametrize("compute_features_kwargs_dtype", ['dict', '1dlist', '2dlist', None])
@mark.parametrize("return_samples", [True, False])
def test_compute_features_3d(sim_args, compute_features_kwargs_error,
                             compute_features_kwargs_dtype, return_samples):

    dim1 = 3
    dim2 = 2

    sigs_2d = np.array([sim_args['sig']] * dim2)
    sigs_3d = np.array([sigs_2d] * dim1)

    fs = sim_args['fs']
    f_range = sim_args['f_range']

    compute_features_kwargs = {'center_extrema': 'peak'}

    # 1d list of kwargs dicts
    if compute_features_kwargs_dtype == '1dlist':

        if compute_features_kwargs_error is True:
            # Mismatch dimension, error expected
            compute_features_kwargs = [compute_features_kwargs] * (dim1 - 1)
        else:
            # Valid kwargs
            compute_features_kwargs = [compute_features_kwargs] * dim1

    # 2d list of kwargs dicts
    elif compute_features_kwargs_dtype == '2dlist':

        if compute_features_kwargs_error is True:
            # Mismatch dimension, error expected
            compute_features_kwargs = [compute_features_kwargs] * (dim2 - 1)
        else:
            # Valid kwargs
            compute_features_kwargs = [compute_features_kwargs] * dim2

        # Add 2d
        compute_features_kwargs = [compute_features_kwargs] * dim1

    # No kwargs passed
    elif compute_features_kwargs_dtype == None:
        compute_features_kwargs = None


    df_features = \
        compute_features_3d(sigs_3d, fs, f_range, compute_features_kwargs=compute_features_kwargs,
                            return_samples=return_samples, n_jobs=-1, progress=None)

    # Check lengths
    assert len(df_features) == dim1
    assert len(df_features[0]) == dim2

    # Check equal values
    for row_idx in range(dim1):
        for col_idx in range(dim2):

            assert df_features[row_idx][col_idx].equals(df_features[0][0])

